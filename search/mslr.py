"""Optional benchmark on a locally obtained Microsoft MSLR-WEB10K Fold 1."""
from itertools import groupby
from pathlib import Path
from statistics import mean
from time import perf_counter

import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_svmlight_file

from discoverylib.metrics import ndcg, paired_interval
from search.experiment import latency, runtime, write_json


def load(path, limit=None):
    features, labels, qids = load_svmlight_file(
        str(path), n_features=136, dtype=np.float32, query_id=True, zero_based=False,
    )
    order = np.argsort(qids, kind="stable")
    if limit is not None:
        chosen = np.random.default_rng(42).choice(np.unique(qids), min(limit, len(set(qids))), replace=False)
        order = order[np.isin(qids[order], chosen)]
    ordered_qids = qids[order]
    groups = [len(list(rows)) for _, rows in groupby(ordered_qids)]
    return features[order], labels[order].astype(np.int32), ordered_qids, groups


def run_mslr(fold, output, threads=2, max_train_queries=None):
    folder = Path(fold)
    x, y, train_ids, group = load(folder / "train.txt", max_train_queries)
    val_x, val_y, val_ids, val_group = load(folder / "vali.txt")
    test_x, test_y, test_ids, test_group = load(folder / "test.txt")
    groups = [set(ids) for ids in (train_ids, val_ids, test_ids)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Query IDs overlap between supplied Fold 1 splits")
    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=120, learning_rate=0.1,
                           num_leaves=31, random_state=42, n_jobs=threads, verbosity=-1,
                           deterministic=True, force_col_wise=True, lambdarank_truncation_level=15)
    ranker.fit(x, y, group=group, eval_set=[(val_x, val_y)], eval_group=[val_group],
               eval_at=(10,), callbacks=[lgb.early_stopping(12, verbose=False)])
    baseline, learned, durations = {}, {}, []
    start = 0
    for count in test_group:
        stop = start + count
        qid = int(test_ids[start])
        docs = test_x[start:stop]
        grades = {did: int(grade) for did, grade in enumerate(test_y[start:stop])}
        # MSLR feature 108 is the precomputed title BM25 score.
        title_bm25 = docs[:, 107].toarray().ravel()
        tick = perf_counter()
        scores = ranker.booster_.predict(docs, num_threads=1)
        durations.append((perf_counter() - tick) * 1000)
        baseline[qid] = ndcg(list(np.argsort(-title_bm25, kind="stable")), grades)
        learned[qid] = ndcg(list(np.argsort(-scores, kind="stable")), grades)
        start = stop
    report = {
        "dataset": "MSLR-WEB10K Fold 1", "train_queries": len(group), "dev_queries": len(val_group),
        "test_queries": len(test_group), "bm25_title_ndcg@10": mean(baseline.values()),
        "lambdamart_ndcg@10": mean(learned.values()), "delta": paired_interval(baseline, learned),
        "prediction_latency_ms": latency(durations), "setup": runtime(threads),
    }
    write_json(output, report)
    print(report, flush=True)
