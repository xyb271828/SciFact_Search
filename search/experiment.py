"""Reproducible evaluation for lexical, dense, hybrid, and learned search."""
import hashlib
import heapq
import json
import platform
from pathlib import Path
from statistics import mean
from time import perf_counter

import faiss
import lightgbm as lgb
import numpy as np

from discoverylib.metrics import ndcg, paired_interval, percentile, relevance
from search.bm25 import BM25, tokenize
from search.data import scifact
from search.dense import MODEL, REVISION, DenseIndex, Encoder
from search.education import synthetic_lessons
from search.pipeline import FEATURES, Pipeline, query_constraints, rrf


def cpu_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def runtime(threads):
    return {"cpu": cpu_name(), "python": platform.python_version(), "model_threads": threads,
            "faiss_omp_threads": 1, "os": platform.platform()}


def latency(values):
    return {f"p{p}": percentile(values, p) for p in (50, 95, 99)}


def evaluate(queries, qrels, retrieve):
    for query in list(queries.values())[:10]:
        retrieve(query)
    rows, durations = {}, []
    for qid, query in queries.items():
        start = perf_counter()
        ranked = retrieve(query)
        durations.append((perf_counter() - start) * 1000)
        rows[qid] = relevance(ranked, qrels[qid])
    return {
        **{key: mean(row[key] for row in rows.values()) for key in next(iter(rows.values()))},
        "latency_ms": latency(durations),
        "per_query": rows,
    }


def prepare(data_dir, dataset, threads, lexical_only=False):
    faiss.omp_set_num_threads(1)
    data_dir = Path(data_dir)
    ds = scifact(data_dir / "scifact.zip") if dataset == "scifact" else synthetic_lessons()
    documents = {did: doc.full_text for did, doc in ds.docs.items()}
    indices = {analyzer: BM25(documents, analyzer) for analyzer in ("plain", "stem")}
    best = (-1.0, {})
    for analyzer, index in indices.items():
        for k1 in (0.9, 1.2, 1.6):
            for b in (0.4, 0.75):
                score = mean(ndcg(index.search(ds.queries[qid], 10, k1, b), ds.qrels[qid])
                             for qid in ds.splits["dev"])
                if score > best[0]:
                    best = score, {"analyzer": analyzer, "k1": k1, "b": b}
    config = best[1] | {"rrf_k": 60, "depth": 100}
    if lexical_only:
        return ds, Pipeline(ds.docs, indices[config["analyzer"]], config=config), indices["plain"]

    encoder = Encoder(data_dir / "model", threads)
    dense = DenseIndex(ds.docs, encoder, data_dir)
    return ds, Pipeline(ds.docs, indices[config["analyzer"]], encoder, dense, config), indices["plain"]


def reference_baseline(ds, plain):
    from rank_bm25 import BM25Okapi

    ids = sorted(ds.docs)
    reference = BM25Okapi([tokenize(ds.docs[did].full_text) for did in ids], k1=1.2, b=0.75)

    def retrieve(query):
        terms = tokenize(query)
        candidates = set().union(*(plain.postings.get(t, {}) for t in terms))
        scores = reference.get_scores(terms)
        return [did for did, _ in heapq.nsmallest(
            100, ((did, score) for did, score in zip(ids, scores) if did in candidates),
            key=lambda pair: (-pair[1], pair[0]),
        )]

    return retrieve


def ndcg_scores(result):
    return {qid: scores["ndcg@10"] for qid, scores in result["per_query"].items()}


def rank_data(ds, pipeline, split):
    rows, labels, counts, candidates = [], [], [], {}
    qids = ds.splits[split]
    vectors = pipeline.encoder.encode([ds.queries[qid] for qid in qids])
    for i, qid in enumerate(qids):
        query, vector = ds.queries[qid], vectors[i:i + 1]
        selected = pipeline.hybrid(query, vector)
        ids = selected[0]
        candidates[qid] = ids
        counts.append(len(ids))
        rows.append(pipeline.feature_matrix(query, vector, selected))
        labels.extend(ds.qrels[qid].get(did, 0) for did in ids)
    return np.vstack(rows), np.asarray(labels), counts, candidates


def train_lessons(ds, pipeline, threads, ablate=False, prepared=None):
    prepared = prepared or (rank_data(ds, pipeline, "train"), rank_data(ds, pipeline, "dev"))
    train, dev = prepared
    columns = len(FEATURES) - 3 if ablate else len(FEATURES)
    train_x, train_y, train_groups, _ = train
    dev_x, dev_y, dev_groups, _ = dev
    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=120, learning_rate=0.08, num_leaves=15,
        min_child_samples=15, n_jobs=threads, random_state=42, verbosity=-1,
        deterministic=True, force_col_wise=True, lambdarank_truncation_level=15,
    )
    model.fit(
        train_x[:, :columns], train_y, group=train_groups,
        eval_set=[(dev_x[:, :columns], dev_y)], eval_group=[dev_groups], eval_at=(10,),
        feature_name=list(FEATURES[:columns]), callbacks=[lgb.early_stopping(12, verbose=False)],
    )
    return model.booster_


def score_candidates(pipeline, model, query, vector, selected):
    features = pipeline.feature_matrix(query, vector, selected)
    scores = model.predict(features[:, :model.num_feature()], num_threads=1)
    return [did for did, _ in sorted(zip(selected[0], scores), key=lambda pair: (-pair[1], pair[0]))]


def rank_query(pipeline, model, query, depth=100):
    vector = pipeline.encoder.encode([query])
    ids, fusion, lexical, dense = pipeline.hybrid(query, vector)
    reranked = score_candidates(pipeline, model, query, vector, (ids[:depth], fusion, lexical, dense))
    return reranked + ids[depth:]


def tune_rrf(ds, pipeline):
    ids = ds.splits["dev"]
    vectors = pipeline.encoder.encode([ds.queries[qid] for qid in ids])
    runs = {key: [] for key in (10, 30, 60)}
    for i, qid in enumerate(ids):
        query = ds.queries[qid]
        ranked = (pipeline.lexical(query), pipeline.dense.search(vectors[i:i + 1]))
        for constant in runs:
            ids_fused = list(rrf(ranked, constant))[:100]
            runs[constant].append(ndcg(ids_fused, ds.qrels[qid]))
    pipeline.config["rrf_k"] = max(runs, key=lambda key: mean(runs[key]))


def ann_study(ds, pipeline, queries):
    vectors = pipeline.encoder.encode(list(queries.values()))
    rows = {}
    flat = [pipeline.dense.search(vectors[i:i + 1]) for i in range(len(vectors))]
    for ef in (32, 64, 128, 256):
        pipeline.dense.hnsw.hnsw.efSearch = ef
        overlaps, recalls, durations = [], [], []
        for i, qid in enumerate(queries):
            start = perf_counter()
            approx = pipeline.dense.search(vectors[i:i + 1], index="hnsw")
            durations.append((perf_counter() - start) * 1000)
            overlaps.append(len(set(approx) & set(flat[i])) / len(flat[i]))
            recalls.append(relevance(approx, ds.qrels[qid])["recall@100"])
        rows[str(ef)] = {"ann_overlap@100": mean(overlaps), "relevance_recall@100": mean(recalls),
                         "vector_lookup_ms": latency(durations)}
    pipeline.dense.hnsw.hnsw.efSearch = 128
    durations = []
    for i in range(len(vectors)):
        start = perf_counter()
        pipeline.dense.search(vectors[i:i + 1])
        durations.append((perf_counter() - start) * 1000)
    return {"flat_vector_lookup_ms": latency(durations), "hnsw": rows}


def error_examples(ds, pipeline, results, queries):
    before = ndcg_scores(results["bm25_tuned"])
    after = ndcg_scores(results["rrf_flat"])
    ordered = sorted(queries, key=lambda qid: (after[qid] - before[qid], qid))
    groups = {
        "regressions": [qid for qid in ordered if after[qid] < before[qid]][:5],
        "improvements": [qid for qid in reversed(ordered) if after[qid] > before[qid]][:5],
    }

    def top_entries(ids, judgments):
        return [{"id": did, "title": ds.docs[did].title, "judged_grade": judgments.get(did)}
                for did in ids[:3]]

    examples = {}
    for group, qids in groups.items():
        examples[group] = []
        for qid in qids:
            query, judgments = queries[qid], ds.qrels[qid]
            vector = pipeline.encoder.encode([query])
            fused, _, lexical, dense = pipeline.hybrid(query, vector)
            examples[group].append(
                {"query_id": qid, "query": query, "delta": after[qid] - before[qid],
                 "lexical_top3": top_entries(lexical, judgments), "dense_top3": top_entries(dense, judgments),
                 "hybrid_top3": top_entries(fused, judgments),
                 "relevant_added": sorted({did for did in fused[:10] if judgments.get(did, 0)}
                                          - {did for did in lexical[:10] if judgments.get(did, 0)}),
                 "relevant_removed": sorted({did for did in lexical[:10] if judgments.get(did, 0)}
                                            - {did for did in fused[:10] if judgments.get(did, 0)})}
            )
    return examples


def benchmark(dataset, data_dir, output, threads=2, lexical_only=False):
    ds, pipeline, plain = prepare(data_dir, dataset, threads, lexical_only)
    queries = {qid: ds.queries[qid] for qid in ds.splits["test"]}
    retrievers = {"bm25_original": plain.search, "bm25_tuned": pipeline.lexical}
    if dataset == "scifact":
        retrievers["rank_bm25_reference"] = reference_baseline(ds, plain)
    if not lexical_only:
        tune_rrf(ds, pipeline)
        retrievers |= {
            "dense_flat": lambda q: pipeline.dense.search(pipeline.encoder.encode([q])),
            "rrf_flat": lambda q: pipeline.hybrid(q, pipeline.encoder.encode([q]))[0],
            "rrf_hnsw": lambda q: pipeline.hybrid(q, pipeline.encoder.encode([q]), index="hnsw")[0],
        }
        if dataset == "lessons":
            prepared = (rank_data(ds, pipeline, "train"), rank_data(ds, pipeline, "dev"))
            ranker = train_lessons(ds, pipeline, threads, prepared=prepared)
            ablation = train_lessons(ds, pipeline, threads, ablate=True, prepared=prepared)
            retrievers |= {
                "grade_filter_bm25": pipeline.grade_filtered,
                "lambdamart": lambda q: rank_query(pipeline, ranker, q),
                "lambdamart_depth30": lambda q: rank_query(pipeline, ranker, q, depth=30),
                "lambdamart_without_constraints": lambda q: rank_query(pipeline, ablation, q),
            }
            artifact_dir = Path(output).parent
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifact_dir / "lesson-ranker.txt"
            ranker.save_model(str(model_path))
            manifest = {
                "bm25_rrf": pipeline.config, "embedding_revision": REVISION, "synthetic_seed": 42,
                "documents": len(ds.docs), "features": list(FEATURES),
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            }
            (artifact_dir / "lesson-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    results = {}
    for name, callback in retrievers.items():
        results[name] = evaluate(queries, ds.qrels, callback)
        print(name, {k: v for k, v in results[name].items() if k != "per_query"}, flush=True)
    comparisons = {}
    for baseline, contender in (
        ("rank_bm25_reference", "bm25_original"), ("bm25_original", "bm25_tuned"),
        ("bm25_tuned", "rrf_flat"), ("rrf_flat", "lambdamart"),
        ("bm25_tuned", "lambdamart"), ("grade_filter_bm25", "lambdamart"),
        ("lambdamart_depth30", "lambdamart"), ("lambdamart_without_constraints", "lambdamart"),
    ):
        if baseline in results and contender in results:
            comparisons[contender + "_minus_" + baseline] = paired_interval(
                ndcg_scores(results[baseline]), ndcg_scores(results[contender])
            )
    report = {
        "dataset": dataset, "documents": len(ds.docs), "split_sizes": {k: len(v) for k, v in ds.splits.items()},
        "config": pipeline.config, "setup": runtime(threads), "results": results, "comparisons": comparisons,
    }
    if not lexical_only:
        report |= {"encoder": {"model": MODEL, "revision": REVISION},
                   "ann": ann_study(ds, pipeline, queries),
                   "error_examples": error_examples(ds, pipeline, results, queries)}
    if dataset == "lessons":
        subsets = {
            "explicit_difficulty": [qid for qid, q in queries.items() if query_constraints(q)[1]],
            "no_difficulty_given": [qid for qid, q in queries.items() if not query_constraints(q)[1]],
        }
        report["query_slices"] = {
            name: {"queries": len(ids), "ndcg@10": {mode: mean(ndcg_scores(result)[qid] for qid in ids)
                                                 for mode, result in results.items()}}
            for name, ids in subsets.items()
        }
    write_json(output, report)
    return report


def write_json(output, report):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
