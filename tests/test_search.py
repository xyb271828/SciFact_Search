import math

import pytest

from discoverylib.metrics import ndcg, paired_interval, relevance
from search.bm25 import BM25
from search.education import synthetic_lessons
from search.mslr import load
from search.pipeline import query_constraints, rrf


def test_bm25_postings_scores_and_top_k():
    index = BM25({"a": "cat cat and dog", "b": "cat and dog", "c": "the fox", "d": "the fox"})
    norm = 1 - 0.75 + 0.75 * 4 / 2.75
    expected = math.log(2) * 2 * 2.2 / (2 + 1.2 * norm)
    assert index.scores("cat")["a"] == pytest.approx(expected)
    assert index.search("cat", 1) == ["a"]
    assert index.search("fox") == ["c", "d"]
    assert index.search("absent") == []
    assert index.search("cat", candidates=["b", "d"]) == ["b"]


def test_rank_metrics_keep_missed_positive_in_ideal_order():
    assert ndcg(["b", "x"], {"a": 3, "b": 1}) == pytest.approx(1 / (7 + 1 / math.log2(3)))
    assert ndcg(["a", "b"], {"a": 3, "b": 1}) == 1
    assert relevance(["x", "b"], {"a": 3, "b": 1})["mrr@10"] == 0.5
    assert relevance(["x", "b"], {"a": 3, "b": 1})["recall@100"] == 0.5
    assert paired_interval({"q": 0.0}, {"q": 1.0}, rounds=10)["ci95"] == [1.0, 1.0]
    with pytest.raises(ValueError):
        paired_interval({"a": 0}, {"b": 1})


def test_rrf_deduplicates_and_favors_agreement():
    assert list(rrf((["a", "b", "b"], ["c", "b"]))) == ["b", "a", "c"]
    assert query_constraints("Year 5 beginner practice with parts of a whole") == (5, "beginner", "practice")
    assert query_constraints("worksheet for 3rd graders") == (3, "", "")


def test_synthetic_split_separates_topics_and_has_graded_targets():
    data = synthetic_lessons()
    tags = [{qid.rsplit("-", 1)[0] for qid in ids} for ids in data.splits.values()]
    assert len(data.docs) == 5000
    assert len(data.queries) == 300
    assert [len(ids) for ids in data.splits.values()] == [180, 60, 60]
    assert len(set.union(*tags)) == sum(map(len, tags))
    assert all(3 in judgment.values() for judgment in data.qrels.values())


def test_mslr_loader_keeps_whole_query_groups(tmp_path):
    fixture = tmp_path / "mslr.txt"
    fixture.write_text("0 qid:2 1:1 108:4\n3 qid:1 1:2 108:3\n2 qid:2 1:3 108:2\n")
    features, labels, qids, groups = load(fixture)
    assert qids.tolist() == [1, 2, 2]
    assert groups == [1, 2]
    assert labels.tolist() == [3, 0, 2]
    assert features[:, 107].toarray().ravel().tolist() == [3, 4, 2]


def test_shared_metrics_against_small_fixtures():
    from sklearn.metrics import ndcg_score

    from discoverylib.metrics import coverage, ctr_metrics

    # Convert graded judgments to exponential gains before the independent check.
    independent = ndcg_score([[7, 3, 1]], [[0.4, 0.9, 0.7]], k=2)
    assert ndcg(["b", "c", "a"], {"a": 3, "b": 2, "c": 1}, k=2) == pytest.approx(independent)
    assert coverage([["a", "a"], ["b", "other"]], {"a", "b", "c"}) == pytest.approx(2 / 3)
    assert ctr_metrics([0, 1], [0.5, 0.5], 0.5) == pytest.approx(
        {"auc": 0.5, "log_loss": math.log(2), "normalized_entropy": 1, "ece": 0}
    )
