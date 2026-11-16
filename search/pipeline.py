"""Candidate generation, reciprocal rank fusion, and query-document features."""
import re
from collections import defaultdict

import numpy as np

from search.bm25 import BM25, tokenize

FEATURES = (
    "bm25_title", "bm25_text", "cosine", "title_overlap", "text_overlap",
    "text_length", "rrf_score", "lexical_rr", "dense_rr",
    "grade_offset", "difficulty_match", "type_match",
)
GRADE = re.compile(r"\b(?:grade|year)\s+(\d{1,2})\b|\b(\d{1,2})(?:rd|th) graders\b")
LEVEL = re.compile(r"\b(beginner|advanced)\b")
KIND = re.compile(r"\b(practice|lesson|quiz|project)\b")


def rrf(rankings, constant=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, did in enumerate(dict.fromkeys(ranking), start=1):
            scores[did] += 1 / (constant + rank)
    return dict(sorted(scores.items(), key=lambda pair: (-pair[1], pair[0])))


def query_constraints(query):
    query = query.lower()
    grade, level, kind = GRADE.search(query), LEVEL.search(query), KIND.search(query)
    return (
        int(grade.group(1) or grade.group(2)) if grade else 0,
        level.group(1) if level else "",
        kind.group(1) if kind else "",
    )


class Pipeline:
    def __init__(self, docs, bm25, encoder=None, dense=None, config=None):
        self.docs, self.bm25, self.encoder, self.dense = docs, bm25, encoder, dense
        self.config = config or {"k1": 1.2, "b": 0.75, "rrf_k": 60, "depth": 100}
        if dense is not None:
            self.title = BM25({did: d.title for did, d in docs.items()})
            self.body = BM25({did: d.text for did, d in docs.items()})
            self.title_terms = {did: set(tokenize(d.title)) for did, d in docs.items()}
            self.body_terms = {did: set(tokenize(d.text)) for did, d in docs.items()}

    def lexical(self, query, k=100):
        return self.bm25.search(query, k, self.config["k1"], self.config["b"])

    def grade_filtered(self, query, k=100):
        grade, _, _ = query_constraints(query)
        if not grade:
            return self.lexical(query, k)
        eligible = [did for did, doc in self.docs.items() if doc.grade == grade]
        return self.bm25.search(query, k, self.config["k1"], self.config["b"], eligible)

    def hybrid(self, query, vector, index="flat", k=100):
        lexical = self.lexical(query, k=self.config["depth"])
        semantic = self.dense.search(vector, k=self.config["depth"], index=index)
        fusion = rrf((lexical, semantic), self.config["rrf_k"])
        return list(fusion)[:k], fusion, lexical, semantic

    def feature_matrix(self, query, vector, candidate_set):
        ids, fusion, lexical, dense = candidate_set
        title_scores = self.title.scores(query, candidates=ids)
        body_scores = self.body.scores(query, candidates=ids)
        query_terms = set(tokenize(query))
        query_length = len(query_terms) or 1
        lexical_rr = {did: 1 / i for i, did in enumerate(lexical, 1)}
        dense_rr = {did: 1 / i for i, did in enumerate(dense, 1)}
        grade, level, kind = query_constraints(query)
        rows = []
        for did, cosine in zip(ids, self.dense.scores(vector, ids)):
            doc = self.docs[did]
            rows.append([
                title_scores.get(did, 0), body_scores.get(did, 0), cosine,
                len(query_terms & self.title_terms[did]) / query_length,
                len(query_terms & self.body_terms[did]) / query_length,
                self.body.lengths[did], fusion[did], lexical_rr.get(did, 0), dense_rr.get(did, 0),
                abs(grade - doc.grade) if grade and doc.grade else -1,
                int(level == doc.level) if level else -1,
                int(kind == doc.kind) if kind else -1,
            ])
        return np.asarray(rows, dtype=np.float32)
