"""Sparse lexical search with deterministic heap top-K."""
import heapq
import math
import re
from collections import Counter, defaultdict

from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

WORD = re.compile(r"[a-z0-9]+")
STEMMER = SnowballStemmer("english")


def tokenize(text, analyzer="plain"):
    terms = WORD.findall(text.lower())
    if analyzer == "stem":
        return [STEMMER.stem(term) for term in terms if term not in ENGLISH_STOP_WORDS]
    return terms


class BM25:
    def __init__(self, texts, analyzer="plain"):
        self.analyzer = analyzer
        self.postings = defaultdict(dict)
        self.lengths = {}
        for did, text in texts.items():
            terms = tokenize(text, analyzer)
            self.lengths[did] = len(terms)
            for term, count in Counter(terms).items():
                self.postings[term][did] = count
        self.avg_length = sum(self.lengths.values()) / max(1, len(self.lengths)) or 1.0
        self.idf = {
            term: math.log1p((len(texts) - len(matches) + 0.5) / (len(matches) + 0.5))
            for term, matches in self.postings.items()
        }

    def scores(self, query, k1=1.2, b=0.75, candidates=None):
        result = defaultdict(float)
        for term, query_count in Counter(tokenize(query, self.analyzer)).items():
            matches = self.postings.get(term, {})
            if candidates is not None:
                matches = {did: matches[did] for did in candidates if did in matches}
            for did, tf in matches.items():
                norm = 1 - b + b * self.lengths[did] / self.avg_length
                result[did] += query_count * self.idf[term] * tf * (k1 + 1) / (tf + k1 * norm)
        return result

    def search(self, query, k=100, k1=1.2, b=0.75, candidates=None):
        scores = self.scores(query, k1, b, candidates)
        return [did for did, _ in heapq.nsmallest(k, scores.items(), key=lambda p: (-p[1], p[0]))]
