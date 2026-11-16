"""Manually authored smoke cases for lesson intent and metadata constraints."""
import argparse

from fastapi.testclient import TestClient

from search.experiment import write_json
from search.service import make_app

# Expected criteria were written independently of the benchmark query templates.
PROBES = [
    ("Can I get a beginner practice activity for Year 5 on fractions that represent the same amount?",
     "lesson-00-", 5, "beginner", "practice"),
    ("An advanced quiz on paths that make a bulb light, for Grade 6",
     "lesson-07-", 6, "advanced", "quiz"),
    ("For Grade 4, a beginner lesson about measuring around the edge of a shape",
     "lesson-03-", 4, "beginner", "lesson"),
    ("Grade 7 project for beginners on how pushes and pulls change movement",
     "lesson-08-", 7, "beginner", "project"),
    ("A beginner practice worksheet for fifth grade about estimating real distances on a map",
     "lesson-20-", 5, "beginner", "practice"),
]


def probe(data_dir, artifacts):
    records = []
    with TestClient(make_app(data_dir, artifacts)) as client:
        for query, topic, grade, level, kind in PROBES:
            row = {"query": query, "expected": {"topic_id_prefix": topic, "grade": grade,
                   "difficulty": level, "content_type": kind}}
            for mode in ("bm25", "rrf", "lambdamart"):
                response = client.post("/search", json={"query": query, "k": 5, "mode": mode})
                response.raise_for_status()
                found = response.json()["results"]
                ranks = [i for i, item in enumerate(found, 1) if item["id"].startswith(topic)
                         and item["grade"] == grade and item["difficulty"] == level
                         and item["content_type"] == kind]
                row[mode] = {"first_exact_rank": ranks[0] if ranks else None,
                             "top_results": [item["id"] for item in found]}
            records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".cache")
    parser.add_argument("--artifacts", default="runs")
    parser.add_argument("--output", default="runs/probes.json")
    args = parser.parse_args()
    records = probe(args.data_dir, args.artifacts)
    write_json(args.output, records)
    for row in records:
        print(row["query"], {mode: row[mode]["first_exact_rank"]
                             for mode in ("bm25", "rrf", "lambdamart")})


if __name__ == "__main__":
    main()
