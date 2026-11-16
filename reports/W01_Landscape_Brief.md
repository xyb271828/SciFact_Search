# Week 1 landscape brief

## Where retrieval and ranking change the experience

| Surface | Retrieval and ranking | Offline measurement | Online measurement to design |
|---|---|---|---|
| Lesson search | Retrieve with BM25 and embeddings, fuse candidates, then rank by text relevance and course attributes. Apply strict grade or content type filters only when the query or UI makes them mandatory. | Recall@100 for candidate coverage, NDCG@10 for graded relevance, and a constraint violation rate for required filters. Report the latency of each stage. | Successful lesson opens or saves, quick reformulations, time to a useful lesson, and p99 latency. |
| Activity recommendation | Retrieve eligible activities, rank for predicted usefulness, then diversify the list and apply eligibility rules. Repeat an activity only when the product permits it. | Recall@K and NDCG@K on later sessions, catalog coverage and diversity, and eligibility violations. | Activity starts and completions, dismissals, exposure diversity, and p99 latency. |
| Skills catalog ads | Retrieve eligible listings, compute relevance and click probability, then apply auction, budget, relevance, and placement rules. | AUC for discrimination, log loss and ECE for probability quality, and pacing/relevance constraints on replay. | Qualified clicks, spend pacing and ad load, and satisfaction with search. |

These are design proposals. The measured work in this repository is the offline search evaluation and local search serving. Each stage requires its own latency measurement. Stage percentiles identify hotspots. Measure end-to-end p99 separately because percentiles of individual stages cannot be added to recover it.

## Logging contract proposed for a later integration

| Event | Fields | Purpose |
|---|---|---|
| Impression | request ID, timestamp, surface, item ID, rank, retrieval/ranker version, experiment assignment if applicable | Reconstruct which items a query actually exposed and in what order. |
| Interaction | request ID, item ID, event timestamp, event type (open, save, dismiss, completion) | Attribute an outcome to an impression and distinguish superficially similar outcome types. |
| Serving telemetry | request ID, timestamp, mode, stage durations, status, fallback indicator | Explain slow requests and distinguish fallback traffic from model traffic. |

Join the impression and interaction on request and item IDs. Version the evaluator and model. A later system with timestamped behavior should compute features as of the request time and split by time. This project's SciFact and synthetic experiments use query splits and static item content. The lesson catalog and query benchmark are fictional and use no learner profiles.
