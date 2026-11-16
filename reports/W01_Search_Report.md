# Week 1 search report

## Experiment design

SciFact used 5,183 documents and the 300 official test queries. A seeded subset of 200 official training queries served as development data. The lesson catalog contained 5,000 simulated resources and 300 queries. The 180 training, 60 development, and 60 test lesson queries use disjoint topic groups. Each corpus remains searchable in full. Public SciFact label judgments measure relevance. The lesson labels grade generated topic, year, difficulty, and resource-type matching. Queries without an explicit difficulty can match either difficulty level.

The BM25 grid covers two analyzers, three k1 choices, and two b choices. The RRF constant also comes from a development-only grid. Both lexical and dense retrieval each contribute up to 100 candidates. Fusion deduplicates the union and keeps its top 100. The lesson LambdaMART experiment holds those 100 candidates fixed, fits on grouped training queries, and uses only the development groups for early stopping. The metadata ablation retrains on the same groups with the final three features removed. Paired percentile bootstrap intervals use 2,000 query-level resamples and seed 42. Lesson intervals treat queries as resampling units and are descriptive because queries within a topic share templates and judgments.

## SciFact

| Run | NDCG@10 | Recall@100 | MRR@10 | p99 in process (ms) |
|---|---:|---:|---:|---:|
| Original BM25 | 0.6622 | 0.8826 | 0.6302 | 9.27 |
| Dev-tuned BM25 | 0.6829 | 0.9253 | 0.6478 | 4.18 |
| rank_bm25 reference | 0.6549 | 0.8764 | 0.6212 | 37.81 |
| MiniLM + Flat | 0.6451 | 0.9250 | 0.6047 | 9.94 |
| BM25 + MiniLM + RRF (Flat) | 0.7225 | 0.9550 | 0.6843 | 15.46 |
| BM25 + MiniLM + RRF (HNSW) | 0.7223 | 0.9550 | 0.6840 | 12.38 |

RRF versus tuned BM25 gives ΔNDCG@10 **+0.0397** (95% CI **[+0.0157, +0.0624]**). Tuned versus original BM25 gives **+0.0206** (95% CI **[+0.0020, +0.0409]**). These runs answer different questions. Development tuning changes the base lexical retriever. Adding MiniLM and RRF to that tuned retriever measures the additional benefit of hybrid retrieval.

The original and reference BM25 share lexical tokenization and numeric parameters but use different IDF formulas. Their observed NDCG difference was **+0.0074** (95% CI **[-0.0016, +0.0166]**). The unit tests separately check handwritten scoring against a numeric fixture.

### Sampled ranking changes

These are the five largest increases and five largest decreases in NDCG@10. They describe test failures after the configuration was selected. They are examples rather than a representative sample.

| Direction | Query IDs and ΔNDCG@10 | Observed pattern |
|---|---|---|
| Increased | 1279 (+1.000), 514 (+0.667), 1278 (+0.631), 72 (+0.569), 628 (+0.569) | Fusion surfaced judged items using related wording and abbreviations. Queries 1279 and 1278 share a judged document and should not be interpreted as wholly independent semantic cases. |
| Decreased | 785 (−1.000), 48 (−0.667), 1363 (−0.613), 1024 (−0.569), 644 (−0.569) | Fusion sometimes pushed down the precise evidence article in favor of broader topical matches. Query 785 lost its judged hit from the top ten. |

### ANN sweep

| Vector index | Overlap@100 vs Flat | Relevance Recall@100 | Vector lookup p99 (ms) |
|---|---:|---:|---:|
| Flat | 1.0000 | 0.9250 | 0.198 |
| HNSW efSearch=32 | 0.8441 | 0.9267 | 0.174 |
| HNSW efSearch=64 | 0.9434 | 0.9267 | 0.155 |
| HNSW efSearch=128 | 0.9855 | 0.9250 | 0.228 |
| HNSW efSearch=256 | 0.9975 | 0.9250 | 0.270 |

Overlap@100 compares approximate neighbors with exact vector neighbors. Relevance recall compares either search output with human judgments. Lookup timing is measured on prepared query vectors. It excludes tokenization and encoding. The two index types showed different timings by efSearch, with small absolute differences at this corpus size.

## Synthetic lesson search

| Run | NDCG@10 | Recall@100 | MRR@10 | p99 in process (ms) |
|---|---:|---:|---:|---:|
| Original BM25 | 0.6017 | 0.4954 | 0.8593 | 5.37 |
| Dev-tuned BM25 | 0.6054 | 0.4942 | 0.8593 | 5.49 |
| MiniLM + Flat | 0.2910 | 0.4907 | 0.8708 | 6.43 |
| BM25 + MiniLM + RRF (Flat) | 0.4988 | 0.5004 | 0.9056 | 13.84 |
| BM25 + MiniLM + RRF (HNSW) | 0.4988 | 0.5004 | 0.9056 | 11.02 |
| Grade filter + BM25 | 0.7121 | 0.2644 | 1.0000 | 2.71 |
| RRF + LambdaMART (100) | 0.7154 | 0.5004 | 1.0000 | 16.35 |
| RRF + LambdaMART (30) | 0.7016 | 0.5004 | 1.0000 | 18.54 |
| RRF + LambdaMART (without metadata constraints) | 0.6634 | 0.5004 | 0.9806 | 13.36 |

LambdaMART versus RRF gives ΔNDCG@10 **+0.2166** (95% CI **[+0.1799, +0.2594]**). Against tuned BM25 the observed difference is **+0.1100** (95% CI **[+0.0563, +0.1712]**). Against the stronger grade-filtered BM25 baseline it is only **+0.0033** (95% CI **[-0.0124, +0.0173]**). The stricter baseline trades away candidate recall because the relevance rubric gives some credit to related resources in neighboring grades.

Fusion increased candidate recall slightly while reducing top-ten quality. Ranked by LambdaMART, the same set reached NDCG 0.7154. Removing grade, difficulty and type features lowered NDCG by **+0.0520** (95% CI **[+0.0227, +0.0859]**). The ablation measures the value of these features under the synthetic relevance rubric. It cannot estimate the lift on real learner or teacher traffic.

Five separately authored prompts exercised the local API. The ranker placed a result meeting all written constraints in the top five on four of five prompts. The missed case used “fifth grade”, which the current parser does not recognize as Grade 5. The cases are qualitative smoke checks, and no model selection used them. See [probe outputs](results/probes.json).

## Local HTTP serving

One Uvicorn process on AMD EPYC 9V74 80-Core Processor with ONNX CPU inference, two model threads, and one FAISS thread per request. The load client ran on the same host. Each configuration received 300 requests sampled cyclically from the 60 test queries at an offered rate of 25 QPS after 10 warm-up requests. Latency measures HTTP request dispatch through parsed response. Initial scheduling delay is tracked separately. Cached document embeddings and a preloaded model represent steady-state serving. No request-result cache was used.

| Mode | Requests | Achieved success QPS | Errors | p50 HTTP (ms) | p95 HTTP (ms) | p99 HTTP (ms) |
|---|---:|---:|---:|---:|---:|---:|
| bm25_100 | 300 | 25.07 | 0 | 6.07 | 7.62 | 9.53 |
| rrf_100 | 300 | 25.06 | 0 | 10.81 | 18.29 | 25.21 |
| lambdamart_30 | 300 | 25.06 | 0 | 10.55 | 15.07 | 24.75 |
| lambdamart_100 | 300 | 25.05 | 0 | 11.37 | 16.88 | 26.65 |

Reranking 30 candidates instead of 100 reduced measured HTTP p99 from **26.65 to 24.75 ms** and reduced NDCG@10 from **0.7154 to 0.7016**. A single run with 300 requests per setup provides a local performance observation. A repeat under sustained traffic and larger catalogs is needed before inferring capacity or stable tail differences. Docker packaging and CI configuration are present. This run used a local Python process.

## MSLR-WEB10K Fold 1

This independent learning-to-rank experiment uses the official query splits and all 136 supplied features. The training, validation, and test files contain 6,000, 2,000, and 2,000 disjoint query groups, respectively. LambdaMART was fitted on grouped training examples with validation-only early stopping. Both methods score the same test candidates. The title BM25 baseline sorts candidates by the dataset's precomputed feature 108.

| Ranker | Test NDCG@10 |
|---|---:|
| Title BM25 (feature 108) | 0.2750 |
| LightGBM LambdaMART | 0.4680 |

The paired difference was **+0.1930** (95% CI **[+0.1839, +0.2029]**). This compares a learned multi-feature scorer with a single title-only lexical feature on fixed candidates. It measures ranking quality on MSLR and provides no estimate of candidate retrieval gains or the effect of any one feature.

The model recorded **1.602 ms p99** for CPU scoring of one query's supplied candidates with one inference thread. This excludes retrieval, the offline generation of features, and HTTP overhead. See the [benchmark output](results/mslr.json).

## Scope

SciFact gives a public retrieval result on a small scientific corpus. The lesson workload is synthetic, and product-specific ranking results depend on its label rules and candidate coverage. MSLR evaluates an independent fixed-candidate ranking task with its own relevance labels. No real learner records, online engagement measurements, or deployed traffic were used.
