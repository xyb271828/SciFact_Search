# Week 1 engineering log

The search prototype now has independent lexical, dense, and ranking stages. I kept the original BM25 notebook as a runnable historical benchmark, extracted the reusable search and evaluation code, and added tests against hand-calculated scores and an independent NDCG implementation. Both the synthetic catalog and the search configuration can be reproduced from the documented commands.

On the 300 official SciFact queries, development tuning raised the lexical baseline to 0.6829 NDCG@10. Combining its ranked list with MiniLM/FAISS results using reciprocal rank fusion raised NDCG@10 to 0.7225 and Recall@100 to 0.9550. I reviewed queries where the hybrid model missed articles found by BM25 to understand how fusion can favor broad matches over precise evidence.

The synthetic lesson experiment produced a different result. Fusion recovered a slightly greater share of judged items but lowered top-ten relevance. A LambdaMART model restored the usefulness of the top results by learning from document and query features. A simple grade filter with BM25 nearly matched it on NDCG, so I kept that baseline in the report. This gives a more informative comparison than using the fusion baseline alone.

The local HTTP service returned search results and per-stage timings. At 25 offered QPS the full reranker recorded a 26.65 ms p99 with zero errors over 300 requests. I measured a shallower rerank budget for comparison. An independently authored query exposed a failure to interpret ordinal grade phrases. Next, I would address that parsing gap and evaluate the ranker on the official MSLR fold to measure transfer beyond the synthetic catalog.

**Resume bullet** — Built a BM25 and MiniLM/FAISS hybrid retriever, raising NDCG@10 from 0.683 to 0.723 and Recall@100 from 92.5% to 95.5% across 300 held-out SciFact queries.
