"""Local JSON search service using the frozen lesson catalog and trained model."""
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import faiss
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from search.bm25 import BM25
from search.dense import REVISION, DenseIndex, Encoder
from search.education import synthetic_lessons
from search.pipeline import FEATURES, Pipeline


def make_app(data_dir, artifact_dir, threads=2):
    data_dir, artifact_dir = Path(data_dir), Path(artifact_dir)

    @asynccontextmanager
    async def lifespan(app):
        manifest = json.loads((artifact_dir / "lesson-manifest.json").read_text())
        model_path = artifact_dir / "lesson-ranker.txt"
        if (manifest["embedding_revision"] != REVISION or tuple(manifest["features"]) != FEATURES
                or hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest["model_sha256"]):
            raise RuntimeError("Model and search manifest do not match")
        ds = synthetic_lessons(manifest["synthetic_seed"])
        config = manifest["bm25_rrf"]
        encoder = Encoder(data_dir / "model", threads)
        flat = DenseIndex(ds.docs, encoder, data_dir, build_hnsw=False)
        # Reuse the encoder loaded by this process for every request.
        app.state.pipeline = Pipeline(
            ds.docs, BM25({did: d.full_text for did, d in ds.docs.items()}, config["analyzer"]),
            encoder, flat, config,
        )
        app.state.ranker = lgb.Booster(model_file=str(model_path))
        yield

    app = FastAPI(title="Lesson search prototype", lifespan=lifespan)

    class Request(BaseModel):
        query: str = Field(min_length=1, max_length=512)
        k: int = Field(default=10, ge=1, le=100)
        mode: str = "lambdamart"
        rerank_depth: int = Field(default=100, ge=1, le=100)

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.post("/search")
    def search(request: Request):
        if request.mode not in {"bm25", "rrf", "lambdamart"}:
            raise HTTPException(status_code=422, detail="mode must be bm25, rrf, or lambdamart")
        if not request.query.strip() or (request.mode == "lambdamart" and request.k > request.rerank_depth):
            raise HTTPException(status_code=422, detail="Provide a query and k no greater than rerank_depth")
        pipeline = app.state.pipeline
        faiss.omp_set_num_threads(1)
        start = perf_counter()
        times = {}

        def mark(name, previous):
            now = perf_counter()
            times[name] = (now - previous) * 1000
            return now

        if request.mode == "bm25":
            ids = pipeline.lexical(request.query, request.k)
            mark("retrieval", start)
        else:
            vector = pipeline.encoder.encode([request.query])
            tick = mark("query_encoding", start)
            depth = request.rerank_depth if request.mode == "lambdamart" else request.k
            selected = pipeline.hybrid(request.query, vector, k=depth)
            tick = mark("retrieval", tick)
            ids = selected[0]
            if request.mode == "lambdamart":
                features = pipeline.feature_matrix(request.query, vector, selected)
                tick = mark("features", tick)
                scores = app.state.ranker.predict(features, num_threads=1)
                ids = [did for did, _ in sorted(zip(ids, scores), key=lambda pair: (-pair[1], pair[0]))]
                mark("reranking", tick)
        return {
            "mode": request.mode, "stage_ms": times,
            "results": [
                {"id": did, "title": pipeline.docs[did].title, "grade": pipeline.docs[did].grade,
                 "difficulty": pipeline.docs[did].level, "content_type": pipeline.docs[did].kind}
                for did in ids[:request.k]
            ],
        }

    return app


def create_app():
    return make_app(os.getenv("SEARCH_DATA_DIR", ".cache"), os.getenv("SEARCH_ARTIFACT_DIR", "runs"))
