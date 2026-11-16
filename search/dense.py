"""Pinned MiniLM ONNX encoder and FAISS cosine retrieval."""
import hashlib
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from search.data import download

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
BASE_URL = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"


class Encoder:
    def __init__(self, cache, threads=2):
        cache = Path(cache)
        self.tokenizer = Tokenizer.from_file(str(download(f"{BASE_URL}/tokenizer.json", cache / "tokenizer.json")))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        model = download(f"{BASE_URL}/onnx/model.onnx", cache / "model.onnx")
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def encode(self, texts, batch_size=16):
        batches = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer.encode_batch(texts[start:start + batch_size])
            masks = np.array([token.attention_mask for token in tokens], dtype=np.int64)
            inputs = {
                "input_ids": np.array([token.ids for token in tokens], dtype=np.int64),
                "attention_mask": masks,
                "token_type_ids": np.array([token.type_ids for token in tokens], dtype=np.int64),
            }
            hidden = self.session.run(None, {key: inputs[key] for key in self.input_names})[0]
            mask = masks[:, :, None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
            batches.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12))
        return np.ascontiguousarray(np.vstack(batches), dtype=np.float32)


class DenseIndex:
    def __init__(self, docs, encoder, cache, build_hnsw=True):
        # A cache is valid only for the same encoder and ordered document content.
        self.ids = sorted(docs)
        self.positions = {did: i for i, did in enumerate(self.ids)}
        texts = [docs[did].full_text for did in self.ids]
        key = hashlib.sha256(f"{REVISION}:mean:256\n".encode())
        for did, body in zip(self.ids, texts):
            key.update(f"{did}\0{body}\0".encode())
        vector_file = Path(cache) / f"embeddings-{key.hexdigest()[:24]}.npy"
        vector_file.parent.mkdir(parents=True, exist_ok=True)
        if vector_file.exists():
            self.vectors = np.load(vector_file, allow_pickle=False)
        else:
            self.vectors = encoder.encode(texts)
            temporary = vector_file.with_suffix(".part")
            with temporary.open("wb") as output:
                np.save(output, self.vectors, allow_pickle=False)
            temporary.replace(vector_file)

        dims = self.vectors.shape[1]
        self.flat = faiss.IndexFlatIP(dims)
        self.flat.add(self.vectors)
        self.hnsw = None
        if build_hnsw:
            self.hnsw = faiss.IndexHNSWFlat(dims, 32, faiss.METRIC_INNER_PRODUCT)
            self.hnsw.hnsw.efConstruction = 100
            self.hnsw.hnsw.efSearch = 128
            self.hnsw.add(self.vectors)

    def search(self, vector, k=100, index="flat"):
        search_index = self.hnsw if index == "hnsw" else self.flat
        if search_index is None:
            raise ValueError("HNSW was not built")
        _, indices = search_index.search(vector, min(k, len(self.ids)))
        return [self.ids[i] for i in indices[0] if i >= 0]

    def scores(self, vector, ids):
        return self.vectors[[self.positions[did] for did in ids]] @ vector[0]
