"""Dataset adapters for public benchmark and generated education examples."""
import csv
import json
import random
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

SCIFACT = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


@dataclass(frozen=True)
class Document:
    title: str
    text: str
    grade: int = 0
    level: str = ""
    kind: str = ""

    @property
    def full_text(self):
        return f"{self.title} {self.text}"


@dataclass(frozen=True)
class Dataset:
    docs: dict
    queries: dict
    qrels: dict
    splits: dict


def download(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    return path


def scifact(path):
    with ZipFile(download(SCIFACT, path)) as archive:
        def records(name):
            with archive.open(f"scifact/{name}.jsonl") as stream:
                return {row["_id"]: row for row in map(json.loads, stream)}

        docs = {
            key: Document(row["title"], row["text"])
            for key, row in records("corpus").items()
        }
        queries = {key: row["text"] for key, row in records("queries").items()}
        qrels, splits = {}, {}
        for split in ("train", "test"):
            splits[split] = set()
            with archive.open(f"scifact/qrels/{split}.tsv") as stream:
                rows = csv.reader(TextIOWrapper(stream, encoding="utf-8"), delimiter="\t")
                next(rows)
                for qid, did, grade in rows:
                    qrels.setdefault(qid, {})[did] = int(grade)
                    splits[split].add(qid)
        if splits["train"] & splits["test"]:
            raise ValueError("SciFact train and test queries overlap")
        dev = sorted(splits["train"])
        random.Random(42).shuffle(dev)
        splits["dev"] = set(dev[:200])
        splits["train"] -= splits["dev"]
    return Dataset(docs, queries, qrels, {key: sorted(ids) for key, ids in splits.items()})
