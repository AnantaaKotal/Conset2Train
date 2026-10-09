"""JSON/CSV files and elementary metrics; standard library only."""
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment():
    packages = {}
    for p in ("numpy", "torch", "transformers", "peft", "datasets", "sentence-transformers",
              "qdrant-client", "chromadb", "faiss-cpu"):
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), platform=platform.platform(), packages=packages)


def ratio(n, d):
    return n / d if d else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def macro_f1(gold, pred, labels):
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return mean(scores)


def task_f1(gold, pred, labels, multilabel=False):
    if not multilabel:
        return macro_f1([x[0] for x in gold], [x[0] if x else "__invalid__" for x in pred], labels)
    scores = []
    for label in labels:
        tp = sum(label in g and label in p for g, p in zip(gold, pred))
        fp = sum(label not in g and label in p for g, p in zip(gold, pred))
        fn = sum(label in g and label not in p for g, p in zip(gold, pred))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return mean(scores)


def core_metrics(rows):
    gold, pred = [r["gold"] for r in rows], [r["pred"] for r in rows]
    return dict(FAR=ratio(sum(g == "deny" and p == "allow" for g, p in zip(gold, pred)), gold.count("deny")),
                FDR=ratio(sum(g == "allow" and p != "allow" for g, p in zip(gold, pred)), gold.count("allow")),
                F1=macro_f1(gold, pred, ["allow", "deny", "conditional"]),
                SA=mean([r["sa"] for r in rows]),
                OA=mean([r.get("oa") for r in rows]),
                prerequisite_violation=ratio(sum(g == "conditional" and p == "allow" for g, p in zip(gold, pred)), gold.count("conditional")),
                malformed=mean([int(r.get("malformed", False)) for r in rows]),
                p50_ms=percentile([r["ms"] for r in rows], .5),
                p95_ms=percentile([r["ms"] for r in rows], .95))


def bootstrap(rows, metric, seed=11, repetitions=1000):
    """Resample groups, preserving sibling scenarios and their future probes."""
    rng = random.Random(seed)
    grouped = {}
    for r in rows:
        grouped.setdefault(r.get("group",r.get("id")),[]).append(r)
    groups = list(grouped.values())
    samples = [metric([r for _ in groups for r in rng.choice(groups)]) for _ in range(repetitions)]
    keys = samples[0] if samples else {}
    return {k: [percentile([s[k] for s in samples if s[k] is not None], .025),
                percentile([s[k] for s in samples if s[k] is not None], .975)] for k in keys}
