"""Normalize three corpora and construct separately reviewable policy cases."""
import csv
import gzip
import json
import random
import re
from pathlib import Path
from urllib.request import urlopen

from common import file_hash, fingerprint, read_jsonl, write_csv, write_json, write_jsonl
from policy import FUTURE, artifact, oracle, rule, source, transfer

DATASETS = ("abcd", "multiwoz", "bitext")
BITEXT = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def download(url, path):
    path = Path(path)
    if not path.exists():
        print("Downloading", url, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def normalize_abcd(row):
    scene = {k.lower(): v for k, v in row["scenario"].items()}
    turns = []
    for t in row["delexed"]:
        t = {k.lower(): v for k, v in t.items()}
        if t["speaker"].lower() in {"customer", "agent"}:
            turns.append(dict(speaker=t["speaker"].lower(), text=t["text"]))
    label = scene.get("subflow")
    if not label:
        raise ValueError("ABCD scenario has no subflow: inspect dataset revision")
    return dict(id="abcd:" + str(row["convo_id"]), turns=turns,
                input="\n".join(t["text"] for t in turns if t["speaker"] == "customer"),
                labels=[label])


def normalize_multiwoz(row):
    turns = [dict(speaker="customer" if t["speaker"] == "USER" else "agent", text=t["utterance"])
             for t in row["turns"]]
    return dict(id="multiwoz:" + row["dialogue_id"], turns=turns,
                input="\n".join(t["text"] for t in turns if t["speaker"] == "customer"),
                labels=sorted(row["services"]))


def normalize_bitext(row, index):
    return dict(id="bitext:" + str(index),
                turns=[dict(speaker="customer", text=row["instruction"]),
                       dict(speaker="agent", text=row["response"])],
                input=row["instruction"], labels=[row["intent"]])


def grouped_split(rows, seed):
    """Keep exact and local near duplicates together without quadratic comparisons.

    Character 4-gram Jaccard >= .85; candidate blocking by first/last token and
    nearby character length. This is an explicitly limited near-duplicate rule.
    """
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    exact, buckets, grams = {}, {}, []
    for i, r in enumerate(rows):
        text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", r["input"].lower())).strip()
        gs = {text[j:j+4] for j in range(max(1, len(text)-3))}
        grams.append(gs)
        if text in exact:
            parent[root(i)] = root(exact[text])
        else:
            tokens = text.split()
            for token in set(tokens[:1] + tokens[-1:]):
                for length in range(len(text)//8 - 1, len(text)//8 + 2):
                    for j in buckets.get((token, length), []):
                        if len(gs & grams[j]) / max(1, len(gs | grams[j])) >= .85:
                            parent[root(i)] = root(j)
                buckets.setdefault((token, len(text)//8), []).append(i)
            exact[text] = i
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(root(i), []).append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    out = {s: [] for s in ("train", "dev", "test")}
    total = 0
    for key in keys:
        split = "train" if total < .8*len(rows) else "dev" if total < .9*len(rows) else "test"
        for r in groups[key]:
            r["duplicate_group"] = str(key)
        out[split].extend(groups[key])
        total += len(groups[key])
    return out


def prepare(args):
    out = Path(args.data) / args.dataset
    raw = Path(args.raw) if args.raw else out / "raw"
    if args.fixture:
        # Never treat these synthetic examples as downloaded dataset measurements.
        splits = {s: [dict(id=f"fixture:{args.dataset}:{s}:{i}",
                          turns=[dict(speaker="customer", text=f"Help with issue {i%3}"),
                                 dict(speaker="agent", text=f"Instructions for issue {i%3}")],
                          input=f"Help with issue {i%3}", labels=[f"intent_{i%3}"])
                      for i in range(n)]
                  for s, n in (("train", args.train), ("dev", args.dev), ("test", args.test))}
        origin = {"fixture": True}
    elif args.dataset == "abcd":
        path = raw / "abcd_v1.1.json.gz"
        if not path.exists() and not (raw / "abcd_v1.1.json").exists():
            download(f"https://raw.githubusercontent.com/asappresearch/abcd/{args.revision}/data/abcd_v1.1.json.gz", path)
        p = raw / "abcd_v1.1.json" if (raw / "abcd_v1.1.json").exists() else path
        content = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
        raw_data = json.loads(content)
        splits = {s: [normalize_abcd(r) for r in raw_data[s]] for s in ("train", "dev", "test")}
        origin = {"source": "asappresearch/abcd", "revision": args.revision, "sha256": file_hash(p)}
    elif args.dataset == "multiwoz":
        splits, hashes = {}, {}
        for s, count in (("train", 17), ("dev", 2), ("test", 2)):
            records = []
            for i in range(1, count+1):
                path = raw / s / f"dialogues_{i:03d}.json"
                url = f"https://raw.githubusercontent.com/budzianowski/multiwoz/{args.revision}/data/MultiWOZ_2.2/{s}/{path.name}"
                download(url, path)
                records.extend(json.loads(path.read_text()))
                hashes[str(path.relative_to(raw))] = file_hash(path)
            splits[s] = [normalize_multiwoz(r) for r in records]
        origin = {"source": "budzianowski/multiwoz", "revision": args.revision, "sha256": hashes}
    else:
        csvs = list(raw.glob("*.csv")) if raw.exists() else []
        if csvs:
            with csvs[0].open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            origin = {"source": str(csvs[0]), "sha256": file_hash(csvs[0])}
        else:
            from datasets import load_dataset
            ds = load_dataset(BITEXT, split="train", revision=args.hf_revision)
            rows = list(ds)
            origin = {"source": BITEXT, "revision": args.hf_revision, "fingerprint": ds._fingerprint}
        splits = grouped_split([normalize_bitext(r, i) for i, r in enumerate(rows)], args.seed)
    for s, limit in (("train", args.train), ("dev", args.dev), ("test", args.test)):
        rows = splits[s]
        random.Random(args.seed).shuffle(rows)
        if len(rows) < limit:
            raise ValueError(f"Requested {limit} {s} records; only {len(rows)} available")
        write_jsonl(out / f"{s}.jsonl", rows[:limit])
    write_json(out / "manifest.json", dict(dataset=args.dataset, origin=origin, split_seed=args.seed,
               counts={s: n for s, n in (("train", args.train), ("dev", args.dev), ("test", args.test))},
               split_hashes={s: file_hash(out/f"{s}.jsonl") for s in splits},
               bitext_dedup="normalized exact + blocked character-4gram Jaccard .85"))
    print("Prepared", out, flush=True)


FAMILIES = ["allow", "train_prohibition", "unfulfilled_removal", "fulfilled_removal",
            "removed_field_prohibition", "adapter_purpose", "excluded_source",
            "mixed_training", "expired_grant", "withdrawal", "second_authority",
            "adapter_merge", "missing_lineage", "unsupported_guard", "index_entry"]


def make_case(records, index, seed):
    rng = random.Random(seed*100000 + index)
    chosen = rng.sample(records, min(2, len(records)))
    ids = [r["id"] for r in chosen]
    if len(ids) == 1:
        ids.append(ids[0] + ":sibling")
    sources = [source(sid) for sid in ids]
    family = index % len(FAMILIES)
    action, purpose, kind, removed, keep = "retrieve", "service", "chunk", [], None
    if family == 1:
        sources[0]["rules"].append(rule("prohibit", actions=["train"]))
        action, kind = "train", "source"
    elif family in (2, 3):
        sources[0]["rules"].append(rule("require", actions=["embed", "train"], removed="identifier"))
        action, kind = "embed", "chunk"
        if family == 3:
            removed = [(ids[0], "identifier")]
    elif family == 4:
        sources[0]["rules"].append(rule("prohibit", field="identifier", actions=["retrieve"]))
        removed = [(ids[0], "identifier")]
    elif family == 5:
        sources[0]["rules"].append(rule("prohibit", types=["adapter"], actions=["deploy"], purposes=["marketing"]))
        action, purpose, kind = "deploy", "marketing", "adapter"
    elif family in (6, 7, 14):
        sources[1]["rules"].append(rule("prohibit", actions=["retrieve", "train", "deploy"]))
        if family == 6:
            keep = [ids[0]]
        if family == 7:
            action = "train"
        if family == 14:
            keep, kind = [ids[0]], "index"
    elif family in (8, 9):
        sources[0]["rules"] = [rule(before=5)]
        action, kind = "embed", "chunk"
        if family == 9:
            sources[0]["version"] = 2
    elif family == 10:
        sources[0]["authorities"].append("organization")
    elif family == 11:
        sources[1]["rules"].append(rule("prohibit", types=["adapter"], purposes=["marketing"], actions=["deploy"]))
        action, purpose, kind = "deploy", "marketing", "adapter"
    elif family == 13:
        sources[0]["rules"].append(rule("require", custom_predicate=True))
    registry = {s["id"]: s for s in sources}
    parents = [artifact(sources)]
    # Intermediate chunk must preserve downstream adapter restrictions.
    if kind == "adapter":
        parents = [transfer(parents, "chunk", registry)]
    a = transfer(parents, kind, registry, keep=keep, remove=removed)
    if family == 12:
        a["complete"] = False
    case = dict(id=f"s{seed}:case:{index}", group=ids[0], family=FAMILIES[family],
                registry=registry, artifact=a, action=action, purpose=purpose, now=10,
                text=chosen[0]["input"], review_status="generated")
    case["generated_gold"] = oracle(a, registry, action, purpose)
    return case


def build_cases(args):
    records = read_jsonl(Path(args.data)/args.dataset/"test.jsonl")
    if not records:
        raise ValueError("No test records")
    # Human candidates and generated coverage cases use disjoint case IDs.
    cases = [make_case(records, i, args.seed) for i in range(args.count)]
    write_jsonl(args.cases, cases)
    requests = []
    for c in cases[:args.review_count]:
        future = [dict(type=kind, action="deploy" if kind=="adapter" else "retrieve" if kind in {"chunk","index"} else "embed", purpose=purpose)
                  for kind in sorted(FUTURE[c["artifact"]["type"]])
                  for purpose in ("service","improvement","marketing")]
        # Do not reveal generated answers in the reviewer packet.
        packet = {k:v for k,v in c.items() if k!="generated_gold"}
        requests.append(dict(packet,future_requests=future))
    write_jsonl(Path(args.cases).with_suffix(".review.jsonl"),requests)
    review = [dict(case_id=c["id"], reviewer1="", reviewer2="", gold="",
                   future_gold="", obligation_gold="", notes="") for c in cases[:args.review_count]]
    write_csv(Path(args.cases).with_suffix(".review.csv"), review)
    write_jsonl(Path(args.cases).with_suffix(".coverage.jsonl"),cases[args.review_count:])
    print("Cases:", args.cases, "Review labels are blank until independently annotated.", flush=True)


def training_policies(records, seed):
    """Random policies within label strata; IDs influence policies, never labels."""
    rng = random.Random(seed)
    registry = {}
    for r in sorted(records, key=lambda x: (x["labels"], x["id"])):
        s = source(r["id"])
        draw = rng.random()
        if draw < .25:
            s["rules"].append(rule("prohibit", actions=["train"]))
        elif draw < .5:
            s["rules"].append(rule("require", actions=["train", "embed"], removed="identifier"))
        elif draw < .75:
            s["rules"].append(rule("prohibit", types=["adapter"], actions=["deploy"], purposes=["marketing"]))
        if rng.random() < .2:
            s["rules"].append(rule("prohibit", actions=["retrieve"], purposes=["service"]))
        registry[s["id"]] = s
    return registry


def chunks(records, registry, max_words=128):
    result = []
    for r in records:
        s = registry[r["id"]]
        a = artifact([s])
        # Insert and then verifiably remove only this fictional structured field.
        a = transfer([a], "chunk", registry, remove=[(s["id"], "identifier")])
        turns = [t["text"] for t in r["turns"]]
        parts, current = [], []
        for turn in turns:
            words = turn.split()
            if len(words) > max_words:
                if current:
                    parts.append("\n".join(current))
                    current = []
                parts.extend(" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words))
                continue
            if current and sum(len(x.split()) for x in current) + len(words) > max_words:
                parts.append("\n".join(current))
                previous = current[-1]
                current = [previous] if len(previous.split())+len(words) <= max_words else []
            current.append(turn)
        if current:
            parts.append("\n".join(current))
        for i, text in enumerate(parts):
            result.append(dict(id=f"{r['id']}:chunk:{i}", record=r["id"], text=text,
                               labels=r["labels"], artifact=a))
    return result
