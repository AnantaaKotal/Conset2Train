"""Policy, RAG, and scaling experiments. All outputs are real measurements."""
import csv
import json
import math
import random
import resource
import time
from copy import deepcopy
from pathlib import Path

from common import (bootstrap, core_metrics, environment, file_hash, mean,
                    percentile, read_json, read_jsonl, task_f1, write_csv,
                    write_json, write_jsonl)
from data import chunks, training_policies
from models import LLM, llm_decision, parse_labels
from policy import FUTURE, artifact, decide, invalidate, oracle, rule, source, transfer
from search import Search


def make_llm(args):
    return LLM(args.provider, args.model, args.model_revision, args.device,
               args.max_tokens, args.max_context, args.seed,
               args.cache if getattr(args,"cache",None) else None)


def save_run(args, metrics, rows, **extra):
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Output already exists: {out}; use a new --out directory")
    out.mkdir(parents=True)
    write_jsonl(out/"raw.jsonl", rows)
    write_json(out/"metrics.json", metrics)
    dataset_path = Path(args.data)/args.dataset/"manifest.json" if hasattr(args,"dataset") else None
    source_hashes = {p.name:file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    write_json(out/"run.json", dict(settings=vars(args), environment=environment(),
               source_sha256=source_hashes,
               data_manifest=read_json(dataset_path) if dataset_path and dataset_path.exists() else None, **extra))
    print("Results:", out, flush=True)


def future_requests(a):
    for kind in sorted(FUTURE[a["type"]]):
        for purpose in ("service", "improvement", "marketing"):
            probe = dict(a, type=kind)
            action = "deploy" if kind == "adapter" else "retrieve" if kind in {"chunk","index"} else "embed"
            yield probe, action, purpose


def annotate(args):
    cases = read_jsonl(args.cases)
    with open(args.labels, newline="") as f:
        labels = {r["case_id"]:r for r in csv.DictReader(f)}
    out = []
    pairs = []
    for c in cases:
        row = labels.get(c["id"])
        if not row or not row["gold"]:
            continue
        for key in ("reviewer1", "reviewer2", "gold"):
            if row[key] not in {"allow", "deny", "conditional"}:
                raise ValueError(f"Missing/invalid {key} for {c['id']}")
        future = json.loads(row["future_gold"])
        expected = list(future_requests(c["artifact"]))
        if len(future) != len(expected) or any(x not in {"allow","deny","conditional"} for x in future):
            raise ValueError(f"future_gold must have {len(expected)} ordered decisions")
        if row["obligation_gold"] not in {"fulfilled","unfulfilled","not_applicable"}:
            raise ValueError("Annotate obligation_gold too")
        out.append(dict(c, gold=row["gold"], future_gold=future,
                        obligation_gold=row["obligation_gold"], review_status="reviewed"))
        pairs.append((row["reviewer1"],row["reviewer2"]))
    if not pairs:
        raise ValueError("No completed review rows")
    agreement = mean([a == b for a,b in pairs])
    chance = sum(sum(a==l for a,b in pairs)*sum(b==l for a,b in pairs)/len(pairs)**2 for l in ("allow","deny","conditional"))
    write_jsonl(args.out, out)
    write_json(Path(args.out).with_suffix(".agreement.json"),dict(n=len(pairs), agreement=agreement,
               kappa=(agreement-chance)/(1-chance) if chance != 1 else None))
    print("Reviewed cases:",len(out), flush=True)


def obligation_state(a, registry, method):
    effective = deepcopy(a)
    if method in {"C", "S", "no_evidence"}:
        effective["evidence"] = []
    relevant = [sid for sid,_ in effective["contributors"] if any(
                r["modality"] == "require" and "removed" in r["guard"]
                for r in registry[sid]["rules"])]
    if not relevant:
        return "not_applicable"
    evidence = {tuple(x) for x in effective["evidence"]}
    fulfilled = all((sid,r["guard"]["removed"]) in evidence for sid in relevant
                    for r in registry[sid]["rules"] if r["modality"] == "require" and "removed" in r["guard"])
    return "fulfilled" if fulfilled else "unfulfilled"


def policies(args):
    cases = read_jsonl(args.cases)
    if not cases:
        raise ValueError("Case file is empty")
    model = make_llm(args) if args.method == "LLM" else None
    if model:
        model.load()
    imported = None
    if args.method == "G":
        if not args.predictions:
            raise ValueError("G requires --predictions from the actual DToU baseline")
        with open(args.predictions,newline="") as f:
            imported = {r["case_id"]:r for r in csv.DictReader(f)}
    rows = []
    for i,c in enumerate(cases):
        a, reg = deepcopy(c["artifact"]), c["registry"]
        if args.missing_lineage:
            rng = random.Random(args.seed+i)
            if any(rng.random() < args.missing_lineage for _ in a["contributors"]):
                a["complete"] = False
        gold = c.get("gold",c["generated_gold"])
        responses, malformed = [], False
        start = time.perf_counter()
        if imported is not None:
            imported_row = imported[c["id"]]
            pred = imported_row["pred"]
            if pred not in {"allow", "deny", "conditional"}:
                raise ValueError("Invalid DToU prediction")
        elif model:
            pred, malformed, response = llm_decision(model,a,reg,c["action"],c["purpose"],c["now"])
            responses.append(response)
        else:
            pred = decide(a,reg,c["action"],c["purpose"],c["now"],args.method)
        ms = 1000*(time.perf_counter()-start)
        probes = []
        for j,(future,action,purpose) in enumerate(future_requests(a)):
            fg = c["future_gold"][j] if "future_gold" in c else oracle(future,reg,action,purpose,c["now"])
            if imported is not None:
                fp = json.loads(imported_row["future_pred"])[j]
            elif model:
                fp, _, response = llm_decision(model,future,reg,action,purpose,c["now"])
                responses.append(response)
            else:
                fp = decide(future,reg,action,purpose,c["now"],args.method)
            probes.append(dict(type=future["type"],action=action,purpose=purpose,gold=fg,pred=fp))
        og = c.get("obligation_gold",obligation_state(c["artifact"],reg,"C2T"))
        op = obligation_state(a,reg,args.method) if args.method not in {"LLM","G"} else None
        rows.append(dict(id=c["id"],group=c["group"],family=c["family"], gold=gold,pred=pred,
                         sa=mean([p["gold"]==p["pred"] for p in probes]),
                         oa=int(op==og) if op is not None else None,probes=probes,ms=ms,
                         malformed=malformed,responses=responses,complete=a["complete"]))
        if (i+1)%25==0:
            print("Policy cases",i+1,"/",len(cases), flush=True)
    metrics = core_metrics(rows)
    intervals = bootstrap(rows,core_metrics,args.seed,args.bootstrap)
    # Bootstrap zero-error FAR alone gives a misleading [0,0]. Add exact bound.
    denied = sum(r["gold"]=="deny" for r in rows)
    metrics["zero_FAR_upper95"] = 1-.05**(1/denied) if denied and metrics["FAR"]==0 else None
    metrics["blocked_incomplete_fraction"] = mean([not r["complete"] for r in rows])
    if imported is not None:
        metrics["p50_ms"] = metrics["p95_ms"] = None  # Import parsing is not baseline latency.
    save_run(args,metrics,rows,ci95=intervals,cases_sha256=file_hash(args.cases),
             reviewed=all(c["review_status"]=="reviewed" for c in cases),
             fixture=read_json(Path(args.data)/args.dataset/"manifest.json")["origin"].get("fixture",False))


def corpus(args):
    base = Path(args.data)/args.dataset
    records, queries = read_jsonl(base/"train.jsonl"), read_jsonl(base/f"{args.query_split}.jsonl")[:args.queries]
    registry = training_policies(records,args.seed)
    docs = chunks(records,registry)
    for d in docs:
        input_artifact = transfer([d["artifact"]],"embedding",registry) if args.retriever!="bm25" else d["artifact"]
        d["entry"] = transfer([input_artifact],"index",registry)
    return docs,queries,registry


def make_search(args,docs):
    return Search(docs,args.retriever,args.embedding_model,args.backend,
                  not args.approximate,args.embedding_device,args.embedding_revision)


def pool(args):
    docs,queries,registry = corpus(args)
    search = make_search(args,docs)
    rows = []
    for q in queries:
        eligible = [i for i,d in enumerate(docs) if oracle(d["entry"],registry,"retrieve",args.purpose)=="allow"]
        ids = set(search.query(q["input"],k=20)) | set(search.query(q["input"],eligible,k=20))
        for i in sorted(ids):
            d = docs[i]
            rows.append(dict(query_id=q["id"],chunk_id=d["id"],grade="",query=q["input"],document=d["text"]))
    write_csv(args.out,rows)
    print("Pool:",args.out,"Annotate grade 0, 1, or 2; union pools before evaluation.", flush=True)


def rag(args):
    docs,queries,registry = corpus(args)
    model = make_llm(args) if args.model else None
    if model:
        model.load()
    with open(args.qrels,newline="") as f:
        judgments = {}
        for r in csv.DictReader(f):
            if not r["grade"]:
                continue
            if r["grade"] not in {"0","1","2"}:
                raise ValueError("Qrel grade must be 0, 1, or 2")
            judgments[(r["query_id"],r["chunk_id"])] = int(r["grade"])
    allowed_gold = {i for i,d in enumerate(docs) if oracle(d["entry"],registry,"retrieve",args.purpose)=="allow"}
    def gate(a,action):
        return oracle(a,registry,action,args.purpose) if args.method=="Oracle" else decide(a,registry,action,args.purpose,method=args.method)
    # Gate embedding and index admission separately; no denied example is embedded.
    admitted = []
    for i,d in enumerate(docs):
        a = d["artifact"]
        if args.retriever != "bm25" and gate(a,"embed") != "allow":
            continue
        if args.retriever != "bm25":
            a = transfer([a],"embedding",registry)
        if gate(a,"index") == "allow":
            admitted.append(i)
    eligible = [i for i in admitted if gate(docs[i]["entry"],"retrieve")=="allow"]
    if not admitted:
        raise ValueError("No authorized index entries; report empty index separately")
    search = make_search(args,[docs[i] for i in admitted])
    labels = sorted({l for r in read_jsonl(Path(args.data)/args.dataset/"train.jsonl") for l in r["labels"]})
    rows = []
    for qi,q in enumerate(queries):
        start = time.perf_counter()
        # Include the simple uncached authorization sweep in request latency.
        eligible = {i for i in admitted if gate(docs[i]["entry"],"retrieve")=="allow"}
        local_eligible = [j for j,i in enumerate(admitted) if i in eligible]
        ranked = [admitted[j] for j in search.query(q["input"],local_eligible,k=10)]
        # Recheck policy before context assembly. U intentionally permits all.
        ranked = [i for i in ranked if gate(docs[i]["entry"],"retrieve")=="allow"]
        relevant = {i:judgments[(q["id"],d["id"])] for i,d in enumerate(docs)
                    if i in allowed_gold and judgments.get((q["id"],d["id"]),0)>0}
        missing = [i for i in ranked if (q["id"],docs[i]["id"]) not in judgments]
        if missing and not args.allow_unjudged:
            raise ValueError(f"{q['id']}: {len(missing)} returned chunks lack qrels; add to the pool and judge them")
        recall = len(set(ranked[:5]) & set(relevant))/len(relevant) if relevant else None
        gain = sum((2**relevant.get(i,0)-1)/math.log2(j+2) for j,i in enumerate(ranked[:10]))
        ideal = sum((2**g-1)/math.log2(j+2) for j,g in enumerate(sorted(relevant.values(),reverse=True)[:10]))
        response, pred = None, []
        if model:
            context = "\n\n".join(docs[i]["text"] for i in ranked[:5])
            response = model("Classify the customer query using the context. Return ONLY a JSON array of labels. Valid labels: "+json.dumps(labels),
                             "Context:\n"+context+"\nQuery:\n"+q["input"])
            pred = parse_labels(response["text"],labels,args.dataset=="multiwoz")
        elapsed = 1000*(time.perf_counter()-start)
        unsafe = sum(i not in allowed_gold for i in ranked[:5])
        rows.append(dict(id=q["id"],group=q["id"],ranked=[docs[i]["id"] for i in ranked],
                  R5=recall,nDCG10=gain/ideal if ideal else None,
                  unsafe_context=int(unsafe>0) if ranked else None,unsafe_chunks=unsafe,
                  context_chunks=min(5,len(ranked)),unjudged=len(missing),returned=len(ranked),
                  answerable=bool(relevant),ms=elapsed,gold=q["labels"],pred=pred,response=response))
        print("RAG queries",qi+1,"/",len(queries), flush=True)
    metrics = dict(R5=mean([r["R5"] for r in rows]),nDCG10=mean([r["nDCG10"] for r in rows]),
                   UCR=mean([r["unsafe_context"] for r in rows]),p95_ms=percentile([r["ms"] for r in rows],.95),
                   answerable=sum(r["answerable"] for r in rows),empty_context=sum(r["context_chunks"]==0 for r in rows),
                   unsafe_chunk_rate=sum(r["unsafe_chunks"] for r in rows)/max(1,sum(r["context_chunks"] for r in rows)),
                   unjudged_rate=sum(r["unjudged"] for r in rows)/max(1,sum(r["returned"] for r in rows)),
                   task_F1=task_f1([r["gold"] for r in rows],[r["pred"] for r in rows],labels,args.dataset=="multiwoz") if model else None,
                   query_count=len(rows))
    def metric(sample):
        return dict(R5=mean([r["R5"] for r in sample]),nDCG10=mean([r["nDCG10"] for r in sample]),
                    UCR=mean([r["unsafe_context"] for r in sample]),
                    task_F1=task_f1([r["gold"] for r in sample],[r["pred"] for r in sample],labels,args.dataset=="multiwoz") if model else None)
    save_run(args,metrics,rows,qrels_sha256=file_hash(args.qrels),search_mode=search.mode,
             ci95=bootstrap(rows,metric,args.seed),
             truncated_documents=search.truncated,admitted=len(admitted),eligible=len(eligible),
             fixture=read_json(Path(args.data)/args.dataset/"manifest.json")["origin"].get("fixture",False))


def scale(args):
    rng = random.Random(args.seed)
    registry = {}
    graph = {}
    timings = []
    start = time.perf_counter()
    # Each node selects parents in the preceding layer, guaranteeing acyclicity.
    layers = [[] for _ in range(args.depth)]
    for i in range(args.artifacts):
        level = min(args.depth-1, i*args.depth//args.artifacts)
        sid = f"s{i}"
        if level == 0:
            rules = [rule(purposes=["service","improvement"])]
            if rng.random() < args.density:
                rules.append(rule("prohibit",actions=["deploy"],purposes=["marketing"]))
                if rng.random() < args.conflict:
                    rules.append(rule("prohibit",actions=["retrieve"],purposes=["service"]))
            sid = f"s{i % args.sources}"
            if sid not in registry:
                registry[sid] = source(sid,rules)
            a = artifact([registry[sid]])
        else:
            parents = rng.sample(layers[level-1],min(args.fan_in,len(layers[level-1])))
            a = transfer([graph[p] for p in parents],"chunk",registry)
        graph[str(i)] = a
        layers[level].append(str(i))
    propagation = time.perf_counter()-start
    requests = rng.sample(list(graph),min(args.requests,len(graph)))
    outcomes = []
    for key in requests:
        start = time.perf_counter()
        outcomes.append(decide(graph[key],registry,"retrieve","service"))
        timings.append(1000*(time.perf_counter()-start))
    # Updates traverse actual descendant contributor metadata.
    withdrawn = next(iter(registry))
    start = time.perf_counter()
    registry[withdrawn]["rules"] = [rule(before=0)]
    registry[withdrawn]["version"] += 1
    affected = invalidate(withdrawn,graph)
    withdrawal_ms = 1000*(time.perf_counter()-start)
    metadata = len(json.dumps(dict(graph=graph,registry=registry),separators=(",",":")).encode())
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    memory_mb = memory/(1024**2 if sys.platform=="darwin" else 1024)
    metrics = dict(artifacts=len(graph),p95_ms=percentile(timings,.95),prop_s=propagation,
                   bytes_artifact=metadata/len(graph),memory_mb=memory_mb,
                   withdrawal_ms=withdrawal_ms,affected=len(affected),source_count=len(registry),
                   allowed_requests=outcomes.count("allow"),denied_requests=outcomes.count("deny"))
    save_run(args,metrics,[dict(ms=x) for x in timings],cache="none",memory_scope="whole fresh process peak RSS")
