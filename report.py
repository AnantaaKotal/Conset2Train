"""Collect runs, retain missing results as TBD, and export pasteable LaTeX rows."""
import csv
import math
from pathlib import Path

from common import fingerprint, mean, read_json, read_jsonl, write_csv, write_json

CORPORA = ("abcd","multiwoz","bitext")
CORE = ("U","S","C","G (DToU)","L-Qwen","L-Llama","L-SmolLM","C2T")
FT = ("Qwen3-4B","Llama-3.2-3B","SmolLM2-1.7B")
RAG = ("Strongest baseline","Oracle","C2T canonical","C2T + Llama","C2T + SmolLM",
       "C2T + BM25","C2T + MiniLM","C2T + Chroma","C2T + FAISS")
ABLATIONS = ("C2T","no_type","record_only","no_evidence","index_wide")


def model_label(model):
    if "Qwen3-4B" in model:
        return "Qwen3-4B"
    if "Llama-3.2-3B" in model:
        return "Llama-3.2-3B"
    if "SmolLM2-1.7B" in model:
        return "SmolLM2-1.7B"
    return model


def escape(text):
    replacements={"\\":r"\textbackslash{}","_":r"\_","%":r"\%","&":r"\&","#":r"\#",
                  "$":r"\$","{":r"\{","}":r"\}","~":r"\textasciitilde{}","^":r"\textasciicircum{}"}
    return "".join(replacements.get(c,c) for c in text)


def export(args):
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    runs, flat = [], []
    seen = set()
    for path in sorted(Path(args.results).rglob("metrics.json")):
        run = read_json(path.parent/"run.json")
        settings = run["settings"]
        metrics = read_json(path)
        if run.get("fixture") and not args.include_fixtures:
            continue
        identity = (settings["command"],settings.get("dataset"),settings.get("seed"),
                    settings.get("label"),settings.get("method"),settings.get("model"),
                    settings.get("artifacts"), settings.get("missing_lineage",0))
        # Different output paths must never silently count as independent repeats.
        if identity in seen:
            raise ValueError(f"Duplicate run identity at {path.parent}; select a clean results root")
        seen.add(identity)
        runs.append(dict(run=run,settings=settings,metrics=metrics,path=str(path.parent)))
        flat.append({"path":str(path.parent),**settings,**metrics})
    write_csv(out/"all_runs.csv",flat)
    write_json(out/"runs.json",runs)
    def values(command,dataset,label,metric,method=None,model=None):
        selected = []
        for r in runs:
            s = r["settings"]
            if s["command"] != command or s.get("dataset") != dataset:
                continue
            if s.get("missing_lineage",0):
                continue
            if command in {"policies","ablate"} and not r["run"].get("reviewed",False) and not args.include_generated:
                continue
            if label is not None and s.get("label") != label:
                continue
            if method is not None and s.get("method") != method:
                continue
            if model is not None and model_label(s.get("model","")) != model:
                continue
            if command=="rag" and metric=="p95_ms" and not s.get("model"):
                continue  # Retrieval-only time is not the paper's end-to-end latency.
            selected.append(r)
        seeds = sorted({r["settings"].get("seed") for r in selected})
        if len(selected) != len(seeds):
            raise ValueError("Multiple configurations share a table label; use distinct --label values")
        signature_keys = ("method","provider","model","model_revision","retriever","embedding_model",
                          "embedding_revision","backend","approximate","query_split","purpose","queries",
                          "max_tokens","max_context","batch_size","micro_batch","max_length","epochs",
                          "steps","learning_rate","eval_count","targets")
        signatures = {tuple(r["settings"].get(k) for k in signature_keys) for r in selected}
        if len(signatures)>1:
            raise ValueError("A table row combines different configurations across seeds; separate results roots or labels")
        data_hashes = {fingerprint(r["run"].get("data_manifest",{})) for r in selected}
        if len(data_hashes)>1:
            raise ValueError("A table row combines different prepared datasets across seeds")
        # Require three paper seeds unless explicitly exporting a pilot.
        if not args.pilot and seeds != [11,23,37]:
            return None
        return mean([r["metrics"].get(metric) for r in selected])
    def cell(x,percent=False):
        return "TBD" if x is None else f"{x*100 if percent else x:.3f}"
    def write(name,rows,description):
        note = "% " + description + "\n% Paste only these rows between \\midrule and \\bottomrule.\n"
        (out/name).write_text(note+"\n".join(" & ".join(row)+r" \\" for row in rows)+"\n")
    budgets=[]
    for dataset in CORPORA:
        manifest = Path(args.data)/dataset/"manifest.json"
        meta = read_json(manifest) if manifest.exists() else {}
        counts = meta.get("counts",{}) if args.include_fixtures or not meta.get("origin",{}).get("fixture") else {}
        reviewed = [r for r in runs if r["settings"]["command"]=="policies" and r["settings"].get("dataset")==dataset and r["run"].get("reviewed")]
        n = len(read_jsonl(Path(reviewed[0]["path"])/"raw.jsonl")) if reviewed else None
        budgets.append([escape(dataset)]+[str(counts.get(k,"TBD")) for k in ("train","dev","test")]+[str(n) if n is not None else "TBD"])
    write("table_II_datasets.tex",budgets,"Actual prepared counts and reviewed-case counts; targets must be replaced if they differ.")
    core_labels = list(CORE)+sorted({s["settings"].get("label","") for s in runs if s["settings"]["command"]=="policies"}-set(CORE)-{None,""})
    rows = [[escape(label)]+[cell(values("policies",d,label,m),m in {"FAR","FDR"}) for d in CORPORA for m in ("FAR","FDR","F1","SA")] for label in core_labels]
    write("table_III_core.tex",rows,"Core results; reviewed cases by default. Three seeds unless --pilot.")
    rows=[]
    for model in list(FT)+sorted({model_label(r["settings"].get("model","")) for r in runs if r["settings"]["command"]=="train"}-set(FT)):
        for method in ("U","C","C2T"):
            rows.append([escape(model),method]+[cell(values("train",d,None,m,method,model),m in {"UA","AC"}) for d in CORPORA for m in ("F1","UA","AC")])
    write("table_IV_finetuning.tex",rows,"Task F1 and admission metrics; UA/AC in percent.")
    rows=[]
    for label in list(RAG)+sorted({r["settings"].get("label","") for r in runs if r["settings"]["command"]=="rag"}-set(RAG)-{None,""}):
        vals = [cell(values("rag",d,label,m),m=="UCR") for d in CORPORA for m in ("R5","nDCG10","UCR")]
        latency = [values("rag",d,label,"p95_ms") for d in CORPORA]
        rows.append([escape(label)]+vals+[cell(mean(latency)) if all(x is not None for x in latency) else "TBD"])
    write("table_V_rag.tex",rows,"Qrel-based RAG; p95 is mean of corpus-specific end-to-end p95, never cached calls.")
    rows=[]
    for method in ABLATIONS:
        vals=[]
        for metric in ("FAR","FDR","SA","OA"):
            v = [values("ablate",d,None,metric,method) for d in CORPORA]
            vals.append(cell(mean(v),metric in {"FAR","FDR"}) if all(x is not None for x in v) else "TBD")
        rows.append([escape(method)]+vals)
    write("table_VI_ablations.tex",rows,"Macro-average of three corpora; require all three.")
    rows=[]
    for n in (1000,10000,100000):
        relevant = [r for r in runs if r["settings"]["command"]=="scale" and r["settings"]["artifacts"]==n and r["settings"]["depth"]==6 and r["settings"]["fan_in"]==8]
        seeds = sorted({r["settings"]["seed"] for r in relevant})
        configurations = {tuple(r["settings"][k] for k in ("sources","density","conflict","requests")) for r in relevant}
        if len(configurations)>1 or len(relevant)!=len(seeds):
            raise ValueError("Scaling row mixes configurations or repeats; use a clean results root")
        ok = args.pilot or seeds==[11,23,37]
        rows.append([str(n)]+[cell(mean([r["metrics"][m] for r in relevant])) if ok else "TBD" for m in ("p95_ms","prop_s","bytes_artifact","memory_mb")])
    write("table_VII_scaling.tex",rows,"Depth 6/fan-in 8. Inspect root count and density in run metadata.")
    warnings = []
    if args.include_generated: warnings.append("Includes unreviewed generated cases; do not describe Table III as curated human-reviewed results.")
    if args.include_fixtures: warnings.append("Contains synthetic fixture data; not dataset experiments.")
    if args.pilot: warnings.append("Pilot export can contain fewer than three seeds.")
    (out/"EXPORT_NOTES.txt").write_text("\n".join(warnings or ["Review settings, CI files, and run counts before submission."])+"\n")
    print("Exported",out,flush=True)


def compare(args):
    left = {r["id"]:r for r in read_jsonl(Path(args.left)/"raw.jsonl")}
    right = {r["id"]:r for r in read_jsonl(Path(args.right)/"raw.jsonl")}
    if set(left)!=set(right):
        raise ValueError("Paired runs require identical case IDs")
    b=c=0
    for key in left:
        a,z = left[key],right[key]
        if a["gold"]!=z["gold"]:
            raise ValueError("Gold labels differ")
        ag,zg = a["gold"]==a["pred"],z["gold"]==z["pred"]
        b += ag and not zg
        c += zg and not ag
    n=b+c
    p=min(1,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1
    write_json(args.out,dict(left_only_correct=b,right_only_correct=c,mcnemar_exact_p=p,
               holm_note="Apply Holm correction over your predeclared comparison family, not each run separately."))
    print("Exact paired test:",args.out,flush=True)


def holm(args):
    rows = [dict(file=path,**read_json(path)) for path in args.inputs]
    rows.sort(key=lambda r:r["mcnemar_exact_p"])
    previous=0
    for i,r in enumerate(rows):
        previous=max(previous,min(1,(len(rows)-i)*r["mcnemar_exact_p"]))
        r["holm_p"]=previous
    write_json(args.out,rows)
    print("Holm family:",args.out,flush=True)


def qrels(args):
    rows={}
    for path in args.inputs:
        with open(path,newline="") as f:
            for r in csv.DictReader(f):
                key=(r["query_id"],r["chunk_id"])
                if key in rows and rows[key]["grade"] and r["grade"] and rows[key]["grade"]!=r["grade"]:
                    raise ValueError("Conflicting grades for "+str(key))
                if key not in rows or r["grade"]:
                    rows[key]=r
    write_csv(args.out,[rows[key] for key in sorted(rows)])
    print("Union qrels:",args.out,flush=True)
