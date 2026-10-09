#!/usr/bin/env python3
"""Single command line entry point. Run python run.py --help."""
import argparse
from pathlib import Path

from data import DATASETS


def main():
    p = argparse.ArgumentParser(description="Consent2Train experiments and IEEE table rows")
    sub = p.add_subparsers(dest="command",required=True)
    def dataset(parser):
        parser.add_argument("--dataset",choices=DATASETS,required=True)
        parser.add_argument("--data",default="data")
        parser.add_argument("--seed",type=int,default=11)
    def llm(parser,required=False):
        parser.add_argument("--provider",choices=["hf","openrouter"],default="hf")
        parser.add_argument("--model",required=required)
        parser.add_argument("--model-revision",default="main")
        parser.add_argument("--device",default="auto")
        parser.add_argument("--max-tokens",type=int,default=128)
        parser.add_argument("--max-context",type=int,default=8192)
    def retrieval(parser):
        parser.add_argument("--retriever",choices=["bm25","dense"],default="bm25")
        parser.add_argument("--embedding-model",default="BAAI/bge-small-en-v1.5")
        parser.add_argument("--embedding-revision",default="main")
        parser.add_argument("--embedding-device",default="cpu")
        parser.add_argument("--backend",choices=["numpy","qdrant","chroma","faiss"],default="numpy")
        parser.add_argument("--approximate",action="store_true")
        parser.add_argument("--queries",type=int,default=150)
        parser.add_argument("--query-split",choices=["dev","test"],default="test")
        parser.add_argument("--purpose",choices=["service","improvement","marketing"],default="service")
    a=sub.add_parser("prepare",help="Download/normalize a corpus")
    dataset(a)
    a.add_argument("--raw",help="Local source folder instead of download")
    a.add_argument("--revision",default="master",help="Git commit or branch for ABCD/MultiWOZ")
    a.add_argument("--hf-revision",default="main",help="Bitext dataset revision")
    a.add_argument("--train",type=int,default=2000)
    a.add_argument("--dev",type=int,default=500)
    a.add_argument("--test",type=int,default=500)
    a.add_argument("--fixture",action="store_true",help="Synthetic smoke test; never paper results")
    a=sub.add_parser("cases",help="Generate cases and blank independent-review CSV")
    dataset(a)
    a.add_argument("--count",type=int,default=600,help="100 review candidates + 500 coverage cases")
    a.add_argument("--review-count",type=int,default=100)
    a.add_argument("--cases",required=True)
    a=sub.add_parser("annotate",help="Import adjudicated review CSV into cases")
    a.add_argument("--cases",required=True)
    a.add_argument("--labels",required=True)
    a.add_argument("--out",required=True)
    for command in ("policies","ablate"):
        a=sub.add_parser(command,help="Evaluate policy decisions or semantic ablations")
        dataset(a); llm(a)
        a.add_argument("--method",choices=["U","S","C","C2T","G","LLM"] if command=="policies" else ["C2T","no_type","record_only","no_evidence","index_wide"],default="C2T")
        a.add_argument("--cases",required=True)
        a.add_argument("--label",help="Table row label; custom LLMs get their own row")
        a.add_argument("--predictions",help="Actual DToU predictions CSV for method G")
        a.add_argument("--cache",help="Optional LLM response cache; not for timing experiments")
        a.add_argument("--bootstrap",type=int,default=1000)
        a.add_argument("--missing-lineage",type=float,default=0)
        a.add_argument("--out",required=True)
    a=sub.add_parser("pool",help="Produce blank relevance judgments; run for each retriever")
    dataset(a); retrieval(a)
    a.add_argument("--out",required=True)
    a=sub.add_parser("rag",help="Evaluate retrieval/context and optional LLM generation")
    dataset(a); retrieval(a); llm(a)
    a.add_argument("--method",choices=["U","S","C","C2T","Oracle"],default="C2T")
    a.add_argument("--qrels",required=True)
    a.add_argument("--allow-unjudged",action="store_true",help="Pilot only: treat unjudged as nonrelevant and report rate")
    a.add_argument("--label",required=True,help="e.g. 'C2T canonical' or 'C2T + Chroma'")
    a.add_argument("--out",required=True)
    a=sub.add_parser("train",help="Local causal-LM LoRA adaptation and held-out evaluation")
    dataset(a); llm(a,required=True)
    a.add_argument("--method",choices=["U","C","C2T"],default="C2T")
    a.add_argument("--epochs",type=float,default=2)
    a.add_argument("--steps",type=int,help="Override fixed optimizer-step budget for smoke tests")
    a.add_argument("--batch-size",type=int,default=16)
    a.add_argument("--micro-batch",type=int,default=1)
    a.add_argument("--learning-rate",type=float,default=2e-4)
    a.add_argument("--max-length",type=int,default=512)
    a.add_argument("--eval-count",type=int,default=500)
    a.add_argument("--targets",default="q_proj,v_proj",help="Comma-separated LoRA module names for your model")
    a.add_argument("--out",required=True)
    a=sub.add_parser("merge",help="Equal-weight same-base LoRA merge with union lineage")
    a.add_argument("--left",required=True)
    a.add_argument("--right",required=True)
    a.add_argument("--out",required=True)
    a=sub.add_parser("scale",help="Run metadata-only graph measurement in a fresh process")
    a.add_argument("--seed",type=int,default=11)
    a.add_argument("--artifacts",type=int,default=1000)
    a.add_argument("--depth",type=int,default=6)
    a.add_argument("--fan-in",type=int,default=8)
    a.add_argument("--sources",type=int,default=32,help="Distinct source records reused across graph roots")
    a.add_argument("--density",type=float,default=.5)
    a.add_argument("--conflict",type=float,default=.1)
    a.add_argument("--requests",type=int,default=1000)
    a.add_argument("--out",required=True)
    a=sub.add_parser("export",help="Collect runs and write IEEE LaTeX table rows")
    a.add_argument("--results",default="results")
    a.add_argument("--data",default="data")
    a.add_argument("--out",default="tables")
    a.add_argument("--pilot",action="store_true",help="Allow fewer than three seeds")
    a.add_argument("--include-generated",action="store_true",help="Include non-reviewed cases with export warning")
    a.add_argument("--include-fixtures",action="store_true",help="Smoke-test outputs only")
    a=sub.add_parser("compare",help="Exact paired McNemar test for two policy runs")
    a.add_argument("--left",required=True)
    a.add_argument("--right",required=True)
    a.add_argument("--out",required=True)
    a=sub.add_parser("holm",help="Correct a predeclared family of paired-test p-values")
    a.add_argument("--inputs",nargs="+",required=True)
    a.add_argument("--out",required=True)
    a=sub.add_parser("qrels",help="Union relevance pools or completed judgments")
    a.add_argument("--inputs",nargs="+",required=True)
    a.add_argument("--out",required=True)
    args=p.parse_args()
    if hasattr(args,"seed") and args.seed<0: p.error("seed must be nonnegative")
    for name in ("train","dev","test","count","review_count","queries","batch_size","micro_batch",
                 "max_length","eval_count","artifacts","depth","fan_in","sources","requests","bootstrap","max_tokens","max_context"):
        if hasattr(args,name) and getattr(args,name)<=0: p.error(name+" must be positive")
    for name in ("missing_lineage","density","conflict"):
        if hasattr(args,name) and not 0<=getattr(args,name)<=1: p.error(name+" must be in [0,1]")
    if args.command in {"policies","ablate"}:
        if args.method=="LLM" and not args.model: p.error("LLM requires --model")
        if not args.label:
            args.label="G (DToU)" if args.method=="G" else "L-"+args.model.split("/")[-1] if args.method=="LLM" else args.method
    if args.command=="rag" and args.retriever=="bm25" and args.backend!="numpy":
        p.error("BM25 uses --backend numpy; dense backends require --retriever dense")
    if args.command=="scale" and args.depth>args.artifacts:
        p.error("depth must not exceed artifacts")
    if args.command in {"policies","ablate","rag","train","scale"} and Path(args.out).exists():
        p.error("Output directory exists; choose a new --out to avoid overwriting a run")
    if args.command in {"prepare","cases"}:
        from data import prepare, build_cases
        (prepare if args.command=="prepare" else build_cases)(args)
    elif args.command=="train":
        from training import train
        train(args)
    elif args.command=="merge":
        from training import merge
        merge(args)
    elif args.command in {"export","compare","holm","qrels"}:
        from report import export,compare,holm,qrels
        {"export":export,"compare":compare,"holm":holm,"qrels":qrels}[args.command](args)
    else:
        from experiments import annotate,policies,pool,rag,scale
        {"annotate":annotate,"policies":policies,"ablate":policies,"pool":pool,"rag":rag,"scale":scale}[args.command](args)


if __name__=="__main__":
    main()
