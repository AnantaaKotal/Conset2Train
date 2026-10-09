"""Plain PyTorch LoRA loop, with a fixed step budget and completion-only loss."""
import math
import random
from pathlib import Path

from common import bootstrap, file_hash, read_json, read_jsonl, task_f1, write_json
from data import training_policies
from experiments import save_run
from models import LLM, parse_labels
from policy import artifact, decide, oracle, transfer


def train(args):
    if args.provider != "hf":
        raise ValueError("LoRA training needs local Hugging Face weights; OpenRouter is inference only")
    if Path(args.out).exists():
        raise ValueError("Use a new output directory")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(args.seed)
    base = Path(args.data)/args.dataset
    candidates = read_jsonl(base/"train.jsonl")
    tests = read_jsonl(base/"test.jsonl")[:args.eval_count]
    labels = sorted({l for r in candidates for l in r["labels"]})
    registry = training_policies(candidates,args.seed)
    selected, authorized, artifacts = [], set(), {}
    for r in candidates:
        a = transfer([artifact([registry[r["id"]]])],"source",registry,
                     remove=[(r["id"],"identifier")])
        artifacts[r["id"]] = a
        if oracle(a,registry,"train","improvement")=="allow":
            authorized.add(r["id"])
        if decide(a,registry,"train","improvement",method=args.method)=="allow":
            selected.append(r)
    if not selected:
        raise ValueError("No admitted training examples")
    tokenizer = AutoTokenizer.from_pretrained(args.model,revision=args.model_revision)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    device = args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    network = AutoModelForCausalLM.from_pretrained(args.model,revision=args.model_revision,torch_dtype=dtype).to(device)
    network = get_peft_model(network,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
                                target_modules=args.targets.split(","),task_type="CAUSAL_LM",bias="none"))
    network.print_trainable_parameters()
    network.config.use_cache = False
    network.train()
    optimizer = torch.optim.AdamW((p for p in network.parameters() if p.requires_grad),lr=args.learning_rate)
    truncated = 0
    encoded = []
    instruction = "Classify the customer input. Return only a JSON array of labels. Valid labels: " + str(labels)
    for r in selected:
        messages = [dict(role="system",content=instruction),dict(role="user",content=r["input"])]
        prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                   enable_thinking=False) if tokenizer.chat_template else instruction+"\n"+r["input"]+"\nAnswer:"
        import json
        response = json.dumps(r["labels"])
        pids = tokenizer.encode(prompt,add_special_tokens=False)
        tids = tokenizer.encode(response,add_special_tokens=False)+[tokenizer.eos_token_id]
        room = args.max_length-len(tids)
        if room < 1:
            raise ValueError("Label completion exceeds maximum sequence length")
        truncated += len(pids)>room
        pids = pids[-room:]
        encoded.append((pids+tids,[-100]*len(pids)+tids))
    steps = args.steps or math.ceil(args.epochs*len(candidates)/args.batch_size)
    rng = random.Random(args.seed)
    exposures = {r["id"]:0 for r in selected}
    pool = []
    losses = []
    accum = args.batch_size//args.micro_batch
    if args.batch_size % args.micro_batch:
        raise ValueError("batch-size must be divisible by micro-batch")
    for step in range(steps):
        optimizer.zero_grad()
        loss_value = 0
        for _ in range(accum):
            ids = []
            for _ in range(args.micro_batch):
                if not pool:
                    pool = list(range(len(encoded)))
                    rng.shuffle(pool)
                ids.append(pool.pop())
            max_len = max(len(encoded[i][0]) for i in ids)
            tokens, targets, masks = [], [], []
            for i in ids:
                x,y = encoded[i]
                pad = max_len-len(x)
                tokens.append(x+[tokenizer.pad_token_id]*pad)
                targets.append(y+[-100]*pad)
                masks.append([1]*len(x)+[0]*pad)
                exposures[selected[i]["id"]] += 1
            loss = network(input_ids=torch.tensor(tokens,device=device),
                           attention_mask=torch.tensor(masks,device=device),
                           labels=torch.tensor(targets,device=device)).loss
            (loss/accum).backward()
            loss_value += loss.detach().float().item()/accum
        torch.nn.utils.clip_grad_norm_(network.parameters(),1.0)
        optimizer.step()
        losses.append(dict(step=step+1,loss=loss_value))
        print(f"Training {step+1}/{steps}, loss={loss_value:.4f}",flush=True)
    # Save adapter after evaluation so the result folder remains an atomic run.
    model = LLM("hf",args.model,args.model_revision,device,args.max_tokens,args.max_context,args.seed)
    model.tokenizer, model.network, model.loaded = tokenizer,network,True
    network.config.use_cache = True
    network.eval()
    rows = []
    import json
    for i,r in enumerate(tests):
        response = model(instruction,r["input"])
        pred = parse_labels(response["text"],labels,args.dataset=="multiwoz")
        rows.append(dict(id=r["id"],group=r["id"],gold=r["labels"],pred=pred,response=response))
        print("Evaluation",i+1,"/",len(tests),flush=True)
    admitted_ids = {r["id"] for r in selected}
    metrics = dict(F1=task_f1([r["gold"] for r in rows],[r["pred"] for r in rows],labels,args.dataset=="multiwoz"),
                   UA=len(admitted_ids-authorized)/len(admitted_ids),
                   AC=len(admitted_ids & authorized)/len(authorized) if authorized else None,
                   unique_examples=len(selected),exposures=sum(exposures.values()),optimizer_steps=steps,
                   truncated_training_prompts=truncated)
    used_ids = {sid for sid,n in exposures.items() if n>0}
    adapter_artifact = transfer([artifacts[sid] for sid in sorted(used_ids)],"adapter",registry)
    save_run(args,metrics,rows,fixture=read_json(base/"manifest.json")["origin"].get("fixture",False),
             ci95=bootstrap(rows,lambda sample:{"F1":task_f1([r["gold"] for r in sample],[r["pred"] for r in sample],labels,args.dataset=="multiwoz")},args.seed),
             train_sha256=file_hash(base/"train.jsonl"),test_sha256=file_hash(base/"test.jsonl"),
             device=device,dtype=str(dtype),model_commit=getattr(network.config,"_commit_hash",None))
    out = Path(args.out)
    network.save_pretrained(out/"adapter")
    tokenizer.save_pretrained(out/"adapter")
    write_json(out/"manifest.json",dict(selected_ids=sorted(admitted_ids),used_ids=sorted(used_ids),exposures=exposures,
               artifact=adapter_artifact,registry=registry,losses=losses))


def merge(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from common import environment
    left = read_json(Path(args.left)/"manifest.json")
    right = read_json(Path(args.right)/"manifest.json")
    lr = read_json(Path(args.left)/"run.json")["settings"]
    rr = read_json(Path(args.right)/"run.json")["settings"]
    if lr["model"] != rr["model"] or lr["model_revision"] != rr["model_revision"]:
        raise ValueError("Adapters must have the same base model and revision")
    registry = dict(left["registry"])
    for sid,s in right["registry"].items():
        if sid in registry and s != registry[sid]:
            raise ValueError("Policy snapshots differ for shared source; reconcile them first")
        registry[sid] = s
    if Path(args.out).exists():
        raise ValueError("Use a new output directory")
    base = AutoModelForCausalLM.from_pretrained(lr["model"],revision=lr["model_revision"],torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base,Path(args.left)/"adapter",adapter_name="left")
    model.load_adapter(Path(args.right)/"adapter",adapter_name="right")
    model.add_weighted_adapter(["left","right"],[.5,.5],"merged",combination_type="cat")
    model.set_adapter("merged")
    model.save_pretrained(Path(args.out)/"adapter",selected_adapters=["merged"])
    merged = transfer([left["artifact"],right["artifact"]],"adapter",registry)
    rows = [dict(purpose=p,decision=decide(merged,registry,"deploy",p),gold=oracle(merged,registry,"deploy",p))
            for p in ("service","improvement","marketing")]
    write_json(Path(args.out)/"manifest.json",dict(artifact=merged,registry=registry,
               decisions=rows,shared_sources=len(set(left["selected_ids"]) & set(right["selected_ids"])),
               environment=environment(),combination="cat; exact .5/.5 sum of adapter deltas"))
    print("Merged adapter and lineage:",args.out,flush=True)
