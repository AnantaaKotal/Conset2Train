# Consent2Train: simple experiment runner

Run everything through `python run.py`. There is no web app, orchestration service, or database server to deploy. The policy engine, BM25, policy metrics, graph scaling, and LaTeX export use Python's standard library. Models and dense-search backends are optional dependencies.

This is a runnable research prototype for the six-page manuscript. It implements the finite policy profile and main experiment blocks. **It does not contain measured paper results.** Human consent-case judgments, retrieval relevance judgments, the actual DToU baseline, and model runs must be supplied or performed before filling the corresponding tables.

## 1. Install and check

Python 3.10+ on Linux/macOS is recommended. `scale` uses the Unix `resource` module; on Windows use WSL.

```bash
python -m venv .venv
source .venv/bin/activate
python -m unittest discover -s tests -v
python run.py --help
```

For a small three-corpus **synthetic fixture** demo without installation or downloads:

```bash
python examples/smoke.py
```

It creates a dated `smoke-*` folder with example result files and table rows. These fixture values check the runner only; they must not be used as paper measurements.

For all optional features:

```bash
pip install -r requirements.txt
pip freeze > environment.txt
```

Or install only what you need:

```bash
# Download Bitext with Hugging Face datasets; unnecessary for a local CSV.
pip install 'datasets>=3,<5'

# Local HF inference and LoRA.
pip install 'torch>=2.4,<3' 'transformers>=4.51,<5' 'peft>=0.15,<1' 'accelerate>=1.2,<2'

# Dense retrieval and the three optional local backends.
pip install 'sentence-transformers>=3.4,<6' 'numpy>=1.26,<3' \
  'qdrant-client>=1.14,<2' 'chromadb>=1.0,<2' 'faiss-cpu>=1.10,<2'
```

Install the appropriate CUDA PyTorch build for your machine. Larger HF models and training benefit from a GPU. Training uses BF16 on supported CUDA devices and FP32 otherwise; this prototype does not implement quantization. Change the paper's settings to match actual runs rather than implying QLoRA was used.

## 2. Prepare each dataset

The defaults produce the paper's planned 2,000 train / 500 dev / 500 test records. Prepare once and reuse these splits across methods and seeds.

```bash
python run.py prepare --dataset abcd
python run.py prepare --dataset multiwoz
python run.py prepare --dataset bitext
```

Files appear under `data/<dataset>/`: `train.jsonl`, `dev.jsonl`, `test.jsonl`, and `manifest.json`. Each normalized record contains an ID, customer-only classification input, conversation turns, and labels. Target annotations are excluded from model input.

| Dataset argument | Source and label | Split handling |
|---|---|---|
| `abcd` | Official ABCD `abcd_v1.1.json.gz`; scenario subflow | Sample within official train/dev/test; use delexicalized turns |
| `multiwoz` | Official MultiWOZ 2.2 dialogue shards; service/domain set | Sample within official train/dev/test; multi-label classification |
| `bitext` | `bitext/Bitext-customer-support-llm-chatbot-training-dataset`; intent | Group duplicates first, then create approximately 80/10/10 splits and sample |

The Bitext grouping combines normalized exact matches and blocked character-4gram Jaccard similarity >= 0.85. Blocking uses first/last tokens and neighboring length buckets. It is a limited near-duplicate heuristic, not an exhaustive semantic paraphrase detector. It does not use intent labels for clustering. Validate this threshold on development data before the final experiment and describe it accurately in the paper.

For reproducible downloads, replace mutable branches with commit hashes:

```bash
python run.py prepare --dataset abcd --revision YOUR_GIT_COMMIT
python run.py prepare --dataset multiwoz --revision YOUR_GIT_COMMIT
python run.py prepare --dataset bitext --hf-revision YOUR_HF_COMMIT
```

If downloads are unavailable, provide original files locally:

```bash
# Directory containing abcd_v1.1.json.gz or abcd_v1.1.json
python run.py prepare --dataset abcd --raw /path/to/abcd

# Directory containing train/, dev/, test/ dialogue shards
python run.py prepare --dataset multiwoz --raw /path/to/MultiWOZ_2.2

# Directory containing the original CSV with instruction,response,intent columns
python run.py prepare --dataset bitext --raw /path/to/bitext
```

Existing raw downloads are reused. Keep the source licenses with your downloaded corpora; simulated consent is separate from each corpus's license. The code archive contains no corpus or model weights.

## 3. Build and independently review policy cases

```bash
mkdir -p cases
for dataset in abcd multiwoz bitext; do
  python run.py cases --dataset "$dataset" --cases "cases/$dataset.jsonl"
done
```

Each call creates 600 cases: 100 review candidates and 500 separate coverage cases. It also creates:

- `<dataset>.review.jsonl`: reviewer packet, including policies, lineage, current request, and ordered future requests. Generated answers are omitted.
- `<dataset>.review.csv`: blank reviewer1, reviewer2, adjudicated gold, future_gold, obligation_gold, and notes.
- `<dataset>.coverage.jsonl`: the other 500 automatically labeled cases.

Two reviewers independently assign current-request labels; adjudicate disagreements. `gold` must be `allow`, `deny`, or `conditional`. `future_gold` is a JSON array of decisions in the order of `future_requests` in the packet. `obligation_gold` is `fulfilled`, `unfulfilled`, or `not_applicable` for the supported removal prerequisite. Enter all fields for the reviewed cases. Review future requests independently too; the CSV stores their adjudicated labels. The reported agreement/kappa concerns the current request only.

```bash
for dataset in abcd multiwoz bitext; do
  python run.py annotate --cases "cases/$dataset.jsonl" \
    --labels "cases/$dataset.review.csv" --out "cases/$dataset.reviewed.jsonl"
done
```

This writes reviewed cases and `*.agreement.json`. No rows are marked reviewed automatically. For a quick **generated-case pilot** before review:

```bash
python run.py policies --dataset abcd --cases cases/abcd.coverage.jsonl \
  --method C2T --out results/generated-abcd-c2t-s11
```

Fifteen families test permits, training bans, unmet/completed removal, field-specific bans, adapter-purpose restrictions, excluded sources, mixed inputs, expiry, withdrawal, multiple authorities, adapter composition, missing lineage, unsupported guards, and entry-level index policies. They are controlled fixtures over real corpus IDs, not observed consent from the data subjects. The same reviewed scenario set can be reused for three measurement seeds; deterministic policy methods then have identical decisions and different timing samples. If you generate different policy assignments per seed, review each changed case set and report that larger annotation budget.

## 4. Table III: policy baselines

```bash
for dataset in abcd multiwoz bitext; do
  for seed in 11 23 37; do
    for method in U S C C2T; do
      python run.py policies --dataset "$dataset" --seed "$seed" \
        --cases "cases/$dataset.reviewed.jsonl" --method "$method" \
        --out "results/core-$dataset-$method-s$seed"
    done
  done
done
```

`U` allows everything. `S` uses source-scoped static controls. `C` uses all ancestral fields/restrictions and ignores completion evidence. `C2T` uses typed residual rules, contributing lineage, deny-overrides, and exact completion evidence. The source-level oracle independently enumerates original rules and never reads C2T's residual rule references.

### Try any compatible Hugging Face LLM

```bash
python run.py policies --dataset abcd --cases cases/abcd.reviewed.jsonl \
  --method LLM --provider hf --model Qwen/Qwen3-4B --label L-Qwen \
  --seed 11 --out results/core-abcd-qwen-s11
```

Replace the model ID and, if needed, `--model-revision` and `--device cuda:0`. Compatible models are text causal LMs supported by `AutoModelForCausalLM`; multimodal and encoder-only models are not supported. Gated models require accepted access terms and an `HF_TOKEN` environment variable or Hugging Face login. Remote custom model code is not enabled.

The paper's other model examples:

```bash
# Repeat the command above using these model/label pairs.
# meta-llama/Llama-3.2-3B-Instruct          --label L-Llama
# HuggingFaceTB/SmolLM2-1.7B-Instruct     --label L-SmolLM
# YourOrg/your-compatible-model          --label L-YourModel
```

Repeat for each dataset and seeds 11, 23, 37. New labels produce extra export rows, so trying another model does not overwrite the paper's Qwen/Llama/SmolLM rows. Models use greedy generation. Qwen thinking is disabled through its chat template; other templates may ignore that argument. The default common context budget is 8,192 tokens; overlong local prompts stop rather than silently truncate policies. Adjust the budget to the supported context size of every compared model.

### Try an OpenRouter LLM

```bash
export OPENROUTER_API_KEY='YOUR_KEY'
export OR_MODEL='vendor/model-id-from-openrouter'

python run.py policies --dataset abcd --cases cases/abcd.reviewed.jsonl \
  --method LLM --provider openrouter --model "$OR_MODEL" --label L-OpenRouter \
  --seed 11 --out results/core-abcd-openrouter-s11
```

Use the exact OpenRouter model ID, which may differ from the HF repository ID. There is no OpenAI SDK dependency: the adapter calls OpenRouter's chat-completions API directly. It records returned model identity and token usage without saving the API key. Remote inference sends the serialized case to the selected provider and incurs that provider's usage charges. This makes provider choice part of the experiment settings. Remote services may not honor seeds identically; record the resolved model and provider conditions. OpenRouter enforces its service's context limit; `--max-context` is a local HF input check. Ensure remote prompts fit the same comparison budget when reporting cross-provider results.

One LLM case includes its current request **and multiple future probes**. Start with a small case file to check output format and cost. `--cache cache` optionally stores responses for correctness runs; cached calls are marked and are unsuitable for latency claims. RAG latency runs never enable caching. API/network errors stop the run; malformed successful model responses count as `invalid` decision errors and block execution. They are not converted to correct deny labels.

### Import the actual DToU baseline

Export its predictions to CSV:

```text
case_id,pred,future_pred
s11:case:0,allow,"[""allow"", ""deny"", ...]"
```

`future_pred` must be a complete ordered JSON array matching the review packet's future requests. The ellipsis above is explanatory, not valid input.

```bash
python run.py policies --dataset abcd --cases cases/abcd.reviewed.jsonl \
  --method G --predictions dtou-abcd.csv --seed 11 \
  --out results/core-abcd-dtou-s11
```

This code does not reimplement the published DToU reasoner. Import actual predictions, document extensions and unsupported cases, and repeat the comparison protocol. Without these predictions, its row remains `TBD`.

## 5. Table IV: Hugging Face LoRA fine-tuning

OpenRouter supplies inference, not model weights for this training loop. To compare local adaptation, use `--provider hf`.

```bash
python run.py train --dataset abcd --model Qwen/Qwen3-4B \
  --method C2T --seed 11 --out results/train-abcd-qwen-C2T-s11
```

Full paper matrix: 3 datasets x 3 models x 3 selections x 3 seeds = 81 runs.

```bash
for dataset in abcd multiwoz bitext; do
  for model in Qwen/Qwen3-4B meta-llama/Llama-3.2-3B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct; do
    name=${model##*/}
    for method in U C C2T; do
      for seed in 11 23 37; do
        python run.py train --dataset "$dataset" --model "$model" \
          --method "$method" --seed "$seed" \
          --out "results/train-$dataset-$name-$method-s$seed"
      done
    done
  done
done
```

Defaults: rank 8, alpha 16, dropout .05, query/value projection targets, AdamW at 2e-4, maximum sequence length 512, effective batch 16, micro-batch 1. The shared step budget is `ceil(2 * candidate_count / 16)`, independent of admitted subset size. Eligible examples replay in shuffled cycles. Only label completions contribute to loss; padding and prompt tokens are masked. Training truncates long prompts from the left while preserving the completion and logs the count. There is no held-out data in training.

For another model, choose appropriate LoRA targets, for example `--targets q_proj,v_proj`; incompatible names raise an error. For a small execution check use `--steps 2 --eval-count 5`, a small supported model, and a separate pilot results root. Default evaluation uses up to 500 test records.

Outputs include a LoRA `adapter/`, `manifest.json` with admitted/actually used IDs, exposure counts, contributor lineage and loss logs, and held-out predictions. `UA` is unauthorized admitted examples / all admitted examples; `AC` is admitted authorized examples / authorized candidates. F1 is corpus-specific task macro-F1, with per-label binary F1 for MultiWOZ domain sets. Invalid completions receive no label credit. The completion-format task is a controlled utility measure, not the corpora's original dialogue-system benchmark.

## 6. Table V: RAG retrieval, backends, and LLM generation

### Build relevance pools first

Run these for each dataset (`abcd`, `multiwoz`, `bitext`), on test queries, and union their CSVs. The canonical example below uses ABCD. The pool includes top-20 unrestricted and oracle-eligible results from each retriever; it does **not** invent relevance grades from intent labels.

```bash
mkdir -p qrels
python run.py pool --dataset abcd --retriever bm25 --out qrels/abcd-bm25.csv
python run.py pool --dataset abcd --retriever dense \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 --out qrels/abcd-minilm.csv
python run.py pool --dataset abcd --retriever dense \
  --embedding-model BAAI/bge-small-en-v1.5 --out qrels/abcd-bge.csv
python run.py qrels --inputs qrels/abcd-bm25.csv qrels/abcd-minilm.csv qrels/abcd-bge.csv \
  --out qrels/abcd.csv
```

Review and fill `grade` with 0 (irrelevant), 1 (partly relevant), or 2 (highly relevant). Repeat for each corpus. Union pools from additional retrievers and from policy seeds 23/37 to cover changed eligibility. For development comparisons use `--query-split dev`; do not use test results to select the strongest baseline. RAG refuses unjudged returned documents by default. `--allow-unjudged` is only for pilots; it treats them as nonrelevant and reports their rate.

Recall@5 and nDCG@10 use **judged authorized relevance in this pool**, not exhaustive corpus relevance. Document this pooling limitation. Authorization-empty queries and empty contexts are counted separately; utility is averaged over queries with an authorized judged answer. Changing the pool after inspecting test-method performance requires transparent reporting.

### Run the canonical configuration

```bash
python run.py rag --dataset abcd --method C2T --retriever dense \
  --embedding-model BAAI/bge-small-en-v1.5 --backend qdrant \
  --provider hf --model Qwen/Qwen3-4B --qrels qrels/abcd.csv \
  --label 'C2T canonical' --seed 11 --out results/rag-abcd-canonical-s11
```

Repeat for all datasets and seeds. Source policies are independently assigned within label strata. Only permitted entries are embedded/indexed, eligibility filters apply before top-k, and entries are rechecked before context assembly. BGE queries receive its retrieval prefix; all dense vectors are normalized. No held-out dialogue is indexed. Chunks contain at most 128 whitespace tokens, overlap by one turn when feasible, and split unusually long turns. Actual embedding truncations are logged.

Change **one factor at a time**:

| Table row label | Change to canonical command |
|---|---|
| `C2T + Llama` | `--model meta-llama/Llama-3.2-3B-Instruct` |
| `C2T + SmolLM` | `--model HuggingFaceTB/SmolLM2-1.7B-Instruct` |
| `C2T + BM25` | `--retriever bm25 --backend numpy` |
| `C2T + MiniLM` | `--embedding-model sentence-transformers/all-MiniLM-L6-v2` |
| `C2T + Chroma` | `--backend chroma` |
| `C2T + FAISS` | `--backend faiss` |
| `Oracle` | `--method Oracle`, keep canonical model/retriever/backend |
| `Strongest baseline` | Select `--method S` or `C` using **development** cases/queries, then freeze selection |

Give each command the matching `--label` and a distinct output folder. Other model IDs or configurations work with custom labels and add rows. OpenRouter generation is a direct swap:

```bash
python run.py rag --dataset abcd --method C2T --retriever dense --backend qdrant \
  --provider openrouter --model "$OR_MODEL" --qrels qrels/abcd.csv \
  --label 'C2T + MyOpenRouterModel' --seed 11 --out results/rag-abcd-openrouter-s11
```

Qdrant runs in local in-memory mode; FAISS uses an exact `IndexFlatIP` over eligible vectors. FAISS subset rebuilding is included in per-query latency. Chroma's native search is approximate: default exact mode reads eligible vectors from Chroma and ranks them in NumPy, labeled `exact_adapter` in metadata. For a **separate native ANN experiment**, pass `--backend chroma --approximate` and a distinct label/results root. Do not report Chroma's exact adapter timing as native ANN timing. These local backends do not measure distributed/server deployment behavior.

Omit `--model` for a cheap retrieval-only pilot. Its retrieval scores are usable with appropriate judgments, but its latency is **not** end-to-end RAG latency and export leaves that p95 cell `TBD`. With a generator, p95 measures gate/filtering, query embedding/search, context assembly, and label generation. It excludes initial model loading and index construction, includes remote network time, and uses no response cache. Generator output additionally produces task F1, even though the manuscript's Table V currently reports retrieval utility. Report that extra metric in the text or add a column if comparing answer utility.

## 7. Table VI: ablations and missing-lineage sensitivity

```bash
for dataset in abcd multiwoz bitext; do
  for seed in 11 23 37; do
    for method in C2T no_type record_only no_evidence index_wide; do
      python run.py ablate --dataset "$dataset" --cases "cases/$dataset.reviewed.jsonl" \
        --method "$method" --seed "$seed" \
        --out "results/ablate-$dataset-$method-s$seed"
    done
  done
done

for rate in 0.01 0.05 0.10; do
  python run.py policies --dataset abcd --cases cases/abcd.reviewed.jsonl \
    --method C2T --missing-lineage "$rate" --out "results/missing-abcd-$rate"
done
```

`no_type` removes type scopes; `record_only` loses field granularity; `no_evidence` forgets completion; `index_wide` combines audit siblings for index-entry decisions. Missing-lineage trials explicitly mark affected metadata incomplete and deny execution; they test behavior when incompleteness is **detected**, not the ability to discover secretly omitted edges. Their coverage loss is recorded and they are excluded from primary table averages.

## 8. Table VII: graph scaling

Run each measurement in its own process so peak resident memory is comparable:

```bash
for n in 1000 10000 100000; do
  for seed in 11 23 37; do
    python run.py scale --artifacts "$n" --depth 6 --fan-in 8 --sources 32 \
      --density 0.5 --conflict 0.1 --seed "$seed" --out "results/scale-$n-s$seed"
  done
done
```

This panel reuses 32 distinct source records across roots, preventing contributor sets from growing without bound. It measures scaling in artifact count under that fixed source population; **it does not establish scaling to 100,000 distinct contributors**. Vary `--sources` separately, and report it in the paper. `prop_s` measures graph construction and policy transfer; `bytes_artifact` is serialized lineage/policy metadata divided by artifacts; `memory_mb` is whole-process peak RSS. Gate latency is cold evaluation without a decision cache. Withdrawal reports affected descendants and traversal time.

For sensitivity, vary depth 3/6/12, fan-in 2/8/32, density .25/.5/.75, and conflict 0/.1/.25, using separate results roots. The exporter only uses the depth-6/fan-in-8 panel for Table VII. Keep the same source population and density for its three-seed rows.

## 9. Collect results and fill Overleaf tables

```bash
python run.py export --results results --data data --out tables
```

| Paper table | Generated file | Contents |
|---|---|---|
| I | No results file | Operator specification already in the manuscript |
| II | `table_II_datasets.tex` | Actual prepared split counts and reviewed-case count |
| III | `table_III_core.tex` | FAR, FDR (%), decision macro-F1, future semantic agreement |
| IV | `table_IV_finetuning.tex` | Task F1, UA (%), AC (%) for each model/selection/corpus |
| V | `table_V_rag.tex` | Authorized Recall@5, nDCG@10, UCR (%), end-to-end p95 ms |
| VI | `table_VI_ablations.tex` | Corpus-macro-averaged FAR, FDR, SA, obligation accuracy |
| VII | `table_VII_scaling.tex` | Gate p95, propagation seconds, bytes/artifact, RSS MB |

Paste rows between each table's `\midrule` and `\bottomrule`, or replace those rows with `\input{tables/table_III_core.tex}` after uploading the files. They contain rows, not nested `tabular` environments. Exports use three decimal places; raw metrics are fractions and exported rates are multiplied by 100 exactly once.

The default export requires seeds **11, 23, 37**, excludes synthetic fixture runs and unreviewed policy cases, rejects duplicate run identities/inconsistent configurations, and leaves missing values as `TBD`. Use the paper's exact `--label` strings above. Pilot-only export:

```bash
python run.py export --results pilot-results --out pilot-tables --pilot --include-generated
```

Do not describe those generated-case exports as human-reviewed results. `--include-fixtures` is available only for checking the tool chain. Warnings appear in `EXPORT_NOTES.txt`. Keep development, test, generated coverage, and sensitivity runs in separate results roots to prevent accidental pooling.

Every completed run writes:

- `metrics.json`: numeric metrics; unavailable metrics use JSON `null`, not fabricated zero.
- `raw.jsonl`: individual outcomes, predictions, timings, and future probes/context IDs.
- `run.json`: settings, versions, data/source hashes, and within-run cluster bootstrap 95% intervals where applicable.
- `all_runs.csv` / `runs.json` at export time: flat per-run results and full metadata.

Means in tables average the three paired seeds. Bootstrap intervals use 1,000 group resamples within each run: conversation/scenario groups for policy cases and whole queries for RAG/task utility. They are **not** pooled across-seed confidence intervals. Report per-seed results and the chosen uncertainty convention explicitly. Core runs also provide a one-sided binomial 95% bound when FAR is observed as zero; that bound assumes independent denied cases and must not be presented as group-adjusted coverage.

For paired decision comparisons and Holm correction:

```bash
python run.py compare --left results/core-abcd-S-s11 --right results/core-abcd-C2T-s11 \
  --out tests_abcd.json
# Create the other predeclared paired comparisons first, then correct the whole family.
python run.py holm --inputs tests_abcd.json tests_multiwoz.json tests_bitext.json --out holm.json
```

The exact McNemar test uses case-level discordances and assumes independent paired cases. If multiple cases reuse a conversation, treat this as exploratory or use a group-aware inference procedure on the raw logs before making inferential claims. Do not count repeated deterministic runs as independent consent judgments.

## 10. Adapter merge and withdrawal checks

```bash
python run.py merge --left results/train-left --right results/train-right --out merged
```

This uses same-base/revision PEFT adapters and a `cat` composition with equal weights, giving the exact weighted sum of their low-rank deltas. It unions actual contributors and records deployment decisions. It rejects inconsistent policies for shared sources. For the manuscript's **disjoint subset** experiment, prepare two normalized data roots with disjoint training IDs, train both with the same model/revision/seed, then merge. The default three-seed full-dataset adapters are not disjoint subsets. Shared-source count is reported in the merge manifest. PEFT saves the merged named adapter under `adapter/merged/`.

`policy.invalidate(source_id, graph)` finds contributing descendants; version changes force live-rule reevaluation. Tests cover withdrawing a grant after an artifact was created. No checkpoint forgetting, verified unlearning, distributed commit token, or interruption of active training is implemented. Withdrawals can identify/block affected artifacts, not prove removal of knowledge from weights.

## 11. Scope and validation

This deliberately small prototype uses whole-field atoms (`text`, fictional `identifier`) rather than arbitrary character-span lineage. Field removal is a trusted structured operation and records exact field evidence; it does not anonymize original free text. The supported guards are `before` and `removed`; other guards, including recurring duties, fail closed and require language extensions. There is no natural-language policy-extraction pipeline, remote service policy enforcement, or protection against a compromised adapter. Simulated source fields and the corpus text are separate; the fictional identifier is never fed to the models after removal. Revise broader manuscript claims to match these boundaries.

The 15 generated families provide a starting benchmark, not complete coverage of all rule combinations. Additional stratified density/conflict sweeps, independent reviewers, real baseline integration, and actual model/backend execution are research work to finish. Runtime logs preserve actual settings so the paper can state what was done.

Validation performed with this delivery: standard-library unit tests and the full synthetic CPU workflow (preparation, cases, gates, BM25, scaling, and export). OpenRouter request/response handling is tested with a mock endpoint. Real corpus downloads, HF inference/LoRA, and optional vector backends require your network/model access and were not executed in the delivery environment. Use a small real-data/model run to validate your installed stack before the full matrix.

Source files: `policy.py` (semantics), `data.py` (corpora/cases), `models.py` (HF/OpenRouter), `search.py` (retrieval), `experiments.py` (runs), `training.py` (LoRA/merge), `common.py` (I/O/metrics), `report.py` (tables/statistics), and `run.py` (CLI).

References for adapters and formats:

- ABCD: https://github.com/asappresearch/abcd
- MultiWOZ 2.2: https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2
- Bitext: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- HF generation: https://huggingface.co/docs/transformers/main_classes/text_generation
- PEFT: https://huggingface.co/docs/peft/en/package_reference/peft_model
- OpenRouter: https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
- Qdrant: https://qdrant.tech/documentation/search/filtering/
- Chroma: https://docs.trychroma.com/docs/querying-collections/metadata-filtering
- FAISS: https://github.com/facebookresearch/faiss
