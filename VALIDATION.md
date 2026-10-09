# Delivery validation

Validated with Python 3.12.14 in the delivery environment:

- Standard-library semantic and adapter regression tests.
- CLI workflow: synthetic dataset preparation, case generation, reviewer import, policy evaluation, BM25 pooling/retrieval, graph scaling, and LaTeX export.
- Unknown/missing lineage, exact prerequisite evidence, filtered contributors, adapter-purpose propagation, same-base composition metadata, and live withdrawal checks.
- Mock OpenRouter request/response protocol and malformed-output handling.
- Missing results stay TBD; unreviewed and fixture runs are excluded by default; retrieval-only timing is not exported as RAG generation timing.

These checks validate implementation behavior, not empirical paper claims.

Not executed here: real corpus downloads (external downloads were blocked), HF model inference/LoRA, optional Chroma/Qdrant/FAISS adapters, and live OpenRouter calls (no credentials supplied). Run a small real-data/model trial after installing the optional dependencies. Package version ranges are compatibility targets, not a tested lockfile; preserve `pip freeze`, model/data revisions, and run metadata for your actual experiment.

The actual DToU implementation and human review/relevance judgments are external inputs. No simulated judgments or model scores are shipped as paper results.
