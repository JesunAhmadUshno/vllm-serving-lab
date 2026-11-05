# vllm-serving-lab

LLM inference and serving, expanded in v3 to an inference optimisation lab: vLLM and SGLang, speculative decoding, KV-cache study, one benchmark harness.

Status: Learn DONE, Build DONE, Deep DONE, v1.0, 2026-11-05. Learn: simulated on CPU, no real model served. Build: a real open model (Qwen/Qwen2.5-0.5B-Instruct) served through an OpenAI-compatible API on BlackBox (RTX 5050), a thinking-budget study at three budgets, and a concurrent load test. Deep (compressed): a quantized-serving comparison, FP16 against bitsandbytes INT8 and INT4 NF4, same model, same 10 questions, budget 384, measured on BlackBox, plus the written thinking rule. The engine is Transformers on PyTorch, not vLLM (the node is Windows with no WSL); that substitution is named in level-2-build/README.md and stays the largest grade deduction. The rest of the v2 Deep scope is the v1.1 backlog in ROADMAP.md.
Project: P03. Fleet node: BlackBox (GPU). Wave: Wave 3.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: the core idea, built small and clean, with a teaching path.
- Level 2, Build: a real working system other people can run.
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases.

## Benchmarks

| Test | Result |
| --- | --- |
| Learn: KV bytes per token, example config 32 layers, 8 KV heads, head_dim 128, fp16 (simulated, method in level-1-learn/README.md) | 131,072 bytes per token |
| Learn: 12 labelled requests, prompt / thinking / answer tokens (simulated) | 1,470 / 1,590 / 915, total 3,975 tokens, KV total 521,011,200 bytes |
| Learn: paged blocks of 16 vs contiguous 256 token slab, waste tokens (simulated) | paged 73 waste of 4,048 allocated; contiguous 1,657 waste of 5,632 allocated |
| Learn: token-step batching, 32 tokens per step budget (simulated, deterministic) | 79 steps for 2,505 output tokens, 31.708861 tokens per step (not tokens per second) |
| Learn: unit tests | 20 tests in tests/test_learn_serving.py, all passing, plus scaffold smoke test |
| Build: thinking-budget study, budget 32 / 128 / 384 tokens (real model, BlackBox RTX 5050, greedy, 10 questions) | 0 of 10 / 0 of 10 / 8 of 10 correct; every generation capped at 32 and 128; at 384, 315.25 tokens and about 7.2 seconds per correct answer |
| Build: stop rule at the Final answer line (measured experiment) | never fired: the model ends with a boxed value or prose, not the instructed line; tokens per correct unchanged at 315.25 |
| Build: load test, 12 requests, concurrency 4, max 128 tokens | 12 of 12 succeeded, 0.477 requests per second, 61.05 completion tokens per second, p50 8.2146 s, p95 9.0854 s |
| Build: unit tests | 15 tests in tests/test_build_serving.py (API, streaming, usage math, extraction, study, load test), all passing |
| Deep: quantized comparison, FP16 (real model, BlackBox RTX 5050, greedy, 10 questions, budget 384) | 8 of 10 correct, 2,522 completion tokens, 55.94 tokens per second, 996,322,304 GPU bytes after load |
| Deep: quantized comparison, INT8 bitsandbytes load_in_8bit (same run) | 6 of 10 correct, 2,664 tokens, 10.60 tokens per second, 631,487,488 GPU bytes after load (36.62 percent smaller than FP16) |
| Deep: quantized comparison, INT4 bitsandbytes load_in_4bit NF4 (same run) | 5 of 10 correct, 1,302 tokens, 32.36 tokens per second, 473,807,872 GPU bytes after load (52.44 percent smaller than FP16) |
| Deep: unit tests | 14 tests in tests/test_deep_quant.py (extraction, summary math, comparison deltas, fake run, error lines), all passing |

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
