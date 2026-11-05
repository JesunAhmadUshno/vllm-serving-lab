# vllm-serving-lab

LLM inference and serving, expanded in v3 to an inference optimisation lab: vLLM and SGLang, speculative decoding, KV-cache study, one benchmark harness.

Status: Learn DONE, 2026-11-05 (Level 1, simulated on CPU, no real model served; BlackBox GPU node was unreachable this cycle, NEEDS-BLACKBOX-ON for real GPU serving, which is Build scope). Build and Deep not started.
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

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
