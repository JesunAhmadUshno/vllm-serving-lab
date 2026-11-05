# Release notes draft: vllm-serving-lab

Status: DRAFT notes for releases that already exist. Dated 2026-11-05.
These notes summarise the existing GitHub releases. No new release is
created by this file.

Releases verified with gh release list on 2026-11-05:
v0.1, v0.5, and v1.0, at the URLs below.

## v0.1, Learn: deterministic serving simulator

URL: https://github.com/JesunAhmadUshno/vllm-serving-lab/releases/tag/v0.1

- Deterministic serving simulator on CPU, standard library only:
  token accounting with thinking tokens logged apart from answer
  tokens, KV cache math, paged blocks of 16 against contiguous
  slabs of 256, and a token-step batching simulation. No real model
  was served at this level.
- Labelled set: 12 requests totalling 1,470 prompt tokens, 1,590
  thinking tokens, and 915 answer tokens (3,975 total). Thinking is
  63.5 percent of generated tokens on this set.
- Measured (simulated) run: KV cache 131,072 bytes per token and
  521,011,200 bytes in total for the example config; paged waste
  73 tokens against 1,657 contiguous; batching at 79 steps and
  31.708861 tokens per step (tokens per step, not per second).
- Tests: 20 unit tests in tests/test_learn_serving.py, plus the
  scaffold smoke test.
- Limits carried in the release: simulated only, a labelled
  teaching set, no prefill cost, no scheduling delay, and no real
  vLLM or GPU number claimed.

## v0.5, Build: real model serving and the thinking-budget study

URL: https://github.com/JesunAhmadUshno/vllm-serving-lab/releases/tag/v0.5

- A real open model (Qwen/Qwen2.5-0.5B-Instruct, 494M parameters)
  served through an OpenAI-compatible API on BlackBox (NVIDIA
  GeForce RTX 5050 Laptop GPU), standard library HTTP server,
  streaming, and a usage object with thinking and answer tokens
  counted apart. Engine: Transformers on PyTorch. vLLM itself is
  not used (Windows node, no WSL); the substitution is named in
  the release and the level README.
- Thinking-budget study, same 10 questions, greedy decoding:
  0 of 10 correct at budget 32 (320 completion tokens), 0 of 10
  at 128 (1,280 completion tokens), 8 of 10 at 384 (2,522
  completion tokens, 315.25 tokens per correct, 7.196 seconds
  per correct). Every generation at 32 and 128 hit the cap.
- Stop rule experiment: a stop rule on the Final answer line was
  implemented and the study rerun with identical token counts.
  The rule never fired, because the model does not emit that line.
- Load test: 12 of 12 requests succeeded at 0.477 requests per
  second and 61.05 completion tokens per second, p50 8.2146
  seconds, p95 9.0854 seconds. The stop-rule server measured
  slower on the same test (0.387 requests per second).
- Tests: 15 Build unit tests, 35 in total with Learn.
- Limits carried in the release: one request generated at a time
  (no continuous batching, so load measures queueing), a 0.5B
  model ceiling, and the container drill not done on this node.

## v1.0, Deep: quantized-serving comparison and the thinking rule

URL: https://github.com/JesunAhmadUshno/vllm-serving-lab/releases/tag/v1.0

- Compressed Deep under the December Compression Protocol: a
  quantized-serving comparison on the Build model and the same
  10 questions at budget 384, greedy decoding, each precision in
  its own subprocess, plus the written thinking rule derived
  from the Build study. Full per-question rows are in
  level-3-deep/results_quant.json.
- FP16 (no quantization): 8 of 10 correct, 2,522 completion
  tokens, 45.083 wall seconds, 55.94 tokens per second, 315.25
  tokens per correct, 996,322,304 GPU bytes after load,
  1,018,233,856 peak GPU bytes. The FP16 rows reproduce the
  Build 384 rows exactly, token for token.
- INT8 (bitsandbytes load_in_8bit): 6 of 10 correct, accuracy
  delta -0.200, 2,664 completion tokens, 10.60 tokens per
  second, 444.00 tokens per correct, 631,487,488 GPU bytes
  after load (36.62 percent smaller than FP16), throughput
  ratio 0.189.
- INT4 (bitsandbytes load_in_4bit, NF4): 5 of 10 correct,
  accuracy delta -0.300, 1,302 completion tokens, 32.36 tokens
  per second, 260.40 tokens per correct, 473,807,872 GPU bytes
  after load (52.44 percent smaller than FP16), throughput
  ratio 0.578.
- Failed routes in the final aggregate run: none.
- Tests: 14 Deep unit tests, 49 in total with Learn and Build.
- Grade: Deep 7,600 of 10,000, deductions named in
  level-3-deep/README.md. The largest is that vLLM itself is
  still not the engine.
- The rest of the v2 Deep scope is the v1.1 backlog in
  ROADMAP.md (speculative decoding, vLLM itself on Linux, the
  container drill, AWQ and GPTQ rows, larger question sets).
  It is not part of v1.0 and is not claimed by these notes.
