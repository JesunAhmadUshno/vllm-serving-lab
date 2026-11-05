# DEEP-SPEC: vllm-serving-lab (P03), Level 3 Deep (compressed)

Phase skill: spec-driven-development (house engineering pipeline).
Compression basis: December Compression Protocol. Deep is the single
hardest capability in the SPEC Deep scope, built, tested, benchmarked,
with failure modes written. The rest of the v2 Deep scope is named in
the repo-root ROADMAP.md as the v1.1 backlog.

## Scope chosen

A quantized-serving comparison. One model family, one question set,
one thinking budget, three precision rows, measured values only:

- Model: Qwen/Qwen2.5-0.5B-Instruct, the Build model (same family,
  same checkpoint for every row).
- Questions: the same 10 verifiable questions as Build
  (level-2-build/questions.json), greedy decoding.
- Budget: 384 new tokens, the only budget that scored in Build.
- Rows:
  - fp16: Transformers, torch_dtype float16, no quantization (the
    Build configuration).
  - int8: bitsandbytes load_in_8bit through Transformers
    BitsAndBytesConfig.
  - int4: bitsandbytes load_in_4bit, NF4, compute dtype float16.
- Per row: correct count and accuracy under the Build extraction rule
  (number after "Final answer:" when present, else last number,
  tolerance 0.01), completion tokens, wall seconds for the 10
  questions, completion tokens per second, tokens per correct answer,
  GPU memory allocated after load and peak GPU memory during the run
  (torch.cuda memory counters, measured).

## Honesty rules for this level

- Never claim a quantization that did not run. Each config runs in
  its own process; a config that fails is recorded in
  results_quant.json under failed_routes with the exact error line,
  and the README names it as failed.
- Memory is reported as measured CUDA allocation, the same counter
  for every row. No theoretical weight size is presented as measured
  memory. Theoretical bytes per weight (2, 1, 0.5) are labelled
  theoretical where they appear.
- The fake backend exists for CI only and is never reported as a
  model. Its deterministic run writes results_deep.json, labelled
  fake in the file itself.

## Also in this level

- A written thinking rule, derived from the Build study numbers,
  in level-3-deep/README.md: when more thinking pays and when it
  wastes money.
- A failure-modes section in level-3-deep/README.md.
- Repo-root ROADMAP.md: the rest of the v2 Deep scope (speculative
  decoding, serving with vLLM itself, the container drill) as the
  named v1.1 backlog.

## Acceptance

- Unit tests for the comparison and scoring logic pass with no GPU
  and no model download (fake backend), in CI as in earlier levels.
- results_quant.json committed, with per-question rows for every row
  that ran and exact errors for every route that failed.
- Full suite (Learn, Build, Deep) green locally and in CI.
- Root README carries the Deep benchmark row; DOSSIER.md updated to
  ship state; release v1.0 tagged after the PR merges with green CI.
