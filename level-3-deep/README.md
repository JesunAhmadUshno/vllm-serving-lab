# Level 3, Deep: vllm-serving-lab

Status: Deep DONE, 2026-11-05 (compressed Deep, December
Compression Protocol). A quantized-serving comparison with measured
values only, plus the written thinking rule from the Build study.
The rest of the v2 Deep scope is the v1.1 backlog in the repo-root
ROADMAP.md.

## What this level builds

- `quant_compare.py`: the comparison core, standard library only.
  Scoring (the Build extraction rule, kept identical so Deep rows
  compare with Build rows), summary math, comparison deltas, and a
  deterministic FakeQuantBackend for CI. Never reported as a model.
- `run_deep.py`: the deterministic fake run. Writes
  `results_deep.json`, labelled fake in the file. Counts only, no
  wall times, byte-identical across runs.
- `run_quant.py`: the serving-node run. Each precision runs in its
  own subprocess (clean GPU memory, a failed load cannot take the
  other rows down), partial results are aggregated into
  `results_quant.json`, and failed routes are recorded with their
  exact error line.
- Tests: `tests/test_deep_quant.py` (extraction, summary arithmetic
  pre-computed by hand, memory saving, comparison deltas, fake-run
  determinism, error-line picking).

## Method (real run)

Model: Qwen/Qwen2.5-0.5B-Instruct, the Build checkpoint, public,
downloaded unauthenticated. Node: BlackBox, NVIDIA GeForce RTX 5050
Laptop GPU (8151 MiB). Stack: Python 3.14.7, torch 2.11.0+cu128,
transformers 5.15.0, bitsandbytes 0.50.2, accelerate 1.15.0.
Questions: the same 10 Build questions. Budget: 384 new tokens, the
only budget that scored in Build. Greedy decoding, in-process
generation (not through the HTTP server). Scoring: number after
"Final answer:" when present, else the last number, tolerance 0.01.
Memory: torch.cuda memory_allocated after load, and
max_memory_allocated across the 10 questions, the same counters for
every row. One run per config; no variance bounds are claimed.

## Measured results: quantized-serving comparison

| Config | Correct | Accuracy | Completion tokens | Wall (s) | Tokens per second | Tokens per correct | GPU after load (bytes) | Peak GPU (bytes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16, no quantization | 8 of 10 | 0.800 | 2,522 | 45.083 | 55.94 | 315.25 | 996,322,304 | 1,018,233,856 |
| INT8, bitsandbytes load_in_8bit | 6 of 10 | 0.600 | 2,664 | 251.311 | 10.60 | 444.00 | 631,487,488 | 653,112,832 |
| INT4, bitsandbytes load_in_4bit NF4 | 5 of 10 | 0.500 | 1,302 | 40.230 | 32.36 | 260.40 | 473,807,872 | 496,118,784 |

Deltas against FP16, computed in code from the table above and
verified before any claim was written:

| Config | Memory saving vs FP16 (after load) | Accuracy delta | Throughput ratio |
| --- | --- | --- | --- |
| INT8 | 36.62 percent smaller | -0.200 | 0.189x |
| INT4 NF4 | 52.44 percent smaller | -0.300 | 0.578x |

What the table teaches, from the per-question rows in
`results_quant.json`:

- The FP16 row reproduces Build exactly: same 8 correct, same
  2,522 completion tokens, and every per-question extracted value
  and token count identical to the Build 384 row. Greedy decoding
  is deterministic here, and the Deep table stands on the Build
  table.
- INT8 saves about a third of the GPU memory and loses two
  questions (q01 extracts 2.0 against 4.25, q09 extracts 3 against
  10), and it is far slower on this node: 10.60 tokens per second
  against 55.94. At 0.5B scale the bitsandbytes 8-bit matmul
  overhead costs more time than the smaller weights save. INT8
  here is a memory tool, not a speed tool.
- INT4 NF4 saves about half the memory and loses three questions
  net, but the losses are not a simple blur: it gains q03, the
  discount-and-tax question FP16 gets wrong (74.8, correct), and
  loses q06 in 9 tokens (extracts 1375.49, an early-stop answer).
  Its lower tokens per correct (260.40) is not an efficiency win:
  it spends less because it finishes, or gives up, earlier, and
  scores less.
- No row dominates. The serving choice on this evidence: FP16 when
  accuracy matters and 1 GB of GPU memory is available, INT4 NF4
  when memory is the binding constraint and a 0.500 accuracy on
  this set is acceptable, INT8 only when its specific memory point
  (about 0.63 GB) is the one that fits.

Failed routes in the final aggregate run: none. Every config that
was attempted ran and is in the table. The environment failures
found on the way are in Failure modes, with their exact errors.

## The thinking rule (from the Build study)

Build measured the same 10 questions at budgets of 32, 128, and
384 tokens: 0 of 10, 0 of 10, and 8 of 10 correct, at 320, 1,280,
and 2,522 completion tokens. The rule those numbers support:

1. A thinking budget pays only when it is above the model's
   natural completion length for the task class. This model
   completes these questions in about 190 to 330 tokens. Budgets
   of 32 and 128 cut every generation mid-working, so the spend
   bought zero correct answers: 1,600 tokens for nothing. Measure
   the completion-length distribution first; set the budget above
   its high end, not at its mean.
2. A budget is a cap, not a target. At 384 the model stopped on
   its own for 9 of 10 questions and used 2,522 tokens, not 3,840.
   Raising a cap above the completion length costs nothing extra
   for questions that finish; it only bounds the runaway case
   (q10 used the full 384 and was still wrong).
3. More thinking does not fix a wrong method. The two failures at
   384 are capability errors at full length (a misapplied discount
   and tax; a counting answer of 90 against 6). Extra budget for
   those questions buys more tokens at the same accuracy. Spend
   on a better model or a better prompt for that class, not on a
   larger cap.
4. Cost per correct answer is the unit to watch, not cost per
   request: 315.25 tokens and about 5.6 to 7.2 seconds per correct
   answer at 384 in the two FP16 runs (Deep in-process and Build
   over HTTP). A cheaper per-request setting that scores zero is
   infinitely expensive per correct answer.
5. Stop rules only save money when they match the model's real
   output format. Build measured an unmatched "Final answer:" stop
   rule at exactly zero savings, with slower load-test throughput
   from its per-step check.

## Failure modes

- Quantization accuracy loss is question-shaped, not uniform:
  INT4 gains one question FP16 misses and loses others by
  early stopping (q06, 9 tokens). A single accuracy number hides
  this; the per-question rows are the evidence.
- Small-model quantization overhead: bitsandbytes INT8 is 5.3x
  slower than FP16 here. Quantization marketing assumes large
  models where weight bandwidth dominates; at 0.5B on a laptop
  GPU, dequantization overhead dominates instead.
- Missing dependency, exact error (found in the first probe,
  fixed by installing accelerate 1.15.0): "ValueError: Using a
  `device_map`, `tp_plan`, `torch.device` context manager or
  setting `torch.set_default_device(device)` requires
  `accelerate`." bitsandbytes routes need accelerate installed;
  the FP16 route does not.
- Worker shutdown crash, exact error (first aggregate run):
  "Fatal Python error: _enter_buffered_busy: could not acquire
  lock for <_io.BufferedReader name='<stdin>'> at interpreter
  shutdown". The FP16 worker had already written complete results
  but exited non-zero, and the first parent version discarded
  them on the exit code alone. The runner now exits hard after
  writing its partial file, and the parent accepts a complete
  partial while recording the worker exit code. The shipped
  results_quant.json comes from the clean rerun: no failed
  routes, no worker notes.
- Single run per config: wall times move between runs (FP16 took
  45.083 s here and 57.473 s in the diagnostic run, with identical
  tokens). Throughput ratios are one-run measurements, named as
  such, and accuracy and token counts are the stable evidence.
- Sample size: 10 questions means one question is 10 points of
  accuracy. The deltas (-0.200, -0.300) are real for this set and
  are not claimed as population estimates.
- vLLM is still not the engine (Windows node, no WSL). Serving
  with vLLM itself is v1.1 item 2 in the repo-root ROADMAP.md.

## Grade

Deep grade: 7,600 of 10,000. Deductions, named:

- Minus 650: vLLM itself is still not used; the namesake engine is
  absent across all three levels (named substitution, v1.1 item).
- Minus 350: speculative decoding, part of the original Deep
  scope, displaced to the v1.1 backlog under compression.
- Minus 350: one run per config, no repeat runs, so no variance
  bounds on throughput.
- Minus 350: 10 questions at one budget; accuracy deltas rest on
  2 and 3 questions.
- Minus 250: no AWQ or GPTQ checkpoint rows; quantization coverage
  is bitsandbytes only.
- Minus 250: the container drill (SPEC Added track 2) is still
  not done; it sits in the v1.1 backlog.
- Minus 200: throughput is in-process sequential generation, not
  through the HTTP API and not under concurrent load.

## Files

- `DEEP-SPEC.md`: scope and acceptance, written before the code.
- `quant_compare.py`, `run_deep.py`, `results_deep.json`: the
  comparison core and its deterministic fake run.
- `run_quant.py`, `results_quant.json`: the real measured run
  (per-question rows, summaries, comparison, failed routes).
- Tests: `tests/test_deep_quant.py` in the repo root tests folder.
