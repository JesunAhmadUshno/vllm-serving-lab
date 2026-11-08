# Tutorial: P03 vllm-serving-lab, course module

Module: P03-vllm-serving-lab
By Jesun Ahmad Ushno
Date: 2026-11-05
License for this tutorial: CC BY 4.0. Code in the repo: Apache-2.0.

## What you will do

You will clone the vllm-serving-lab repo, run all three deterministic
levels, run the test suite for each level, and read your own results
against the recorded v1.0 numbers. Every command below was run by
the author of this module on 2026-11-05 to verify it, on Python
3.12, and passed. The deterministic steps need no model, no GPU, no
API, and no credential: the Learn simulator, the Build fake backend,
and the Deep fake backend exist for exactly this purpose and are
never reported as model results.

The real model numbers in this module (Build study, load test, Deep
quantized comparison) were measured on the fleet GPU node, BlackBox
(NVIDIA GeForce RTX 5050 Laptop GPU), with the real model. They are
presented as recorded results to compare against, not as commands
you can rerun without that node. That boundary is stated in each
step, not hidden.

One naming fact runs through the whole module: the engine is
Transformers on PyTorch, not vLLM. The GPU node is Windows with no
WSL, and vLLM needs a Linux runtime. No command in this module runs
vLLM, and no result in this module is a vLLM result.

## Step 0: Clone and enter the repo

```text
git clone https://github.com/JesunAhmadUshno/vllm-serving-lab.git
cd vllm-serving-lab
```

Run the layout test (it replaced the scaffold smoke test on 2026-11-08):

```text
python3 -m unittest tests.test_layout -v
```

Expected: every layout check OK.

## Step 1: Level 1, Learn, the serving simulator

Read REPORT.md (Baseline: Learn) first. It explains token accounting
with thinking tokens logged apart, KV cache math (bytes per token
equals 2 times layers times KV heads times head dimension times
bytes per element), paged blocks of 16 against contiguous slabs of
256, and the token-step batching simulation.

Run the Learn simulation:

```text
python3 level-1-learn/run_learn.py
```

Run the Learn tests:

```text
python3 -m unittest discover -s tests -p "test_learn*.py" -v
```

Read your results:

- Expected totals: 1,470 prompt tokens, 1,590 thinking tokens, 915
  answer tokens, 3,975 tokens in total, KV total 521,011,200 bytes.
- Expected allocation: paged 73 waste tokens of 4,048 allocated;
  contiguous 1,657 waste tokens of 5,632 allocated.
- Expected simulation line: 79 total steps for 2,505 output tokens,
  31.708861 tokens per step. This is tokens per step, not tokens
  per second. No wall time is measured at this level.
- The runner writes level-1-learn/results_learn.json. Run it twice:
  the file is byte-identical, because the simulation has no
  randomness and no clock.
- Expected tests: 20 tests, all passing.

## Step 2: Level 2, Build, the API and the thinking-budget study

Read REPORT.md (Build) and PROPOSAL.md (Build spec). The
server is an OpenAI-compatible API on the Python standard library
(GET /health, GET /v1/models, POST /v1/chat/completions, streaming
with server-sent events, and a usage object with thinking and
answer tokens counted apart). The study scores 10 questions with
verifiable numeric answers at budgets of 32, 128, and 384 tokens,
with one extraction rule: the number after the Final answer marker
when present, else the last number, tolerance 0.01.

Run the deterministic Build run (fake backend, mechanics only):

```text
python3 level-2-build/run_build.py
```

Run the Build tests:

```text
python3 -m unittest discover -s tests -p "test_build*.py" -v
```

Read your results:

- Expected fake run: the study mechanics complete on 10 questions
  with the fake backend (1 correct at each budget in the shipped
  fake run), the health check reports the fake backend, and the
  load test counts report 8 requests, 8 succeeded, 0 failed. These
  are mechanics counts only. They are not model results.
- The runner writes level-2-build/results_build.json.
- Expected tests: 15 tests, all passing (35 in total with Learn).

Recorded real results, GPU node only, from
level-2-build/results_real.json. Do not expect the commands above
to reproduce these numbers; they need the real model on BlackBox:

- Thinking-budget study (Qwen/Qwen2.5-0.5B-Instruct, greedy): 0 of
  10 correct at budget 32 (320 completion tokens), 0 of 10 at 128
  (1,280 completion tokens), 8 of 10 at 384 (2,522 completion
  tokens, 315.25 tokens per correct, 7.196 seconds per correct).
- Stop rule: rerunning the study with a stop rule on the Final
  answer line changed no token count. The rule never fired.
- Load test: 12 of 12 succeeded, 0.477 requests per second, 61.05
  completion tokens per second, p50 8.2146 seconds, p95 9.0854
  seconds.

## Step 3: Level 3, Deep, the quantized-serving comparison

Read REPORT.md (Deep) and PROPOSAL.md (Deep spec). Deep
compares three precisions on the Build model and questions at
budget 384: FP16 with no quantization, bitsandbytes INT8
(load_in_8bit), and bitsandbytes INT4 NF4 (load_in_4bit). It also
carries the written thinking rule derived from the Build study.

Run the deterministic Deep run (fake backend, mechanics only):

```text
python3 level-3-deep/run_deep.py
```

Run the Deep tests:

```text
python3 -m unittest discover -s tests -p "test_deep*.py" -v
```

Read your results:

- Expected fake run line: fake comparison counts of 9 for FP16, 8
  for INT8, and 7 for INT4. Counts only, no wall times, labelled
  fake in the file itself.
- The runner writes level-3-deep/results_deep.json.
- Expected tests: 14 tests, all passing (49 in total).

Recorded real results, GPU node only, from
level-3-deep/results_quant.json. Each config ran in its own
subprocess on BlackBox; failed routes would be recorded in the
file, and the final run has none:

| Config | Correct | Tokens per second | GPU bytes after load |
| --- | --- | --- | --- |
| FP16 | 8 of 10 | 55.94 | 996,322,304 |
| INT8 | 6 of 10 | 10.60 | 631,487,488 (36.62 percent smaller) |
| INT4 NF4 | 5 of 10 | 32.36 | 473,807,872 (52.44 percent smaller) |

- Check the cross-level receipt: the FP16 per-question rows in
  results_quant.json match the Build 384 rows in results_real.json
  on every extracted value and every completion token count.

## Step 4: Run the full test suite

```text
python3 -m unittest discover -s tests -v
```

Expected: 49 tests in total (20 Learn, 15 Build, 14 Deep), all
passing. The scaffold smoke test in Step 0 is a script check and
is not part of the 49.

## Step 5: Check your understanding

1. Why does a 128-token budget score 0 of 10 in the Build study
   when the model clearly does work in those tokens? (Answer: this
   model completes these questions in about 190 to 330 tokens, so
   every generation at 128 is cut mid-working and the extractor
   reads a mid-reasoning number. A budget below the completion
   length buys no answers, not cheaper answers.)
2. INT4 has the lowest tokens per correct (260.40) in Deep. Why is
   that not an efficiency win? (Answer: it spends less because it
   finishes, or gives up, earlier, and it scores 5 of 10 against
   8 of 10 for FP16. Cost per correct only means something next to
   the accuracy it was paid for.)
3. INT8 saves 36.62 percent of GPU memory after load. Why is it
   still the slowest row? (Answer: at 0.5B scale on this laptop
   GPU, the bitsandbytes 8-bit overhead costs more time than the
   smaller weights save: 10.60 tokens per second against 55.94 for
   FP16. Quantization speed claims assume large models where
   weight bandwidth dominates.)

## Limits to keep with the numbers

- Learn is simulated. Build and Deep fake runs are mechanics only.
  Only the recorded GPU-node runs are model results.
- vLLM is not used at any level. The v1.1 backlog in REPORT.md (Future work)
  holds serving with vLLM itself, speculative decoding, the
  container drill, and AWQ and GPTQ rows. None of it is in v1.0.
- Ten questions at one budget is a study, not a benchmark suite.
  One run per Deep config means no variance bounds on throughput.
