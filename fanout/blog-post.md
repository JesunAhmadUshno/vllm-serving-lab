# Serving Is the Product, Measured Honestly: Building vllm-serving-lab in the Open

By Jesun Ahmad Ushno
Date: 2026-11-05
Status: DRAFT, written fan-out for P03 vllm-serving-lab v1.0. Not posted.

## The problem

Serving is where a language model becomes a product, and serving is
where the costs hide. Thinking tokens are generated but are not the
answer, so they cost compute and KV cache while looking like free
reasoning unless they are logged apart. A generation budget set below
the length a model needs to finish buys no answers at all, not cheaper
answers. And quantization, sold as a free memory saving, can cost
accuracy and, at small model scale, can cost speed as well.

This project, P03 in Roadmap v3, builds the serving stack in the
open, one level at a time: a deterministic simulator for token
accounting, KV cache math, and paged allocation; a real open model
served through an OpenAI-compatible API with a thinking-budget study
and a load test; then a quantized-serving comparison of FP16 against
bitsandbytes INT8 and INT4 NF4 on the same model and the same
questions. One substitution is named in every level and in this post:
the engine is Transformers on PyTorch, not vLLM. The fleet GPU node
is a Windows laptop with no WSL, and vLLM needs a Linux runtime. No
vLLM number is claimed anywhere in this repo.

Repository: https://github.com/JesunAhmadUshno/vllm-serving-lab

## Past, present, future (the dossier, condensed)

Past: the vLLM and SGLang projects themselves are the prior art for
this problem. The prior art and the problem history are recorded in
DOSSIER.md and NOVELTY.md in the repo. The novelty claim there stays
marked PROPOSED. This post makes no novelty claim beyond the measured
work below.

Present: v1.0 closes three levels on 2026-11-05. Learn taught serving
with exact simulated accounting on CPU and served no real model.
Build served Qwen/Qwen2.5-0.5B-Instruct on BlackBox (NVIDIA GeForce
RTX 5050 Laptop GPU), measured a thinking-budget study at three
budgets, measured a stop rule that never fired, and ran a concurrent
load test. Compressed Deep ran the quantized-serving comparison at
the one budget that scored in Build, and wrote the thinking rule the
Build numbers support.

Future: the rest of the v2 Deep scope is named in ROADMAP.md in the
repo as the v1.1 backlog. It includes speculative decoding, serving
with vLLM itself on a Linux node, the container drill, AWQ and GPTQ
checkpoint rows, larger question sets, and a batching and KV-cache
pressure study. None of that is done, and none of it is claimed here.

## Main results claim

At v1.0: Learn passed 20 tests on a deterministic CPU simulator with 12 labelled requests totalling 1,470 prompt tokens, 1,590 thinking tokens, and 915 answer tokens (3,975 tokens in total, 2,505 output tokens, thinking at 63.5 percent of generated tokens), KV cache at 131,072 bytes per token and 521,011,200 bytes in total for the example config, paged allocation waste of 73 tokens against 1,657 for contiguous slabs (a saving of 1,584 waste tokens), and a token-step batching simulation at 79 steps and 31.708861 tokens per step; Build passed 15 tests (35 tests in total) serving Qwen/Qwen2.5-0.5B-Instruct through an OpenAI-compatible API on an NVIDIA GeForce RTX 5050 Laptop GPU (BlackBox) with Transformers on PyTorch, scoring 0 of 10 correct at a 32-token budget (320 completion tokens), 0 of 10 at 128 tokens (1,280 completion tokens), and 8 of 10 at 384 tokens (2,522 completion tokens, 315.25 tokens per correct answer, 7.196 seconds per correct answer), with a stop rule at the Final answer line that never fired and a load test of 12 of 12 requests succeeded at 0.477 requests per second and 61.05 completion tokens per second (p50 8.2146 seconds, p95 9.0854 seconds); compressed Deep passed 14 tests (49 tests in total) on the same 10 questions at budget 384 with greedy decoding, scoring FP16 at 8 of 10 correct (2,522 completion tokens, 45.083 wall seconds, 55.94 completion tokens per second, 315.25 tokens per correct, 996,322,304 GPU bytes after load, 1,018,233,856 peak GPU bytes), bitsandbytes INT8 at 6 of 10 correct (accuracy delta -0.200 against FP16, 2,664 completion tokens, 10.60 tokens per second, 444.00 tokens per correct, 631,487,488 GPU bytes after load, 36.62 percent smaller than FP16, throughput ratio 0.189), and bitsandbytes INT4 NF4 at 5 of 10 correct (accuracy delta -0.300, 1,302 completion tokens, 32.36 tokens per second, 260.40 tokens per correct, 473,807,872 GPU bytes after load, 52.44 percent smaller than FP16, throughput ratio 0.578), with no failed routes in the final run and vLLM not used as the engine at any level.

Every number in that claim comes from a run recorded in the level
READMEs and the results files in the repo. Learn numbers are
simulated and deterministic. Build and Deep numbers are real runs on
the named GPU node. Read the limits section below before quoting
any of it.

## Results by level

### Level 1, Learn (release v0.1)

Learn serves no model. It teaches three ideas with exact arithmetic:
thinking tokens logged apart from answer tokens, KV cache size per
token, and paged allocation against a contiguous slab baseline.

| Measure | Result |
| --- | --- |
| Requests | 12 labelled requests, R01 to R12; 5 non-reasoning (thinking 0) |
| Token split | 1,470 prompt, 1,590 thinking, 915 answer; 3,975 total |
| KV bytes per token | 131,072 bytes (2 x 32 layers x 8 KV heads x 128 head_dim x 2 bytes) |
| KV total | 521,011,200 bytes |
| Paged waste | 73 waste tokens of 4,048 allocated (blocks of 16) |
| Contiguous waste | 1,657 waste tokens of 5,632 allocated (slabs of 256) |
| Batching simulation | 79 steps for 2,505 output tokens, 31.708861 tokens per step |
| Tests | 20 |

The Learn lesson for the SPEC scope: thinking is 63.5 percent of all
generated tokens on this set (1,590 of 2,505). If thinking is not
logged apart, most of the generation cost is invisible. Throughput
here is tokens per step, never tokens per second, because no wall
time is measured at this level.

### Level 2, Build (release v0.5)

Build serves the real model through an OpenAI-compatible API
(standard library ThreadingHTTPServer, streaming with server-sent
events, a usage object with thinking and answer tokens counted
apart). The study uses 10 questions with verifiable numeric answers,
greedy decoding, and one extraction rule: the number after the Final
answer marker when present, else the last number, tolerance 0.01.

| Budget (max new tokens) | Correct | Completion tokens | Tokens per correct |
| --- | --- | --- | --- |
| 32 | 0 of 10 | 320 | n/a, no correct answer |
| 128 | 0 of 10 | 1,280 | n/a, no correct answer |
| 384 | 8 of 10 | 2,522 | 315.25 (7.196 seconds per correct) |

At 32 and at 128 every generation ran into the cap mid-working, so
the extracted values are mid-reasoning numbers and score zero. This
model completes these questions in about 190 to 330 tokens, so a
budget below that length buys no answers. The two failures at 384
are genuine model errors, kept in the rows: q03 extracts 61.2 against
74.8, and q10 extracts 90 against 6.

The stop rule experiment is a measured negative. A stopping
criterion on a complete Final answer line was implemented and the
study rerun: the rows are identical in every token count and tokens
per correct stayed at 315.25. The rule never fired, because the
model ends with a boxed value or prose, not the instructed line. The
stop-rule server was also slower in the load test (0.387 requests
per second against 0.477), so the unmatched rule is pure overhead.

Load test (12 requests, concurrency 4, max 128 tokens): 12 of 12
succeeded, 0.477 requests per second, 61.05 completion tokens per
second, p50 8.2146 seconds, p95 9.0854 seconds. The server generates
one request at a time per backend call, so concurrency here measures
queueing, not continuous batching.

Tests: 15 Build tests, 35 in total with Learn.

### Level 3, Deep, compressed (release v1.0)

Deep is the quantized-serving comparison: the Build model, the same
10 questions, budget 384, greedy decoding, in-process generation.
Each precision ran in its own subprocess. Scoring is the Build
extraction rule, kept identical so Deep rows compare with Build rows.

| Config | Correct | Tokens per second | Tokens per correct | GPU bytes after load |
| --- | --- | --- | --- | --- |
| FP16, no quantization | 8 of 10 | 55.94 | 315.25 | 996,322,304 |
| INT8, bitsandbytes load_in_8bit | 6 of 10 | 10.60 | 444.00 | 631,487,488 |
| INT4, bitsandbytes load_in_4bit NF4 | 5 of 10 | 32.36 | 260.40 | 473,807,872 |

The FP16 row reproduces Build exactly: same 8 correct, same 2,522
completion tokens, and every per-question extracted value and token
count identical to the Build 384 row. INT8 saves 36.62 percent of
the GPU memory after load and loses two questions, and it is far
slower on this node (10.60 against 55.94 tokens per second, ratio
0.189): at 0.5B scale the bitsandbytes 8-bit overhead costs more
time than the smaller weights save. INT4 NF4 saves 52.44 percent of
the memory and loses three questions net, but it gains q03, the
question FP16 gets wrong, and its lower tokens per correct is not an
efficiency win: it spends less because it finishes, or gives up,
earlier, and scores less. No row dominates. Grade for Deep: 7,600
of 10,000, with named deductions in level-3-deep/README.md; the
largest is that vLLM itself is still not the engine.

The thinking rule the Build numbers support, in short: a budget pays
only above the model's natural completion length for the task class;
a budget is a cap, not a target; more thinking does not fix a wrong
method; cost per correct answer is the unit to watch; and stop rules
only save money when they match the model's real output format. The
full rule, with its five points, is in level-3-deep/README.md.

Tests: 14 Deep tests, 49 in total with Learn and Build.

## Honest limits

- vLLM is not the engine at any level. The engine is Transformers on
  PyTorch behind an OpenAI-compatible API, because the GPU node is
  Windows with no WSL. Serving with vLLM itself is v1.1 item 2.
- Learn numbers are simulated on a labelled teaching set. They are
  exact for their stated rules and are not traces from a real model.
- Build and Deep use one small model (Qwen/Qwen2.5-0.5B-Instruct,
  494M parameters) and 10 questions. One question is 10 points of
  accuracy, so the Deep deltas (-0.200, -0.300) are real for this
  set and are not population estimates.
- Deep ran one run per config, so throughput ratios are one-run
  measurements with no variance bounds. Accuracy and token counts
  are the stable evidence: the FP16 tokens reproduce Build exactly.
- Quantization coverage is bitsandbytes load-time quantization only
  (INT8 and INT4 NF4). No AWQ or GPTQ checkpoint rows exist in v1.0.
- The Build server generates one request at a time. There is no
  continuous batching, so load numbers are a floor, not a ceiling.
- Compressed Deep is not the full v2 Deep scope. Speculative
  decoding and the container drill are in the v1.1 backlog, not done.

## Where to go next

- Tutorial: fanout/tutorial.md in the repo, a step-by-step module
  that runs all three deterministic levels and the full test suite,
  and names the GPU-node runs honestly as GPU-node runs.
- Releases: v0.1 (Learn), v0.5 (Build), v1.0 (Learn, Build, Deep) on
  the GitHub repo linked above.
- Book chapter: "Serving Is The Product" per SPEC.md, held as a
  chapter draft until the finished project set is assembled.
- License: code Apache-2.0, tutorials CC BY 4.0.
