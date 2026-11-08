# Report: vllm-serving-lab (P03)

Status: FUNCTIONAL, not pre-registered (2026-11-08). A real open model
(Qwen/Qwen2.5-0.5B-Instruct) was served end to end through the repo's
OpenAI-compatible API on one laptop GPU, and the recorded runs are
committed as result files. The engine is Transformers on PyTorch, not
vLLM. No hypothesis was locked before the runs, so nothing here is a
confirmed finding (PROPOSAL.md).

Moved-file map. Text carried over from the old files still names
them. SPEC.md, NOVELTY.md, BUILD-SPEC.md and DEEP-SPEC.md are now
sections of PROPOSAL.md. The level READMEs (level-1-learn/README.md,
level-2-build/README.md, level-3-deep/README.md), DOSSIER.md and
docs/research-review.md are now sections of REPORT.md, and ROADMAP.md
(its "ROADMAP item N") is REPORT.md, Future work. "The README" in that
text means the level README, now the matching REPORT.md section.

## Results at a glance

Every number is read from a committed result file, named in the last
column. GPU runs: BlackBox, NVIDIA GeForce RTX 5050 Laptop GPU, greedy
decoding, the same 10 questions (level-2-build/questions.json), one run
per configuration. Intervals are Wilson 95 percent over the 10
questions; p values are exact two-sided McNemar tests on the paired
per-question rows. Both are computed by `python3
level-3-deep/significance.py`, which also writes
level-3-deep/results_significance.json (checked unchanged in CI).

| Result | Value | Evidence |
| --- | --- | --- |
| Budget study, correct at 32 / 128 / 384 max new tokens | 0 / 0 / 8 of 10 (Wilson 0.000 to 0.278, 0.000 to 0.278, 0.490 to 0.943) | level-2-build/results_real.json, budgets and rows |
| Budget 128 vs 384, paired | 0 vs 8 discordant questions, p = 0.0078 | level-3-deep/results_significance.json |
| Cost per correct answer at 384 | 315.25 completion tokens, 7.196 s over HTTP | level-2-build/results_real.json |
| Stop rule at the "Final answer:" line | never fired; token counts identical | level-2-build/results_real_stop.json |
| Load test, 12 requests, concurrency 4, max 128 tokens | 12 of 12, 0.477 req/s, 61.05 completion tok/s, p50 8.2146 s, p95 9.0854 s | level-2-build/results_real.json, load_test |
| FP16 at budget 384 | 8 of 10, 55.94 tok/s, 996,322,304 GPU bytes after load | level-3-deep/results_quant.json |
| INT8 bitsandbytes | 6 of 10 (Wilson 0.313 to 0.832), 10.60 tok/s, 36.62 percent less memory | same |
| INT4 bitsandbytes NF4 | 5 of 10 (Wilson 0.237 to 0.763), 32.36 tok/s, 52.44 percent less memory | same |
| FP16 vs INT8, FP16 vs INT4, paired | 2 vs 0 (p = 0.5); 4 vs 1 (p = 0.375) | level-3-deep/results_significance.json |
| Simulated KV and paging (CPU, example config, not the served model) | 131,072 bytes per token; paged waste 73 vs contiguous 1,657 tokens | level-1-learn/results_learn.json |

Not in the repo as evidence: per-request load-test latencies (only p50
and p95 were saved), repeat runs for throughput variance, and any vLLM
number. The hardware and library versions of the GPU runs are recorded
inside results_quant.json; the Build run records only model and device.

## Hypotheses

None was written down before the runs (PROPOSAL.md), so none can be
scored as confirmed or refuted. What the runs show, stated as
observations:

| Observation | Status | Why it is not a finding |
| --- | --- | --- |
| A budget below the natural answer length buys no correct answers (0 of 10 at 128, 8 of 10 at 384) | Observed, p = 0.0078 | Post hoc; one model, 10 questions, one run |
| INT8 and INT4 NF4 lose accuracy against FP16 | Not separable (p = 0.5, 0.375) | n = 10 cannot detect a 2 or 3 question difference |
| INT8 is slower than FP16 at this scale (10.60 vs 55.94 tok/s) | Observed once | One run per config; FP16 wall time moved 45.083 s to 57.473 s between runs |
| A stop rule at the instructed answer line saves tokens | Missed: it never fired | The model ignores the instructed format |
| The serving stack is vLLM | Not tested | vLLM never ran (Windows node, no WSL) |

## Field profile check (systems and infrastructure, reliability)

The portfolio assigns this repo the reliability profile. None of its
items is met yet: no stated availability, recovery-time or data-loss
targets; no timed drill of a planned failure; no monitoring that would
catch one; no runbook or postmortem. The work so far is a serving and
cost study, closer to the performance profile, and it does not meet
that profile either (one run per configuration, no warm-up runs, no
co-tenancy record). Evaluated waits on a locked proposal under one
profile.

## Background

Written under the documentation and publishing standard (past, present, future). Updated at ship time.

### Past

Prior art for this project is named in NOVELTY.md (the vLLM and SGLang projects themselves). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

### Present

Current demand evidence: hiring value 5 of 5 (v2 build rank 3) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

Build update, 2026-11-05: Level 2 Build shipped. A real open model (Qwen/Qwen2.5-0.5B-Instruct, 494M parameters) is served through an OpenAI-compatible API on the fleet GPU node (BlackBox, RTX 5050) with Transformers on PyTorch as the engine; vLLM itself needs a Linux runtime the Windows node does not have, and that substitution is named in the level README. The thinking-budget study measured the same 10 hard questions at budgets 32, 128, and 384 tokens: 0, 0, and 8 correct, with every generation capped below 384 and a cost of 315.25 tokens per correct answer at 384. A stop rule at the instructed Final answer line was implemented and measured: it never fired, because the model ends answers in its own format. The load test served 12 of 12 concurrent requests at 0.477 requests per second.

Deep update, 2026-11-05: Level 3 Deep shipped (compressed Deep), v1.0. The quantized-serving comparison ran on BlackBox (RTX 5050) with the Build model, questions, scoring rule, and the 384-token budget: FP16 scored 8 of 10 at 55.94 tokens per second and 996,322,304 GPU bytes after load; bitsandbytes INT8 scored 6 of 10 at 10.60 tokens per second and 631,487,488 bytes (36.62 percent smaller); bitsandbytes INT4 NF4 scored 5 of 10 at 32.36 tokens per second and 473,807,872 bytes (52.44 percent smaller). The FP16 row reproduces the Build 384 row exactly, token for token. The Deep README also carries the written thinking rule derived from the Build study (a budget pays only above the model's natural completion length; cost per correct answer is the unit to watch). The largest named gap is that vLLM itself is still not the engine. Release v1.0.

### Future

The rest of the v2 Deep scope (speculative decoding, serving with vLLM itself on a Linux node, the container drill, AWQ and GPTQ checkpoint rows) is the named v1.1 backlog in the repo-root ROADMAP.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.

## Baseline: Learn (simulated serving on CPU)

Paths in this section are relative to level-1-learn/.

Status: Learn DONE, 2026-11-05. Simulated on CPU. No real model was served.

Honest limit, first: BlackBox, the GPU node for this project, was unreachable
in this cycle (NEEDS-BLACKBOX-ON for real GPU serving). So Learn teaches
serving with a deterministic simulator and exact token accounting on CPU.
Real vLLM serving of one open reasoning model is Build scope. No GPU number
and no real vLLM number is claimed anywhere in this level. Every number
below is simulated, deterministic, with its method stated.

### What this level teaches

The SPEC Learn scope is: serve one open reasoning model, and log thinking
tokens apart from answer tokens on every request. Three ideas carry that:

1. Token accounting. A request is prompt tokens plus thinking tokens plus
   answer tokens. Thinking tokens are generated but are not the answer, so
   they cost compute and KV cache and must be logged apart. Non-reasoning
   requests have thinking = 0.
2. KV cache size. Each token in the sequence needs key and value storage
   in every layer. Bytes per token = 2 (K and V) * layers * kv_heads *
   head_dim * bytes per element.
3. Paged allocation and batching. PagedAttention style blocks of 16 tokens
   waste far less than one contiguous slab per request, and a serving
   engine shares a fixed per-step token budget across the active batch.

### Files

- `serving_sim.py`: request record, log line, KV math, block allocation,
  and the token-step batching simulation. Stdlib only.
- `requests.json`: 12 labelled requests, R01 to R12, with varied prompt,
  thinking, and answer counts. Five are non-reasoning (thinking = 0).
- `run_learn.py`: runs the accounting and simulation, prints the full
  table, and writes `results_learn.json`. Run it twice: the file is
  byte-identical, because the simulation has no randomness and no clock.
- `results_learn.json`: the measured (simulated) results below.

### Method

- Example model config, named `example-32L-8KV-128D-fp16`: 32 layers,
  8 KV heads, head_dim 128, 2 bytes per element (fp16).
- KV bytes per token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes (128 KiB).
  A request KV size is its total tokens times 131,072.
- Paged allocation: blocks of 16 tokens, allocated tokens are total
  rounded up to a whole block, waste is allocated minus total.
- Contiguous baseline slab rule: every request is rounded up to the next
  whole multiple of 256 tokens, however small it is.
- Batching simulation: all 12 requests are active from step 1. Each step
  has a budget of 32 generation tokens. Inside a step, repeated round
  robin passes in request order give 1 token to each active request per
  pass until the budget is used or all active requests finish. Only
  output tokens (thinking plus answer) are generated. Prompt tokens are
  prefill: they count in KV and block accounting, not in steps.
- Throughput is reported in tokens per step, never tokens per second.
  No wall time is measured or claimed.

### Results (simulated, deterministic)

Per-request table, from `run_learn.py` run on 2026-11-05:

| id | prompt | thinking | answer | output | total | kv_bytes | paged_blocks | paged_waste | contig_waste | steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | 120 | 200 | 80 | 280 | 400 | 52428800 | 25 | 0 | 112 | 58 |
| R02 | 60 | 0 | 45 | 45 | 105 | 13762560 | 7 | 7 | 151 | 15 |
| R03 | 200 | 350 | 120 | 470 | 670 | 87818240 | 42 | 2 | 98 | 72 |
| R04 | 45 | 80 | 30 | 110 | 155 | 20316160 | 10 | 5 | 101 | 32 |
| R05 | 90 | 0 | 60 | 60 | 150 | 19660800 | 10 | 10 | 106 | 20 |
| R06 | 300 | 500 | 150 | 650 | 950 | 124518400 | 60 | 10 | 74 | 79 |
| R07 | 30 | 0 | 25 | 25 | 55 | 7208960 | 4 | 9 | 201 | 9 |
| R08 | 150 | 120 | 90 | 210 | 360 | 47185920 | 23 | 8 | 152 | 53 |
| R09 | 200 | 0 | 100 | 100 | 300 | 39321600 | 19 | 4 | 212 | 32 |
| R10 | 55 | 250 | 70 | 320 | 375 | 49152000 | 24 | 9 | 137 | 66 |
| R11 | 180 | 90 | 110 | 200 | 380 | 49807360 | 24 | 4 | 132 | 53 |
| R12 | 40 | 0 | 35 | 35 | 75 | 9830400 | 5 | 5 | 181 | 17 |

Totals across the 12 requests:

- Prompt tokens 1,470. Thinking tokens 1,590. Answer tokens 915.
  Output tokens 2,505. Total tokens 3,975.
- Thinking is 63.5 percent of all generated tokens (1,590 of 2,505).
  That is the Learn lesson for the SPEC scope: if thinking is not logged
  apart, most of the generation cost is invisible.
- KV bytes total 521,011,200 for the example config.
- Paged allocation: 4,048 tokens allocated, 73 waste tokens.
- Contiguous slab allocation: 5,632 tokens allocated, 1,657 waste tokens.
  Paged blocks cut waste by 1,584 tokens on this set (73 vs 1,657).
- Simulation: 79 total steps for 2,505 output tokens, simulated
  throughput 31.708861 tokens per step against a budget of 32. The gap
  below 32 is the last step, where only R06 remains and finishes early.

Every request also prints a log line that separates the split, for
example: `request=R01 prompt=120 thinking=200 answer=80 output=280 total=400 reasoning=true`

### Tests

`tests/test_learn_serving.py`: 20 unit tests with hand-computed values
for KV bytes per token and per request, thinking and answer split sums,
paged vs contiguous waste, and simulation step totals for small cases
that can be checked by hand. `tests/test_smoke.py` still passes (replaced by
`tests/test_layout.py` on 2026-11-08).

### Failure modes and limits

- No real model was served. Token counts in `requests.json` are a
  labelled teaching set, not traces from a real reasoning model.
- The simulator has no prefill cost, no memory bandwidth, no scheduling
  delay, no arrival times, and no sampling. Real vLLM numbers will
  differ and belong to Build, on BlackBox, with a real model.
- KV math assumes every token is kept for the whole request and uses
  the example config only. Grouped-query attention, KV quantization,
  and prefix caching change real sizes.
- The round robin step rule is a teaching model of continuous batching,
  not the vLLM scheduler. It is exact for its stated rule, which is why
  the tests can hold hand-computed step counts.
- Throughput in tokens per step cannot be converted to tokens per
  second without a real step time, which this level does not measure.

### Next

Build: serve one open reasoning model with real vLLM on BlackBox,
log thinking apart from answer on real requests, then run the
thinking-budget study from SPEC.md.

## Build: a served model, the thinking-budget study, a load test

Paths in this section are relative to level-2-build/.

Status: Build DONE, 2026-11-05. A real open model served through an
OpenAI-compatible API on the fleet GPU node, a thinking-budget study
with measured accuracy and cost, and a concurrent load test.

### What this level builds

The SPEC Build scope has two parts. First, the v1 core: serve an open
model through an OpenAI-compatible API, with streaming, batching, and
a load test. Second, Added track 1: a thinking-budget study, the same
hard question set at three budget settings, scored for accuracy and
cost per correct answer.

- `server.py`: the API. Standard library ThreadingHTTPServer, so the
  serving surface has no framework between the reader and the code.
  Endpoints: `GET /health`, `GET /v1/models`,
  `POST /v1/chat/completions`. Streaming answers with server-sent
  events when `stream` is true. Every completion carries a usage
  object with prompt, completion, and total tokens, plus thinking
  tokens and answer tokens counted apart (the text is split at the
  "Final answer:" marker, continuing the Learn level rule that
  thinking is logged apart on every request).
- Backends: `TransformersBackend` serves the real model (PyTorch and
  Transformers on the GPU). `FakeBackend` is deterministic, exists
  for CI, and is never reported as a model.
- `study.py` and `questions.json`: the budget study. Ten questions
  with verifiable numeric answers (multi-step arithmetic, percentages,
  unit reasoning, a sequence, an age problem). Extraction rule: the
  number after "Final answer:" when present, else the last number in
  the text; tolerance 0.01.
- `load_test.py`: concurrent client, thread pool over urllib,
  reporting requests per second, completion tokens per second, and
  p50 and p95 latency.
- `run_build.py`: the deterministic local run (fake backend,
  mechanics only). `run_real.py`: the serving-node run, study and
  load test over live HTTP against the real model.

### The engine substitution, named first

vLLM itself does not run in this build. The fleet GPU node is a
Windows laptop with no WSL installed, and vLLM needs a Linux runtime.
The serving engine is Transformers on PyTorch CUDA behind an
OpenAI-compatible, vLLM-shaped API. Serving with vLLM on a Linux node
is the first Deep item. No vLLM number is claimed anywhere in this
level.

### Measured results: thinking-budget study (real model)

Model: Qwen/Qwen2.5-0.5B-Instruct (494M parameters, public, downloaded
unauthenticated from Hugging Face, no token used). Node: BlackBox,
the Windows laptop, NVIDIA GeForce RTX 5050 Laptop GPU. Method:
greedy decoding, the same 10 questions at each budget, answers scored
by the extraction rule above. Full per-question rows are in
`results_real.json`.

| Budget (max new tokens) | Correct | Accuracy | Completion tokens | Tokens per correct | Seconds per correct |
| --- | --- | --- | --- | --- | --- |
| 32 | 0 of 10 | 0.000 | 320 | n/a, no correct answer | n/a |
| 128 | 0 of 10 | 0.000 | 1,280 | n/a, no correct answer | n/a |
| 384 | 8 of 10 | 0.800 | 2,522 | 315.25 | 7.196 |

What the table teaches, from the per-question rows, not the summary:
this model's natural answer length on multi-step questions is about
190 to 330 tokens. At 32 and at 128 every single generation ran into
the cap mid-working (320 and 1,280 tokens are exactly 10 times the
budget), so the extracted values are mid-reasoning numbers (a
subtotal, a rate, a partial sum) and score zero. A budget below the
model's completion length does not buy cheaper answers; it buys no
answers. At 384, eight answers complete and score, and the cost of a
correct answer is 315.25 tokens and about 7.2 seconds. The two
failures at 384 are genuine model errors, kept in the rows: q03
(applies the discount and tax wrongly, extracts 61.2 against 74.8)
and q10 (counts to the cap, extracts 90 against 6).

One more measured finding: the prompt instructs the model to end
with a "Final answer:" line, and at 384 the model often ignores the
format and ends with a boxed value or plain prose instead. The
extractor's fallback (last number in the text) is what scores those
rows. Format compliance is a serving property, and here it is
measured, not assumed.

### The stop rule experiment (measured, negative)

A serving stack normally stops generation at the answer line. This
build implemented that rule (a stopping criterion on a complete
"Final answer: <number>" line in `TransformersBackend`) and reran
the whole study with it on (`results_real_stop.json`). Result: the
study rows are identical in every token count. The rule never fired,
because the model does not emit that line (see the format finding
above). Tokens per correct answer stayed 315.25. The lesson is
recorded plainly: a stop rule only saves money when it matches the
model's actual output format; an unmatched rule is dead code with a
per-step decode cost (the stop run's load test was slower, below).

### Measured results: load test (real model)

| Requests | Succeeded | Requests per second | Completion tokens per second | p50 latency (s) | p95 latency (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 12 | 0.477 | 61.05 | 8.2146 | 9.0854 | 25.161 |

The stop-rule server, same load test: 12 of 12 succeeded, 0.387
requests per second, 49.54 completion tokens per second, p50 10.2331
seconds, p95 10.5925 seconds. Same generated tokens, slower wall
time: the unmatched stop rule's per-step decode is pure overhead
here.

Method: 12 requests (the first 6 study questions, twice), concurrency
4, max 128 new tokens each, over local HTTP on the serving node.
Latency is end to end per request, including queueing behind other
requests in the same server process. The server generates one request
at a time per backend call, so concurrency here measures queueing,
not continuous batching; a batching engine is Deep scope, and that
limit is stated, not hidden.

### Failure modes

- Budget truncation: a hard cap cuts the reasoning before the answer
  marker. Mitigation used by real serving stacks: budget by task
  class, and a finish reason that names truncation.
- Extraction failure: a correct value in the working but no final
  answer line scores as wrong. The study counts it wrong on purpose;
  a production scorer needs the same honesty.
- Single-request generation: no continuous batching, so concurrent
  load queues. Throughput numbers here are a floor, not a ceiling.
- Small model ceiling: a 0.5B instruct model is not a dedicated
  reasoning model. Its accuracy is reported as measured; the study
  method, not the model score, is the Build claim.
- The container drill (SPEC Added track 2) is not done: no Docker
  runtime was verified on the fleet GPU node this cycle. It stays
  named here and moves to Deep planning.

### Files

- `BUILD-SPEC.md`: scope and design, written before the code.
- `server.py`, `study.py`, `load_test.py`: the system.
- `questions.json`: the 10-question set with expected answers.
- `run_build.py` and `results_build.json`: deterministic fake run.
- `run_real.py`, `results_real.json`, and `results_real_stop.json`:
  the real measured runs (study and load test, without and with the
  stop rule).
- Tests: `tests/test_build_serving.py` in the repo root tests folder
  (API shape, streaming reassembly, usage math, extraction, study
  scoring, load test aggregation), plus the Learn and smoke suites.

## Deep: quantized-serving comparison and the thinking rule

Paths in this section are relative to level-3-deep/.

Status: Deep DONE, 2026-11-05 (compressed Deep, December
Compression Protocol). A quantized-serving comparison with measured
values only, plus the written thinking rule from the Build study.
The rest of the v2 Deep scope is the v1.1 backlog in the repo-root
ROADMAP.md.

### What this level builds

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

### Method (real run)

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

### Measured results: quantized-serving comparison

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

### The thinking rule (from the Build study)

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

### Failure modes

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


### Known gaps named at ship

- vLLM itself is still not used; the namesake engine is absent across
  all three levels (named substitution, v1.1 item).
- Speculative decoding, part of the original Deep scope, was displaced
  to the v1.1 backlog under compression.
- One run per config, no repeat runs, so no variance bounds on
  throughput.
- 10 questions at one budget; accuracy deltas rest on 2 and 3
  questions.
- No AWQ or GPTQ checkpoint rows; quantization coverage is
  bitsandbytes only.
- The container drill (SPEC Added track 2) is still not done; it sits
  in the v1.1 backlog.
- Throughput is in-process sequential generation, not through the HTTP
  API and not under concurrent load.

### Files

- `DEEP-SPEC.md`: scope and acceptance, written before the code.
- `quant_compare.py`, `run_deep.py`, `results_deep.json`: the
  comparison core and its deterministic fake run.
- `run_quant.py`, `results_quant.json`: the real measured run
  (per-question rows, summaries, comparison, failed routes).
- Tests: `tests/test_deep_quant.py` in the repo root tests folder.

## Review (research review, 2026-11-07)

Reviewed 2026-11-07 as a piece of research. Each criterion gets what
the repo has, what that change fixed, and, for every gap still open,
the exact compute or data that would close it. This is a read-through,
not the re-derivation the Review stage of the standard requires. Gaps that could be
fixed on a CPU in minutes, with no GPU, paid API or long run, are
fixed in this change; nothing below is left open for lack of effort.

Sources: level-1-learn, level-2-build and level-3-deep (code, READMEs,
result JSON), ROADMAP.md, SPEC.md, and the new
`level-3-deep/significance.py`.

### Summary

| Criterion | One-line reason |
| --- | --- |
| Research question | Clear practical questions (does a thinking budget pay; what does quantization cost), but the repo's namesake question (vLLM serving) is not yet asked of vLLM. |
| Hypotheses | Expectations are stated informally in the specs; no pre-registered, falsifiable hypotheses with thresholds. |
| Method | Greedy decoding, fixed question set, identical scoring and memory counters across configs, per-question rows saved. Engine is Transformers, not vLLM. |
| Baselines | FP16 is the baseline for quantization; budget 384 anchors the budget study; contiguous slabs are the baseline for paging. No engine baseline. |
| Statistics and uncertainty | Wilson intervals and exact paired McNemar tests now exist in code, CI and on the dashboard. Throughput still has no variance. |
| Reproducibility | Deterministic CPU parts reproduce in CI; GPU runs record versions and device; FP16 Deep rows reproduce Build rows exactly. Raw load-test latencies were not saved. |
| Threats to validity | Named in the level READMEs and now gathered on the dashboard and below. |
| Related work | DOSSIER.md gives background; no systematic comparison with published serving or quantization results. |
| Limitations | Deductions are named plainly in level-3-deep/README.md and ROADMAP.md. |

### Criterion detail

#### Research question

Two answerable questions are asked and answered with measured data on
one consumer GPU: (1) how accuracy and cost per correct answer move
with a generation budget, and (2) what bitsandbytes INT8 and INT4 NF4
cost against FP16 in accuracy, memory and throughput. The repo name and
SPEC promise a vLLM/SGLang serving study; no vLLM number exists.

Open, needs compute: a Linux GPU node (any CUDA GPU with at least 4 GB,
WSL2 or native Linux) with vLLM installed, rerunning
`level-2-build/run_real.py` and `level-3-deep/run_quant.py` against a
vLLM OpenAI server for the same model and questions. Roughly one GPU
hour. ROADMAP item 2.

#### Hypotheses

The specs say what was expected, but no hypothesis was registered with
a decision threshold before the runs. This cannot be fixed after the
fact without HARKing; the honest repair is for the next experiment.

Open, needs a protocol, not compute: write the v1.1 hypotheses (for
example "vLLM continuous batching raises load-test completion tokens
per second at concurrency 4 by at least 2x over the sequential server")
with thresholds into a dated file before running them.

#### Method

Strong points: greedy decoding makes runs deterministic; the same
scoring rule and memory counters are used for every config; every
per-question row is persisted. Weak points: generation is sequential
and in-process for the quantization study; the load-test server decodes
one request at a time; the thinking vs answer split on real runs relies
on a "Final answer:" marker the model mostly ignores, so the real split
is effectively unmeasured.

Open, needs compute: a batched engine (vLLM, or Transformers with
static batching) for the load test, about 30 GPU minutes; and a prompt
or chat template that makes the model emit a parseable reasoning
delimiter (or a reasoning model with native thinking tags), rerun on
the 10 questions, about 15 GPU minutes.

#### Baselines

FP16 is a correct baseline for quantization and contiguous allocation
is the right baseline for paged allocation. Missing: an engine baseline
(vLLM vs Transformers) and pre-quantized checkpoints (AWQ, GPTQ) to
separate load-time from checkpoint quantization.

Open, needs compute: the vLLM run above, and AWQ and GPTQ checkpoints
of Qwen2.5-0.5B-Instruct run through `run_quant.py`, about 30 GPU
minutes. ROADMAP item 4.

#### Statistics and uncertainty

Before this change the repo reported accuracy deltas (-0.2, -0.3) with
no interval or test. Fixed here:

- `level-3-deep/significance.py` computes Wilson 95% intervals for
  every accuracy and exact two-sided McNemar tests on the paired
  per-question rows. Unit tests (`tests/test_significance.py`, run in
  CI) check hand-worked values.
- Result: FP16 vs INT8 has 2 vs 0 discordant questions, p = 0.5; FP16
  vs INT4 NF4 has 4 vs 1, p = 0.375. Neither is separable at n = 10.
  The budget effect is: budget 128 vs 384 has 0 vs 8, p = 0.0078.
- The dashboard computes the same intervals and tests in the browser,
  and its build fails if they differ from the hand-worked values.
- A counterfactual budget curve fills the unmeasured 128 to 384 gap
  from the recorded 384 rows without a GPU: exact on tokens, a lower
  bound on correct answers, and labelled that way. It reproduces all
  three recorded budget points.

Still open, needs compute or data:

- Throughput variance: at least 5 repeat runs per config of
  `run_quant.py` (about 30 GPU minutes in total at the recorded wall
  times), to give a spread on tokens per second. The Deep README
  already notes FP16 moving from 45.083 s to 57.473 s between runs.
- Power: a 100-question set with difficulty tiers (ROADMAP item 5).
  With 10 questions one question is 10 points of accuracy; detecting a
  10-point paired difference needs on the order of 100 or more
  questions. About 1 to 2 GPU hours for all configs.
- A denser budget sweep (for example 192, 256, 320) to replace the
  counterfactual lower bound with measured points, about 10 GPU minutes.

#### Reproducibility

The Learn simulator and the fake Build and Deep runners are
deterministic and rerun in CI on Python 3.11 and 3.12. GPU results
record torch, transformers, bitsandbytes and Python versions and the
device. The FP16 Deep rows match the Build 384 rows question by
question, now also asserted at site build time. Fixed here: the
browser ports of `serving_sim.py` are checked against
`results_learn.json` on every build, so the dashboard simulator is
verifiably the same model as the Python one.

Open, needs compute: the load test persisted only p50 and p95, so the
latency distribution cannot be re-analysed. Rerunning
`level-2-build/load_test.py` with per-request latencies saved takes
about 1 GPU minute per run. A container image for the serving stack
(ROADMAP item 3) needs a Docker runtime on the GPU node.

#### Threats to validity

Internal: single run per config; scoring by extraction heuristic with
no human-checked error rate; truncation and capability errors are
separated by reading tokens against the cap. External: one 0.5B model,
one laptop GPU, 10 arithmetic word problems; bitsandbytes overhead
dominates at this scale, so the INT8 slowdown is not a general claim.
Construct: the Learn KV config (32 layers, 8 KV heads, head_dim 128) is
an example, not the served model, and requests.json is a hand-labelled
teaching set, so simulated numbers teach mechanics and are not
predictions for this GPU.

Not fixed here, deliberately: a second KV preset for the served model.
Its config values (layers, KV heads, head_dim) are not in the repo, and
adding them without a cited data file would put an unsourced number on
the site. Needs: the model's public `config.json` committed as a data
file with its source URL, then one preset added to the calculator.

Open, needs data: a human check of the extraction rule on the 60
recorded study rows plus the 30 quantization rows (about one hour of
reading, no compute), giving an extraction error rate. ROADMAP item 7.

#### Related work

DOSSIER.md covers the problem background and NOVELTY.md is a
self-labelled unverified draft. There is no table placing these
numbers next to published vLLM, PagedAttention or bitsandbytes
results.

Open, needs reading, not compute: a related-work section citing the
PagedAttention paper (Kwon et al., SOSP 2023), the bitsandbytes LLM.int8
and QLoRA/NF4 papers, and published budget-forcing or test-time compute
studies, with a sentence on how each result here agrees or differs.

#### Limitations

The level READMEs name every deduction plainly: no vLLM, no speculative
decoding, one run per config, n = 10, bitsandbytes only, no container
drill, sequential throughput. The dashboard repeats them in its limits
window.

### What this change added

- `level-3-deep/significance.py` and `tests/test_significance.py`
  (Wilson, exact McNemar, recorded-row checks; in CI).
- `site/`: the live dashboard, with `src/sim.check.mjs` asserting the
  browser computations against the repo's result files on every build.
- `.github/workflows/pages.yml` to publish it.

## Future work

### v1.1 backlog (named remainder of the v2 Deep scope, not done)

Written at v1.0 ship time, 2026-11-05. Under the December
Compression Protocol, Deep shipped the single hardest capability in
its scope: the quantized-serving comparison (FP16 against INT8 and
INT4, measured on the fleet GPU node). Everything below is the rest
of the v2 Deep scope, named here as the v1.1 backlog. Nothing below
is built, and nothing below is claimed.

1. Speculative decoding: one measured run of a draft-model or
   prompt-lookup speculative setup against the plain FP16 run, same
   question set, reporting acceptance rate, latency change, and
   whether accuracy moves. This was in the original Deep scope
   (SPEC.md, Added track 1) and was displaced by the quantization
   comparison under compression.
2. Serving with vLLM itself: the Build and Deep levels run
   Transformers on PyTorch behind an OpenAI-compatible API because
   the fleet GPU node is Windows with no WSL and vLLM needs a Linux
   runtime. On a Linux node (or a free Linux GPU tier), serve the
   same model with vLLM, rerun the thinking-budget study and the
   quantization comparison, and publish the side-by-side engine
   table. No vLLM number exists in v1.0.
3. The container drill (SPEC.md, Added track 2, register item 621):
   the serving stack as a small, layered, reproducible Docker
   image, scanned, with the build written up. Not done in v1.0: no
   Docker runtime was verified on the fleet GPU node.
4. AWQ and GPTQ checkpoints: v1.0 quantization is bitsandbytes
   on-the-fly (INT8 and INT4 NF4). Pre-quantized AWQ and GPTQ
   checkpoints of the same model family, run and measured the same
   way, would separate checkpoint quantization from load-time
   quantization.
5. Larger and harder question sets: 10 questions at one budget is a
   study, not a benchmark suite. A 100-question set with difficulty
   tiers would tighten every number in the Deep table.
6. Continuous batching and KV-cache pressure: the Build server
   generates one request at a time. A batching engine comparison
   under load, with KV-cache limits hit on purpose, is open.
7. Human calibration of the judge-free scoring: Deep scoring is the
   Build extraction rule. A small human-checked sample would bound
   the extraction error rate.

### Order

Items 2 and 1 first (they close the two named substitutions in the
v1.0 record), then 3, then the rest as capacity allows in the 2027
improvement phase.

## Conclusion

A served 0.5B model on one laptop GPU shows two things clearly and one
thing not at all. Clearly: a generation budget below the model's
natural answer length buys no correct answers (0 of 10 at 32 and 128,
8 of 10 at 384, exact McNemar p = 0.0078), and load-time quantization
trades memory for accuracy and, at this scale, for speed (INT8 37
percent smaller and 5.3x slower; INT4 NF4 52 percent smaller). Not at
all: whether INT8 or INT4 lose accuracy beyond chance (p = 0.5 and
0.375 at n = 10), and anything about vLLM, which never ran. No
hypothesis was locked before the runs, so these are observations, not
confirmed findings.
