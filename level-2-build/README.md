# Level 2, Build: vllm-serving-lab

Status: Build DONE, 2026-11-05. A real open model served through an
OpenAI-compatible API on the fleet GPU node, a thinking-budget study
with measured accuracy and cost, and a concurrent load test.

## What this level builds

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

## The engine substitution, named first

vLLM itself does not run in this build. The fleet GPU node is a
Windows laptop with no WSL installed, and vLLM needs a Linux runtime.
The serving engine is Transformers on PyTorch CUDA behind an
OpenAI-compatible, vLLM-shaped API. Serving with vLLM on a Linux node
is the first Deep item. No vLLM number is claimed anywhere in this
level.

## Measured results: thinking-budget study (real model)

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

## The stop rule experiment (measured, negative)

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

## Measured results: load test (real model)

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

## Failure modes

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

## Files

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
