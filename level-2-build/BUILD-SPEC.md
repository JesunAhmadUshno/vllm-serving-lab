# BUILD-SPEC: vllm-serving-lab (P03), Level 2 Build

Phase skill: spec-driven-development (house engineering pipeline).
Source scope: SPEC.md v1 core (serve an open model through an
OpenAI-compatible API; streaming, batching, load test) plus Added
track 1 Build (a thinking-budget study: the same hard question set at
three budget settings, scored for accuracy and cost per correct
answer). The container drill in Added track 2 is named in the README
as not done at Build on this node (no Docker on the fleet GPU node
was verified this cycle); it stays on the roadmap for Deep.

## Design

- server.py: an OpenAI-compatible HTTP API on the Python standard
  library (ThreadingHTTPServer). Endpoints: GET /health,
  GET /v1/models, POST /v1/chat/completions with stream support
  (server-sent events) and a usage object on every completion.
  Thinking and answer tokens are logged apart on every request: the
  generated text is split at the "Final answer:" marker, and the two
  parts are counted with the backend tokenizer.
- Backends sit behind one generate interface. FakeBackend is
  deterministic and exists so the API, the study scoring, and the
  load test mechanics are unit-tested in CI with no model download.
  TransformersBackend loads a real open model with PyTorch and
  Transformers on the serving node GPU. Fake backend results are
  never reported as model results.
- study.py: the thinking-budget study. One question set
  (questions.json, 10 questions with verifiable numeric answers),
  three generation budgets (32, 128, 384 new tokens). Answer
  extraction: the number after "Final answer:" when present, else
  the last number in the text; tolerance 0.01. Per budget: correct
  count, completion tokens, wall seconds, tokens per correct answer,
  seconds per correct answer.
- load_test.py: a concurrent client (thread pool, urllib) that
  records per-request latency and usage, and reports requests per
  second, completion tokens per second, and p50 and p95 latency.
- run_build.py: the deterministic local run (FakeBackend): study
  mechanics, an in-process server smoke test, and a load test whose
  counts only (no wall times) enter results_build.json, so two runs
  produce an identical file.
- run_real.py: the serving-node run. Starts the server with
  TransformersBackend on the GPU, runs the study and the load test
  against the live HTTP API, and writes results_real.json with the
  model id, the device name, and every measured value.

## Acceptance

- The API answers a chat completion with the OpenAI shape, streams
  when asked, and reports usage with thinking and answer tokens
  split.
- The study runs the same 10 questions at 3 budgets against the real
  model on the fleet GPU node and reports accuracy and cost per
  correct answer per budget.
- The load test hits the live server concurrently and reports
  throughput and latency percentiles.
- Unit tests pass in CI with no model download (FakeBackend only).
- vLLM itself is not used: the fleet GPU node is a Windows laptop
  with no WSL, and vLLM needs a Linux runtime. The API is
  OpenAI-compatible and vLLM-shaped; the engine is Transformers on
  PyTorch CUDA. This substitution is named in the README as the
  largest deduction, not hidden.
