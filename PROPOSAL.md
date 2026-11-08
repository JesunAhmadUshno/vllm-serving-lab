# Proposal: vllm-serving-lab (P03)

Status: NOT LOCKED. This repo is not pre-registered. The first commit
(018bd7e, 2026-11-04) holds a scope spec written before any code, but it
has no hypotheses, pass lines or analysis plan, so it is a plan of work,
not a locked proposal. The Build and Deep specs below were committed in
the same commits as their code and results (44e0b67 and 1dae29d), so
git cannot prove they came first. Nothing here is tagged and nothing
here is a lock. Under the portfolio standard v2 the repo is capped at
Functional until a confirmatory cycle runs under a newly locked
proposal (METHOD section 7). Writing that proposal is open; its choices
(engine, node, hypotheses, pass lines) are the Owner's.

This file collects the old scope documents unchanged in substance:
SPEC.md, NOVELTY.md, level-2-build/BUILD-SPEC.md and
level-3-deep/DEEP-SPEC.md.

Moved-file map. Text carried over from the old files still names
them. SPEC.md, NOVELTY.md, BUILD-SPEC.md and DEEP-SPEC.md are now
sections of PROPOSAL.md. The level READMEs (level-1-learn/README.md,
level-2-build/README.md, level-3-deep/README.md), DOSSIER.md and
docs/research-review.md are now sections of REPORT.md, and ROADMAP.md
(its "ROADMAP item N") is REPORT.md, Future work. "The README" in that
text means the level README, now the matching REPORT.md section.

## Question

On one consumer GPU, what does a served open model's generation budget
cost in accuracy and tokens per correct answer, and what does load-time
quantization (bitsandbytes INT8, INT4 NF4) cost against FP16 in
accuracy, memory and throughput?

The original scope (below) asked the same of vLLM. vLLM has not run in
this repo; the serving engine is Transformers on PyTorch behind an
OpenAI-compatible API (REPORT.md, Build).

## Original scope (SPEC.md, committed 2026-11-04 in 018bd7e, before any code)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 3.
Hiring evidence: 5 of 5 (v2 build rank 3) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

### Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

vllm-serving-lab: EXPAND

Subject: LLM inference and serving.
v1 core [confirmed, v1]: serve an open model with vLLM through an OpenAI-compatible API; streaming, batching, load test; Deep benchmarks throughput and latency, explains PagedAttention with own numbers, tests AWQ and GPTQ models.
Added track 1, reasoning serving:
- Learn: serve one open reasoning model; log thinking tokens apart from answer tokens on every request.
- Build: a thinking-budget study: the same hard question set at three budget settings, scored for accuracy and cost per correct answer.
- Deep: a written rule for when more thinking pays and when it wastes money, with the study's numbers; one speculative decoding run measured against the plain run.
Added track 2, the container drill (register item 621):
- Build step inside this project: the serving stack ships as a small, layered, reproducible Docker image, scanned, with the build written up. This drill is the on-ramp to P20.
Fan-out: Book chapter "Serving Is The Product". Paper angle: thinking budgets and cost per correct answer, a serving study note. Course module: P03-vllm-serving-lab.

### Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.

## Proposed novelty (NOVELTY.md, draft, never verified)

Status: DRAFT, PROPOSED. Not verified. This text does not move into any public post, paper, or profile until the Jev claim_check gate passes on the exact text.

### Proposed claim

Against the vLLM and SGLang projects themselves: this project aims to differ in the way its SPEC.md section text defines (a from-zero or self-hosted build, measured in public, on a named home fleet, with the failure modes written down). The exact difference is stated at the project's wave, after the prior art is re-read.

### Prior art table

| Prior art | What it does | What this project does differently | Measured difference |
| --- | --- | --- | --- |
| the vLLM and SGLang projects themselves | See DOSSIER.md, Past | PROPOSED, see the claim above | NO MEASUREMENT YET |

### Honest size

If the novelty proves small, this file will say so and name the real value instead (a cleaner reproduction, a better benchmark, a first public dataset). No novelty is invented to fill this section.

## Build spec (level-2-build/BUILD-SPEC.md, committed with the Build code and results in 44e0b67)

Phase skill: spec-driven-development (house engineering pipeline).
Source scope: SPEC.md v1 core (serve an open model through an
OpenAI-compatible API; streaming, batching, load test) plus Added
track 1 Build (a thinking-budget study: the same hard question set at
three budget settings, scored for accuracy and cost per correct
answer). The container drill in Added track 2 is named in the README
as not done at Build on this node (no Docker on the fleet GPU node
was verified this cycle); it stays on the roadmap for Deep.

### Design

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

### Acceptance

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

## Deep spec (level-3-deep/DEEP-SPEC.md, committed with the Deep code and results in 1dae29d)

Phase skill: spec-driven-development (house engineering pipeline).
Compression basis: December Compression Protocol. Deep is the single
hardest capability in the SPEC Deep scope, built, tested, benchmarked,
with failure modes written. The rest of the v2 Deep scope is named in
the repo-root ROADMAP.md as the v1.1 backlog.

### Scope chosen

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

### Honesty rules for this level

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

### Also in this level

- A written thinking rule, derived from the Build study numbers,
  in level-3-deep/README.md: when more thinking pays and when it
  wastes money.
- A failure-modes section in level-3-deep/README.md.
- Repo-root ROADMAP.md: the rest of the v2 Deep scope (speculative
  decoding, serving with vLLM itself, the container drill) as the
  named v1.1 backlog.

### Acceptance

- Unit tests for the comparison and scoring logic pass with no GPU
  and no model download (fake backend), in CI as in earlier levels.
- results_quant.json committed, with per-question rows for every row
  that ran and exact errors for every route that failed.
- Full suite (Learn, Build, Deep) green locally and in CI.
- Root README carries the Deep benchmark row; DOSSIER.md updated to
  ship state; release v1.0 tagged after the PR merges with green CI.

## Next proposal (open, not written)

The 2026-11-07 review (REPORT.md, Review) names what a locked
confirmatory cycle would need: hypotheses with thresholds written into a
dated file before any run, for example "vLLM continuous batching raises
load-test completion tokens per second at concurrency 4 by at least 2x
over the sequential server". For the systems reliability profile it
would also state serving targets (availability, recovery time) and a
timed drill. It needs a Linux GPU node and the Owner's choice of
hypotheses; it is not drafted here.
