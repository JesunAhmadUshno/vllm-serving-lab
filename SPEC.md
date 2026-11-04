# SPEC: vllm-serving-lab (P03)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 3.
Hiring evidence: 5 of 5 (v2 build rank 3) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

## Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

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

## Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.
