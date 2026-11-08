# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning. The tags v0.1, v0.5 and v1.0 were
cut under the old Learn / Build / Deep ladder; they are kept as they
are, and v1.0 there means "all three levels shipped", not the
"v1.0 Released" verdict of the portfolio standard v2. Under the
standard the repo is Functional, not pre-registered.

## [Unreleased]

### Changed
- 2026-11-08: Adopted the portfolio standard v2. README opens with
  the question, headline with its interval, field, job family and
  verdict, then Reproduce, Lifecycle and Layout. SPEC.md, NOVELTY.md,
  level-2-build/BUILD-SPEC.md and level-3-deep/DEEP-SPEC.md moved into
  PROPOSAL.md (not locked: no plan with hypotheses was locked before
  the runs). The three level READMEs, DOSSIER.md,
  docs/research-review.md and ROADMAP.md moved into REPORT.md.
  Self-assigned grades and publicity scores left the public repo
  (grades are internal). PUBLICITY.md moved to fanout/PUBLICITY.md,
  with the README's fan-out list. docs/index.md (the legacy Pages
  summary, superseded by the dashboard) removed; its numbers are all in
  REPORT.md. Code, evidence and results stay in the level folders
  (docs/decisions/0001).
- 2026-11-08: level-3-deep/significance.py also writes
  level-3-deep/results_significance.json, so the headline intervals and
  p values are a committed result that CI regenerates.
- 2026-11-08: Dashboard reads the verdict from README.md and the v1.1
  backlog from REPORT.md, links REPORT.md and PROPOSAL.md instead of the
  removed files, and no longer mentions grades.
- 2026-11-08: CI runs with read-only token permissions, replaces the
  scaffold smoke test with a layout test (required files, abstract and
  lifecycle, no public grade), and fails if any deterministic result
  file differs from a fresh run. CodeQL covers Python and JavaScript;
  Dependabot covers pip, npm (site/) and Actions.

### Added
- 2026-11-08: CHANGELOG.md, CITATION.cff, SECURITY.md,
  docs/decisions/0001.
- 2026-11-07: Wilson intervals and exact McNemar tests over the
  recorded per-question rows (level-3-deep/significance.py), with tests.
- 2026-11-07: Live dashboard in site/, published by the Pages workflow;
  its build fails if its browser computations differ from the result
  files.
- 2026-11-05: Written fan-out set in fanout/ (drafts; nothing posted).
- 2026-11-05: Actions polish pack: CI on Python 3.11 and 3.12 with
  ruff, CodeQL, Dependabot, release, labeler, greetings, stale.

## [1.0] - 2026-11-05 (ladder tag v1.0)

### Added
- Level 3, Deep (compressed): quantized-serving comparison of FP16
  against bitsandbytes INT8 and INT4 NF4 on the GPU node, same model,
  same 10 questions, budget 384; the written thinking rule.

Compressed Deep note: the rest of the v2 Deep scope (speculative
decoding, vLLM itself, the container drill, AWQ and GPTQ rows) is the
v1.1 backlog in REPORT.md, Future work. Nothing in it is claimed.

## [0.5] - 2026-11-05 (ladder tag v0.5)

### Added
- Level 2, Build: Qwen/Qwen2.5-0.5B-Instruct served through an
  OpenAI-compatible API on the GPU node (Transformers on PyTorch, not
  vLLM), the thinking-budget study at 32, 128 and 384 tokens, the stop
  rule experiment, and a concurrent load test.

## [0.1] - 2026-11-05 (ladder tag v0.1)

### Added
- Level 1, Learn: deterministic serving simulator on CPU with token
  accounting, KV-cache sizing, paged block allocation and token-step
  batching.
- 2026-11-04: Scaffold: scope spec, dossier, novelty draft (018bd7e).
