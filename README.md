# vllm-serving-lab

**Question.** On one consumer GPU, what does a served open model's
generation budget cost in accuracy and tokens per correct answer, and
what does load-time quantization cost against FP16 in accuracy, memory
and throughput?

**Headline.** Served through this repo's OpenAI-compatible API on one
RTX 5050 laptop GPU, Qwen2.5-0.5B-Instruct answered 8 of 10 questions
at a 384-token budget (Wilson 95% interval 0.49 to 0.94) and 0 of 10 at
128 (exact McNemar p = 0.0078); bitsandbytes INT8 and INT4 NF4 cut GPU
memory by 36.62 and 52.44 percent, and their lower scores (6 and 5 of
10) cannot be told apart from FP16 at n = 10 (p = 0.5 and 0.375). The
engine is Transformers on PyTorch: vLLM has not run here.

**Field.** Systems and infrastructure, reliability. **Job family.** ML
inference and serving engineering.

**Verdict.** Functional, not pre-registered. Tests pass in CI on Python
3.11 and 3.12, the deterministic result files regenerate unchanged in
CI, and the served model ran end to end on the GPU node with its
recorded runs committed (level-2-build/results_real.json,
level-3-deep/results_quant.json). No plan with hypotheses was locked
before the runs, so the repo is capped at Functional (PROPOSAL.md).

Live dashboard: https://jesunahmadushno.github.io/vllm-serving-lab/

## Reproduce

```text
python3 -m unittest discover -s tests -p "test_*.py" -v
python3 level-1-learn/run_learn.py
python3 level-2-build/run_build.py
python3 level-3-deep/run_deep.py
python3 level-3-deep/significance.py
git diff --exit-code -- level-1-learn/results_learn.json level-2-build/results_build.json level-3-deep/results_deep.json level-3-deep/results_significance.json
```

Python 3.11 or 3.12, standard library only, CPU only, seconds. The
runners use a deterministic fake backend for the serving mechanics; a
clean `git diff` means the committed results were reproduced.
`significance.py` recomputes the headline intervals and p values from
the recorded GPU rows. The GPU runs themselves (`level-2-build/run_real.py`,
`level-3-deep/run_quant.py`) need a CUDA GPU with torch, transformers
and bitsandbytes and are not run in CI (REPORT.md, Build and Deep).

## Lifecycle

- Question: DONE 2026-11-04
- Proposal: OPEN
- Baseline: DONE 2026-11-05
- Build: DONE 2026-11-05
- Evaluate: OPEN
- Report: OPEN
- Review: OPEN
- Release: OPEN

Proposal is open because no plan was locked before the runs (the
2026-11-04 scope spec has no hypotheses or pass lines). Evaluate waits
on a locked confirmatory cycle under the reliability profile, and so
does Report: REPORT.md records what ran, unscored against hypotheses.
The 2026-11-07 research review is a read-through, not the re-derivation
the Review stage requires. The tags v0.1, v0.5 and v1.0 are old ladder
tags, not a "v1.0 Released" verdict (CHANGELOG.md).

## Layout

| Path | Role in the standard | What it holds |
| --- | --- | --- |
| PROPOSAL.md | Proposal (not locked) | Original scope, proposed novelty, Build and Deep specs, what a locked proposal needs |
| REPORT.md | Report | Results with evidence, observations, background, the three runs, review, limits, future work |
| level-1-learn/ | Code, evidence, results | Serving simulator (serving_sim.py, run_learn.py), requests.json, results_learn.json |
| level-2-build/ | Code, evidence, results | OpenAI-compatible server, budget study, load test (server.py, study.py, load_test.py, run_build.py, run_real.py), questions.json, results_build.json (fake), results_real.json and results_real_stop.json (GPU runs) |
| level-3-deep/ | Code, evidence, results | Quantized comparison (quant_compare.py, run_deep.py, run_quant.py), significance.py, results_deep.json (fake), results_quant.json (GPU run), results_significance.json |
| tests/ | Tests | Unit tests per level, significance tests, and the layout test |
| site/ | Dashboard | Browser simulator and replay of the recorded runs, checked against the result files at build time |
| docs/decisions/ | Decisions | 0001: why code and results stay in the level folders |
| fanout/ | Outreach drafts | Blog, tutorial, release notes and other drafts (nothing posted), PUBLICITY.md log |
| CHANGELOG.md, CITATION.cff, SECURITY.md, LICENSE | Release files | |

No experiments/ or results/ folder: the evidence and results sit next
to the code that wrote them (docs/decisions/0001).

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
