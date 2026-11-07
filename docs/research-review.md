# Research review: vllm-serving-lab

Reviewed 2026-11-07 against a doctoral-level research rubric. Each
criterion gets a grade (A strong, B adequate, C weak, D missing), what
the repo has, what this change fixed, and, for every gap still open,
the exact compute or data that would close it. Gaps that could be
fixed on a CPU in minutes, with no GPU, paid API or long run, are
fixed in this change; nothing below is left open for lack of effort.

Sources: level-1-learn, level-2-build and level-3-deep (code, READMEs,
result JSON), ROADMAP.md, SPEC.md, and the new
`level-3-deep/significance.py`.

## Summary

| Criterion | Grade | One-line reason |
| --- | --- | --- |
| Research question | B | Clear practical questions (does a thinking budget pay; what does quantization cost), but the repo's namesake question (vLLM serving) is not yet asked of vLLM. |
| Hypotheses | C | Expectations are stated informally in the specs; no pre-registered, falsifiable hypotheses with thresholds. |
| Method | B | Greedy decoding, fixed question set, identical scoring and memory counters across configs, per-question rows saved. Engine is Transformers, not vLLM. |
| Baselines | B | FP16 is the baseline for quantization; budget 384 anchors the budget study; contiguous slabs are the baseline for paging. No engine baseline. |
| Statistics and uncertainty | B (was D) | Wilson intervals and exact paired McNemar tests now exist in code, CI and on the dashboard. Throughput still has no variance. |
| Reproducibility | B | Deterministic CPU parts reproduce in CI; GPU runs record versions and device; FP16 Deep rows reproduce Build rows exactly. Raw load-test latencies were not saved. |
| Threats to validity | B | Named in the level READMEs and now gathered on the dashboard and below. |
| Related work | C | DOSSIER.md gives background; no systematic comparison with published serving or quantization results. |
| Limitations | A | Deductions are named plainly in level-3-deep/README.md and ROADMAP.md. |

## Criterion detail

### Research question (B)

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

### Hypotheses (C)

The specs say what was expected, but no hypothesis was registered with
a decision threshold before the runs. This cannot be fixed after the
fact without HARKing; the honest repair is for the next experiment.

Open, needs a protocol, not compute: write the v1.1 hypotheses (for
example "vLLM continuous batching raises load-test completion tokens
per second at concurrency 4 by at least 2x over the sequential server")
with thresholds into a dated file before running them.

### Method (B)

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

### Baselines (B)

FP16 is a correct baseline for quantization and contiguous allocation
is the right baseline for paged allocation. Missing: an engine baseline
(vLLM vs Transformers) and pre-quantized checkpoints (AWQ, GPTQ) to
separate load-time from checkpoint quantization.

Open, needs compute: the vLLM run above, and AWQ and GPTQ checkpoints
of Qwen2.5-0.5B-Instruct run through `run_quant.py`, about 30 GPU
minutes. ROADMAP item 4.

### Statistics and uncertainty (B, was D)

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

### Reproducibility (B)

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

### Threats to validity (B)

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

### Related work (C)

DOSSIER.md covers the problem background and NOVELTY.md is a
self-labelled unverified draft. There is no table placing these
numbers next to published vLLM, PagedAttention or bitsandbytes
results.

Open, needs reading, not compute: a related-work section citing the
PagedAttention paper (Kwon et al., SOSP 2023), the bitsandbytes LLM.int8
and QLoRA/NF4 papers, and published budget-forcing or test-time compute
studies, with a sentence on how each result here agrees or differs.

### Limitations (A)

The level READMEs name every deduction plainly: no vLLM, no speculative
decoding, one run per config, n = 10, bitsandbytes only, no container
drill, sequential throughput. The dashboard repeats them in its limits
window.

## What this change added

- `level-3-deep/significance.py` and `tests/test_significance.py`
  (Wilson, exact McNemar, recorded-row checks; in CI).
- `site/`: the live dashboard, with `src/sim.check.mjs` asserting the
  browser computations against the repo's result files on every build.
- `.github/workflows/pages.yml` to publish it.
