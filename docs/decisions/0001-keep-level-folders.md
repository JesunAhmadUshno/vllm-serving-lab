# 0001: Keep code, evidence and results in the level folders

Date: 2026-11-08. Status: accepted.

## Context

The portfolio standard v2 lays a repo out as src/, experiments/ and
results/, and allows domain-named code folders when the README layout
table lists them. This repo's code, inputs and results live in
level-1-learn/, level-2-build/ and level-3-deep/. Their paths are
baked into the runners, the tests (which put each folder on sys.path),
significance.py, the dashboard's build-time imports, its build check
(site/src/sim.check.mjs), and the CI step that fails when a committed
result file differs from a fresh run.

## Decision

Keep the three folders and their files where they are. The README
layout table maps them to the standard's roles: code (*.py), evidence
inputs (requests.json, questions.json), recorded GPU runs
(results_real.json, results_real_stop.json, results_quant.json, never
edited after capture) and regenerated results (results_learn.json,
results_build.json, results_deep.json, results_significance.json,
checked in CI). Only the level READMEs and specs moved, into REPORT.md
and PROPOSAL.md.

## Consequences

No recorded path or result changes, so the migration cannot silently
alter evidence. The folder names still read as the old ladder; a later
rename to src/, experiments/ and results/ is one mechanical pull
request that must update every path listed above together.
