# PUBLICITY: vllm-serving-lab (P03)

Publicity log under the documentation and publishing standard,
Section 7. Every finished level produces its written fan-out set,
each piece graded on the 1,000-point publicity rubric (accuracy 300,
research 200, gap fit 200, clarity 150, reach format 150), ship line
900. Drafts are prepared by the loop. Actual social posting follows
the standing platform rules and gates. The loop itself never posts
to social.

Fan-out set: NOT STARTED yet. The written fan-out set for this
project (blog, tutorial, website page copy, LinkedIn draft, release
notes) is Sprint 2 exit work and has not been prepared in this
cycle. No fan-out piece is claimed, graded, or shipped here.

Project state: v1.0, Learn, Build, and compressed Deep done
2026-11-05. Level grades on record: Learn 8,400 of 10,000, Build
8,000 of 10,000, Deep 7,600 of 10,000.

## Actions polish pack (2026-11-05)

Applied through a pull request on 2026-11-05, following the Actions
polish pack standard. The pack adds: CI on a Python 3.11 and 3.12
matrix with a ruff lint step (all pre-existing P03 CI steps
preserved verbatim in order: scaffold smoke test, Learn tests and
runner, Build tests and runner with the fake backend, Deep tests
tests.test_deep_quant and the Deep runner level-3-deep/run_deep.py),
CodeQL scanning, Dependabot for pip and GitHub Actions, a release
workflow that attaches a source tarball named
vllm-serving-lab-<tag>.tar.gz to v* tag releases, Labeler, Greetings
(snake_case inputs for actions/first-interaction@v3: repo_token,
issue_message, pr_message), Stale in comment-only mode
(days-before-close: -1, never closes), and a docs summary page at
docs/index.md for GitHub Pages. Ruff scope is recorded in ruff.toml
(E, F, I, line-length 120). Ruff fixes in this pull request were
style only: 7 initial errors (2 import-order I001, 5 module import
not at top E402 from the sys.path pattern), fixed by ruff fix
output and noqa E402 markers on the path-insert imports in the
tests, with no change to behaviour. requirements.txt records the
honest dependency picture: CI runs on the standard library plus
the fake backend, while the real backend path in
level-2-build/server.py and level-3-deep/run_quant.py imports
torch and transformers (and bitsandbytes for the quantized routes)
on the GPU node and is not installed in CI.

| Channel | Outcome on 2026-11-05 |
| --- | --- |
| GitHub Pages | Docs page added at docs/index.md in this pull request. Pages enablement is verified after merge by the coordinator; no live claim is made in this file before that check. |
| PyPI publishing | Held: needs a PyPI token, a new credential. Jesun's tap only. Not set up. |
| Social channels (LinkedIn, X, Facebook, Instagram, Threads) | Held: no posts made. The loop never posts to social; Threads waits for Jesun's tap per post under the standing gates. |

Publicity grade for this polish pack log piece, on the 1,000
point rubric (accuracy 300, research 200, gap fit 200, clarity
150, reach format 150), ship line 900:

| Piece | Accuracy /300 | Research /200 | Gap fit /200 | Clarity /150 | Reach /150 | Total /1000 |
| --- | --- | --- | --- | --- | --- | --- |
| PUBLICITY.md polish pack section (2026-11-05) | 295 | 180 | 195 | 145 | 95 | 910 |

At or above the 900 ship line. Named deductions: research minus
20, because the pack is a standard applied across repos and this
section records its application rather than new research; reach
minus 55, because the docs page is a single summary page in this
pull request with no live Pages claim yet, the release workflow
is unproven until the next tag, Greetings live proof is pending,
PyPI publishing is held, and no social channel carries this piece
by the standing gates; accuracy minus 5, gap fit minus 5, and
clarity minus 5 for summary compression of the full pack
standard, which lives in the batch files.
