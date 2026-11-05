# PUBLICITY: vllm-serving-lab (P03)

Publicity log under the documentation and publishing standard,
Section 7. Every finished level produces its written fan-out set,
each piece graded on the 1,000-point publicity rubric (accuracy 300,
research 200, gap fit 200, clarity 150, reach format 150), ship line
900. Drafts are prepared by the loop. Actual social posting follows
the standing platform rules and gates. The loop itself never posts
to social.

Fan-out set date: 2026-11-05 (Sprint 2 early work). Project state:
v1.0, Learn, Build, and compressed Deep done 2026-11-05, 49 tests
passing. Level grades on record: Learn 8,400 of 10,000, Build
8,000 of 10,000, Deep 7,600 of 10,000.

## Channel log (2026-11-05)

| Channel | Outcome on 2026-11-05 |
| --- | --- |
| GitHub releases | Live: v0.1 https://github.com/JesunAhmadUshno/vllm-serving-lab/releases/tag/v0.1 , v0.5 https://github.com/JesunAhmadUshno/vllm-serving-lab/releases/tag/v0.5 , v1.0 https://github.com/JesunAhmadUshno/vllm-serving-lab/releases/tag/v1.0 (verified with gh release list, 2026-11-05). Draft release notes for all three are in fanout/release-notes.md. No new release was created for the fan-out. |
| Portfolio index | Held: fan-out links are ready, the index update is a coordinator step outside this cycle. Named reason: held for index refresh. |
| Website page | Draft in fanout/website-page.md. Held: not published to jesunahmadushno.com. |
| Blog | Draft in fanout/blog-post.md. Held: not posted. |
| Tutorial | In repo: fanout/tutorial.md. Every command in it was run verbatim and verified on 2026-11-05 in a fresh clone (smoke test, three runners, three per-level test discoveries, full suite at 49 tests, all passing). The recorded GPU-node results are labelled in the tutorial as recorded results, not local commands. |
| LinkedIn post | Draft in fanout/linkedin-post.md, marked DRAFT NOT POSTED, link carried as the first-comment line per the LinkedIn rule. Held: not posted, loop never posts. |
| X | Draft held: no X copy is shipped in this set beyond the LinkedIn draft source text. Held under the standing social rules; posting is outside the loop. |
| Facebook | Draft held: no Facebook copy posted. Standing rule applies (no link in body or comments, keyword call to action fulfilled by DM). Held: not posted, loop never posts. |
| Instagram | Draft held: no Instagram copy posted. Standing rule applies (no clickable link in caption, link in bio). Held: not posted, loop never posts. |
| Threads | Draft held: no Threads copy posted. Threads waits for Jesun's tap per post, hash-locked on exact text and media. Held: awaiting Jesun's tap. |
| Video | Held: engine queue, January production for lab projects per the December Compression Protocol. No video was produced in this cycle. |
| Book chapter | Draft angle in fanout/blog-post.md and SPEC.md ("Serving Is The Product"). Held: chapter draft follows the finished project set; not a published chapter. |
| Paper candidate | Draft brief in fanout/paper-candidate.md (thinking budgets and quantized serving, cost per correct answer). Submission is Jesun's tap. Nothing submitted, no venue contacted. The tracker row for P03 already exists at CANDIDATE ("Inference optimisation on consumer hardware paper"); this fan-out does not edit the tracker. |

## Publicity grades (1,000-point rubric, ship line 900)

| Piece | Accuracy /300 | Research /200 | Gap fit /200 | Clarity /150 | Reach format /150 | Total /1000 |
| --- | --- | --- | --- | --- | --- | --- |
| fanout/blog-post.md | 295 | 190 | 195 | 145 | 140 | 965 |
| fanout/tutorial.md | 300 | 185 | 195 | 148 | 142 | 970 |
| fanout/website-page.md | 295 | 180 | 190 | 148 | 145 | 958 |
| fanout/linkedin-post.md | 290 | 175 | 190 | 145 | 140 | 940 |
| fanout/paper-candidate.md | 295 | 195 | 195 | 140 | 130 | 955 |
| fanout/release-notes.md | 298 | 185 | 188 | 144 | 137 | 952 |

All six pieces are at or above the 900 ship line. No piece was
shipped ungraded. Piece totals sum to 5,740 of 6,000, which scales
to an overall fan-out grade of 9,567 of 10,000.

## Named deductions (unclosable in this cycle)

- Blog post, minus 35: research is condensed from DOSSIER.md v1,
  whose prior art section is brief (minus 10 research); the
  10-question Deep deltas carry reader interpretation risk even
  stated plainly (minus 5 accuracy); condensed gap framing
  (minus 5 gap fit, minus 5 clarity); the piece is a held draft
  with no live channel yet (minus 10 reach).
- Tutorial, minus 30: research framing is thinner than the blog by
  design, because the module teaches runs rather than the field
  (minus 15 research); no worked example beyond the shipped sets
  (minus 5 gap fit, minus 2 clarity, minus 8 reach).
- Website page, minus 42: condensed research (minus 20), condensed
  gap framing (minus 10), and the page is copy only, not a live
  page (minus 5 reach, minus 5 accuracy, minus 2 clarity).
- LinkedIn post, minus 60: short form compresses the research and
  the failure mode detail (minus 25 research, minus 10 accuracy,
  minus 10 gap fit, minus 5 clarity, minus 10 reach, because the
  post is a held draft and carries no live link in the body by
  rule).
- Paper candidate, minus 45: this is a brief, not a drafted paper
  (minus 20 reach, minus 10 clarity); venue is classed, not named
  (minus 5 research, minus 5 gap fit, minus 5 accuracy for the
  single-node, 10-question evidence base the paper must carry).
- Release notes, minus 48: notes summarise releases that already
  exist, so research and gap framing are summary grade (minus 15
  research, minus 12 gap fit); format is repo-bound (minus 13
  reach, minus 2 accuracy, minus 6 clarity).

## Jev gate on the fan-out claim (2026-11-05)

claim_check on the blog post main results paragraph, extracted
byte-exact from fanout/blog-post.md (sha256 of the paragraph:
4ad078486ad3d89976b1e353b9b72f60f894246ecdd0b61037135f115f4a4f0f).
Model: jev-1.13.0. Usage: 3,113 input tokens, 20 output tokens.
Question id: supported. Noul: 0.68. Route: REVIEW (bands: clear at
0.85 or above, block at 0.10 or below, review between). Not
blocked; the claim ships as a held draft under the review route,
with all arithmetic pre-computed in code before the call. Spec
and results are in the goal hidden_files (jev-p03-fanout-claim-
spec-2026-11-05.json and jev-p03-fanout-claim-results-2026-11-05.json).

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
