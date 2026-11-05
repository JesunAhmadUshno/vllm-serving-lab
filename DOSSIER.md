# DOSSIER: vllm-serving-lab (P03) v1, 2026-11-04

Written under the documentation and publishing standard (past, present, future). Updated at ship time.

## Past

Prior art for this project is named in NOVELTY.md (the vLLM and SGLang projects themselves). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

## Present

Current demand evidence: hiring value 5 of 5 (v2 build rank 3) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

Build update, 2026-11-05: Level 2 Build shipped. A real open model (Qwen/Qwen2.5-0.5B-Instruct, 494M parameters) is served through an OpenAI-compatible API on the fleet GPU node (BlackBox, RTX 5050) with Transformers on PyTorch as the engine; vLLM itself needs a Linux runtime the Windows node does not have, and that substitution is named in the level README. The thinking-budget study measured the same 10 hard questions at budgets 32, 128, and 384 tokens: 0, 0, and 8 correct, with every generation capped below 384 and a cost of 315.25 tokens per correct answer at 384. A stop rule at the instructed Final answer line was implemented and measured: it never fired, because the model ends answers in its own format. The load test served 12 of 12 concurrent requests at 0.477 requests per second. Build grade and deductions are recorded in the batch ledger.

## Future

Open problems and the project's next steps sit in the Deep level scope in SPEC.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.
