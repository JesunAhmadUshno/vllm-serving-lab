# DOSSIER: vllm-serving-lab (P03) v1, 2026-11-04

Written under the documentation and publishing standard (past, present, future). Updated at ship time.

## Past

Prior art for this project is named in NOVELTY.md (the vLLM and SGLang projects themselves). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

## Present

Current demand evidence: hiring value 5 of 5 (v2 build rank 3) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

Build update, 2026-11-05: Level 2 Build shipped. A real open model (Qwen/Qwen2.5-0.5B-Instruct, 494M parameters) is served through an OpenAI-compatible API on the fleet GPU node (BlackBox, RTX 5050) with Transformers on PyTorch as the engine; vLLM itself needs a Linux runtime the Windows node does not have, and that substitution is named in the level README. The thinking-budget study measured the same 10 hard questions at budgets 32, 128, and 384 tokens: 0, 0, and 8 correct, with every generation capped below 384 and a cost of 315.25 tokens per correct answer at 384. A stop rule at the instructed Final answer line was implemented and measured: it never fired, because the model ends answers in its own format. The load test served 12 of 12 concurrent requests at 0.477 requests per second. Build grade and deductions are recorded in the batch ledger.

Deep update, 2026-11-05: Level 3 Deep shipped (compressed Deep), v1.0. The quantized-serving comparison ran on BlackBox (RTX 5050) with the Build model, questions, scoring rule, and the 384-token budget: FP16 scored 8 of 10 at 55.94 tokens per second and 996,322,304 GPU bytes after load; bitsandbytes INT8 scored 6 of 10 at 10.60 tokens per second and 631,487,488 bytes (36.62 percent smaller); bitsandbytes INT4 NF4 scored 5 of 10 at 32.36 tokens per second and 473,807,872 bytes (52.44 percent smaller). The FP16 row reproduces the Build 384 row exactly, token for token. The Deep README also carries the written thinking rule derived from the Build study (a budget pays only above the model's natural completion length; cost per correct answer is the unit to watch). Deep grade 7,600 of 10,000 with named deductions in level-3-deep/README.md; the largest is that vLLM itself is still not the engine. Release v1.0.

## Future

The rest of the v2 Deep scope (speculative decoding, serving with vLLM itself on a Linux node, the container drill, AWQ and GPTQ checkpoint rows) is the named v1.1 backlog in the repo-root ROADMAP.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.
