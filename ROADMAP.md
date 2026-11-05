# ROADMAP: vllm-serving-lab (P03), v1.1 backlog

Written at v1.0 ship time, 2026-11-05. Under the December
Compression Protocol, Deep shipped the single hardest capability in
its scope: the quantized-serving comparison (FP16 against INT8 and
INT4, measured on the fleet GPU node). Everything below is the rest
of the v2 Deep scope, named here as the v1.1 backlog. Nothing below
is built, and nothing below is claimed.

## v1.1 backlog

1. Speculative decoding: one measured run of a draft-model or
   prompt-lookup speculative setup against the plain FP16 run, same
   question set, reporting acceptance rate, latency change, and
   whether accuracy moves. This was in the original Deep scope
   (SPEC.md, Added track 1) and was displaced by the quantization
   comparison under compression.
2. Serving with vLLM itself: the Build and Deep levels run
   Transformers on PyTorch behind an OpenAI-compatible API because
   the fleet GPU node is Windows with no WSL and vLLM needs a Linux
   runtime. On a Linux node (or a free Linux GPU tier), serve the
   same model with vLLM, rerun the thinking-budget study and the
   quantization comparison, and publish the side-by-side engine
   table. No vLLM number exists in v1.0.
3. The container drill (SPEC.md, Added track 2, register item 621):
   the serving stack as a small, layered, reproducible Docker
   image, scanned, with the build written up. Not done in v1.0: no
   Docker runtime was verified on the fleet GPU node.
4. AWQ and GPTQ checkpoints: v1.0 quantization is bitsandbytes
   on-the-fly (INT8 and INT4 NF4). Pre-quantized AWQ and GPTQ
   checkpoints of the same model family, run and measured the same
   way, would separate checkpoint quantization from load-time
   quantization.
5. Larger and harder question sets: 10 questions at one budget is a
   study, not a benchmark suite. A 100-question set with difficulty
   tiers would tighten every number in the Deep table.
6. Continuous batching and KV-cache pressure: the Build server
   generates one request at a time. A batching engine comparison
   under load, with KV-cache limits hit on purpose, is open.
7. Human calibration of the judge-free scoring: Deep scoring is the
   Build extraction rule. A small human-checked sample would bound
   the extraction error rate.

## Order

Items 2 and 1 first (they close the two named substitutions in the
v1.0 record), then 3, then the rest as capacity allows in the 2027
improvement phase.
