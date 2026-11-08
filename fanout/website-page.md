# Website project page copy: vllm-serving-lab

Site: jesunahmadushno.com
Status: COPY ONLY, draft for the project page. Dated 2026-11-05.
Nothing on any website was changed by this file.

## Headline

vllm-serving-lab: thinking budgets and quantized serving, measured in the open

## Summary

vllm-serving-lab is an open LLM inference and serving project built
by Jesun Ahmad Ushno as project P03 in Roadmap v3. It builds the
serving stack in three levels: a deterministic simulator for token
accounting, KV cache math, and paged allocation; a real open model
served through an OpenAI-compatible API with a thinking-budget
study and a load test; and a quantized-serving comparison of FP16
against bitsandbytes INT8 and INT4 NF4. The engine is Transformers
on PyTorch, not vLLM: the fleet GPU node is Windows with no WSL,
and vLLM needs a Linux runtime. That substitution is named on this
page, in the repo, and in every level write-up.

Status: v1.0, 2026-11-05. Learn, Build, and compressed Deep are
done. Total tests: 49, all passing at v1.0.

## Results

All results below are recorded runs in the GitHub repo. Learn
results are simulated and deterministic. Build and Deep results are
real runs on BlackBox, an NVIDIA GeForce RTX 5050 Laptop GPU, with
Qwen/Qwen2.5-0.5B-Instruct and greedy decoding.

| Level | Set | Headline result | Notes |
| --- | --- | --- | --- |
| Learn | 12 labelled requests, simulated | Thinking 1,590 of 2,505 generated tokens (63.5 percent); paged waste 73 against 1,657 contiguous | KV 131,072 bytes per token, 521,011,200 bytes total; 79 steps at 31.708861 tokens per step; 20 tests |
| Build | 10 questions at budgets 32, 128, 384, real model | 0 of 10, 0 of 10, 8 of 10 correct; 315.25 tokens per correct at 384 | Stop rule never fired; load test 12 of 12 at 0.477 requests per second, 61.05 tokens per second; 15 tests, 35 total |
| Deep (compressed) | Same 10 questions at budget 384, three precisions | FP16 8 of 10 at 55.94 tokens per second; INT8 6 of 10 at 10.60; INT4 NF4 5 of 10 at 32.36 | INT8 memory 36.62 percent smaller than FP16, INT4 52.44 percent smaller; FP16 rows reproduce Build exactly; 14 tests, 49 total |

## Honest limits

- vLLM itself is not the engine in v1.0. Serving with vLLM on a
  Linux node is v1.1 item 2 in the repo REPORT.md (Future work).
- One small model and 10 questions. Accuracy deltas in Deep rest
  on 2 and 3 questions and are not population estimates.
- One run per Deep config, so throughput has no variance bounds.
- Quantization is bitsandbytes load-time only. No AWQ or GPTQ
  checkpoint rows, no speculative decoding, and no container drill
  in v1.0. All are named in the v1.1 backlog.

## Links

- GitHub repo: https://github.com/JesunAhmadUshno/vllm-serving-lab
- Release v0.1 (Learn): https://github.com/JesunAhmadUshno/vllm-serving-lab/releases/tag/v0.1
- Release v0.5 (Build): https://github.com/JesunAhmadUshno/vllm-serving-lab/releases/tag/v0.5
- Release v1.0 (Learn, Build, Deep): https://github.com/JesunAhmadUshno/vllm-serving-lab/releases/tag/v1.0
- Tutorial: fanout/tutorial.md in the GitHub repo
- Blog post draft: fanout/blog-post.md in the GitHub repo

## License note

Code: Apache-2.0. Tutorials: CC BY 4.0.
