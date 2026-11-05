# vllm-serving-lab

LLM inference and serving, built in three levels with measured results at each level. The engine is Transformers on PyTorch, not vLLM (the GPU node is Windows with no WSL); that substitution is named in the level write-ups.

- Learn: simulated serving on CPU. Token accounting with thinking tokens logged apart from answer tokens, KV cache math, paged allocation, and token-step batching.
- Build: a real open model (Qwen/Qwen2.5-0.5B-Instruct) served through an OpenAI-compatible API on BlackBox (RTX 5050), a thinking-budget study, and a concurrent load test. Thinking-budget study (real model, 10 questions): 0 of 10 correct at a budget of 32, 0 of 10 at 128, and 8 of 10 at 384.
- Deep (compressed): a quantized-serving comparison on the same model and questions at budget 384. FP16: 8 of 10 correct at 55.94 tokens per second. INT8: 6 of 10 correct. INT4: 5 of 10 correct, with memory 52.44 percent smaller than FP16.

Grades on record (of 10,000): Learn 8,400, Build 8,000, Deep 7,600.

Every number above is already published in this repo (README.md and the level write-ups) and the deterministic parts are reproduced by the runners in this repo, checked in CI. The full write-ups are in the level folders: level-1-learn, level-2-build, level-3-deep.

Releases: https://github.com/JesunAhmadUshno/vllm-serving-lab/releases
