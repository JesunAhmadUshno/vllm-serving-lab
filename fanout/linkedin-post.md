# LinkedIn post draft: P03 vllm-serving-lab v1.0

Status: DRAFT NOT POSTED
Date drafted: 2026-11-05
Author: Jesun Ahmad Ushno
Rule for this platform: the link goes in the first comment, not in
the post body. The first comment line is at the end of this file.

## Post body

A thinking budget below the length a model needs to finish does not
buy cheaper answers. It buys no answers. I measured it.

vllm-serving-lab v1.0 is done: Learn, Build, and a compressed Deep,
49 tests. One naming fact first, because it matters: the engine is
Transformers on PyTorch, not vLLM. My GPU node is Windows with no
WSL, and vLLM needs Linux. No vLLM number is claimed anywhere here.

Learn: a deterministic simulator for serving. Thinking tokens were
1,590 of 2,505 generated tokens (63.5 percent). If you do not log
thinking apart from the answer, most of your generation cost is
invisible. Paged blocks wasted 73 tokens where contiguous slabs
wasted 1,657 on the same set.

Build: Qwen2.5-0.5B-Instruct served through an OpenAI-compatible API
on an RTX 5050 laptop GPU. Same 10 questions at three budgets: 0 of
10 correct at 32 tokens, 0 of 10 at 128, 8 of 10 at 384, at 315.25
tokens per correct answer. At the low budgets every generation was
cut mid-working. I also measured a stop rule on the Final answer
line: it never fired, because the model does not emit that line, and
its load test was slower. An unmatched stop rule is dead code with
a per-step cost.

Deep: quantized serving on the same questions at budget 384. FP16:
8 of 10, 55.94 tokens per second, 996,322,304 GPU bytes after load.
INT8: 6 of 10, 10.60 tokens per second, 36.62 percent less memory.
INT4 NF4: 5 of 10, 32.36 tokens per second, 52.44 percent less
memory. INT8 saving memory while running 5 times slower is the
finding I did not expect: at 0.5B scale, dequantization overhead
beats the bandwidth saving.

The honest limits, in the post and not in a footnote: one small
model, 10 questions, one run per Deep config, bitsandbytes
quantization only, and vLLM itself still to run on a Linux node.
All of that is named as the v1.1 backlog in the repo.

If you serve models: what sets your generation budget, a measured
completion length, or a guess?

## First comment (post this as the first comment, per the standing rule)

Repo, releases v0.1, v0.5, and v1.0, and the full tutorial: https://github.com/JesunAhmadUshno/vllm-serving-lab
