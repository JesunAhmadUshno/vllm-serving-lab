# Paper candidate brief: inference optimisation on consumer hardware paper

Tracker row: existing CANDIDATE row for P03 (publications pipeline
tracker, "Inference optimisation on consumer hardware paper")
Project: P03 vllm-serving-lab
Status: CANDIDATE, brief written 2026-11-05
Submission note: submission is Jesun's tap. Nothing is submitted by
this brief, and no venue has been contacted.

## Working title

Thinking budgets and quantized serving on consumer hardware:
cost per correct answer for a 0.5B instruct model on a laptop GPU

## Abstract draft

Based only on the measured v1.0 results recorded in the repo.
Numbers below are from one model, one question set, and one GPU
node, and are stated as such.

Serving cost is usually reported per request or per token, which
hides the two effects this study measures: a generation budget below
a model's completion length produces truncated working and no
correct answers, and load-time quantization trades accuracy and,
at small scale, speed for memory. A real open model
(Qwen/Qwen2.5-0.5B-Instruct, 494M parameters) was served through an
OpenAI-compatible API on an NVIDIA GeForce RTX 5050 Laptop GPU with
Transformers on PyTorch; vLLM was not used, because the node is
Windows with no WSL, and that substitution is stated in the paper
as a limit. In the thinking-budget study, the same 10 questions
with verifiable numeric answers were run greedily at budgets of
32, 128, and 384 new tokens: 0 of 10, 0 of 10, and 8 of 10 correct,
at 320, 1,280, and 2,522 completion tokens, with cost at 384 of
315.25 tokens and 7.196 seconds per correct answer. A stop rule on
the instructed Final answer line was implemented and the study
rerun with identical token counts: the rule never fired, because
the model ends in its own format. A concurrent load test served
12 of 12 requests at 0.477 requests per second and 61.05 completion
tokens per second (p50 8.2146 seconds, p95 9.0854 seconds), with
single-request generation, so concurrency measures queueing. In
the quantized-serving comparison at budget 384, FP16 scored 8 of
10 at 55.94 completion tokens per second and 996,322,304 GPU bytes
after load; bitsandbytes INT8 scored 6 of 10 (delta -0.200) at
10.60 tokens per second and 631,487,488 bytes (36.62 percent
smaller); bitsandbytes INT4 NF4 scored 5 of 10 (delta -0.300) at
32.36 tokens per second and 473,807,872 bytes (52.44 percent
smaller). The FP16 rows reproduce the budget study exactly, token
for token. The paper claims the budget rule and the cost-per-
correct framing on this set, and the small-model quantization
overhead finding as a one-node observation, not performance at
scale.

## Method outline

1. Serving surface: an OpenAI-compatible API on the Python standard
   library (ThreadingHTTPServer), streaming with server-sent
   events, and a usage object that counts thinking tokens apart
   from answer tokens by splitting generated text at the Final
   answer marker. Backends: a deterministic fake backend for CI
   and the Transformers backend for measured runs.
2. Question set: 10 questions with verifiable numeric answers
   (multi-step arithmetic, percentages, unit reasoning, a
   sequence, an age problem), the same set at every budget and
   every precision. Scoring: the number after the Final answer
   marker when present, else the last number in the text,
   tolerance 0.01. Extraction failures count as wrong.
3. Budget study: greedy decoding at 32, 128, and 384 new tokens.
   Per budget: correct count, completion tokens, wall seconds,
   tokens per correct answer, seconds per correct answer.
4. Stop rule experiment: a stopping criterion on a complete Final
   answer line, study rerun, token counts compared row by row,
   and the load test rerun on the stop-rule server.
5. Load test: 12 requests (the first 6 study questions, twice),
   concurrency 4, max 128 new tokens, over local HTTP on the
   serving node. Requests per second, completion tokens per
   second, p50 and p95 end-to-end latency.
6. Quantized comparison: FP16 (torch_dtype float16), INT8
   (bitsandbytes load_in_8bit), INT4 NF4 (load_in_4bit, compute
   float16), each in its own subprocess at budget 384. Memory is
   torch.cuda memory allocated after load and peak allocation
   across the run, the same counters for every row. Deltas
   (memory saving percent, accuracy delta, throughput ratio) are
   computed in code from the measured rows.
7. Learn context: a deterministic simulator establishes the token
   accounting the API logs (thinking at 63.5 percent of generated
   tokens on its 12-request teaching set) and the KV cache and
   paged allocation arithmetic. Simulated numbers are labelled
   simulated wherever they appear.

## Results table

Measured 2026-11-05, from the level READMEs and results files in
the repo. One model, 10 questions, one GPU node.

| Study | Setting | Correct | Cost or speed | Memory or latency |
| --- | --- | --- | --- | --- |
| Budget | 32 tokens | 0 of 10 | 320 completion tokens | n/a per correct |
| Budget | 128 tokens | 0 of 10 | 1,280 completion tokens | n/a per correct |
| Budget | 384 tokens | 8 of 10 | 315.25 tokens per correct, 7.196 s per correct | 2,522 completion tokens |
| Load test | 12 requests, concurrency 4 | 12 of 12 succeeded | 0.477 req/s, 61.05 tok/s | p50 8.2146 s, p95 9.0854 s |
| Quantized | FP16 | 8 of 10 | 55.94 tok/s, 315.25 tok/correct | 996,322,304 bytes after load |
| Quantized | INT8 | 6 of 10 | 10.60 tok/s, 444.00 tok/correct | 631,487,488 bytes, 36.62 percent smaller |
| Quantized | INT4 NF4 | 5 of 10 | 32.36 tok/s, 260.40 tok/correct | 473,807,872 bytes, 52.44 percent smaller |

Named cases: q03 at FP16 (extracts 61.2 against 74.8, a misapplied
discount and tax) is answered correctly by INT4 (74.8), so
quantization loss is question-shaped, not uniform; q06 under INT4
extracts 1375.49 in 9 tokens, an early-stop answer. The paper must
carry both cases, not only the summary deltas.

## Venue class note

Per SPEC.md, the paper angle for P03 is thinking budgets and cost
per correct answer, a serving study note, extended at Deep with
the quantized-serving comparison. Venue class is a workshop or
conference in ML systems or inference, or a journal of the AJSE
class, per the publications pipeline tracker rules. One venue at
a time. No venue is named as submitted, accepted, or contacted in
this brief. Fees, if any, arise only after acceptance and remain
Jesun's call under the free-only rule.

## Limits the paper must carry

- vLLM is not the engine. The study runs Transformers on PyTorch
  behind a vLLM-shaped API on a Windows node. Engine comparison
  with vLLM itself is future work (v1.1 item 2).
- One model (0.5B instruct, not a dedicated reasoning model), 10
  questions, one budget for the quantized rows, one run per
  config. Accuracy deltas rest on 2 and 3 questions; throughput
  ratios have no variance bounds.
- Quantization is bitsandbytes load-time only. No AWQ or GPTQ
  checkpoint rows, no speculative decoding measurement.
- No continuous batching: the server generates one request at a
  time, so load numbers measure queueing and are a floor.
- The Learn simulator numbers are simulated by design and must
  never be presented as measured serving results.

## Submission gate

Submission is Jesun's tap and nothing is submitted. This brief is
the candidate only. The tracker row for P03 already exists at
CANDIDATE; this brief does not edit the tracker.
