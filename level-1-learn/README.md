# Level 1, Learn: vllm-serving-lab

Status: Learn DONE, 2026-11-05. Simulated on CPU. No real model was served.

Honest limit, first: BlackBox, the GPU node for this project, was unreachable
in this cycle (NEEDS-BLACKBOX-ON for real GPU serving). So Learn teaches
serving with a deterministic simulator and exact token accounting on CPU.
Real vLLM serving of one open reasoning model is Build scope. No GPU number
and no real vLLM number is claimed anywhere in this level. Every number
below is simulated, deterministic, with its method stated.

## What this level teaches

The SPEC Learn scope is: serve one open reasoning model, and log thinking
tokens apart from answer tokens on every request. Three ideas carry that:

1. Token accounting. A request is prompt tokens plus thinking tokens plus
   answer tokens. Thinking tokens are generated but are not the answer, so
   they cost compute and KV cache and must be logged apart. Non-reasoning
   requests have thinking = 0.
2. KV cache size. Each token in the sequence needs key and value storage
   in every layer. Bytes per token = 2 (K and V) * layers * kv_heads *
   head_dim * bytes per element.
3. Paged allocation and batching. PagedAttention style blocks of 16 tokens
   waste far less than one contiguous slab per request, and a serving
   engine shares a fixed per-step token budget across the active batch.

## Files

- `serving_sim.py`: request record, log line, KV math, block allocation,
  and the token-step batching simulation. Stdlib only.
- `requests.json`: 12 labelled requests, R01 to R12, with varied prompt,
  thinking, and answer counts. Five are non-reasoning (thinking = 0).
- `run_learn.py`: runs the accounting and simulation, prints the full
  table, and writes `results_learn.json`. Run it twice: the file is
  byte-identical, because the simulation has no randomness and no clock.
- `results_learn.json`: the measured (simulated) results below.

## Method

- Example model config, named `example-32L-8KV-128D-fp16`: 32 layers,
  8 KV heads, head_dim 128, 2 bytes per element (fp16).
- KV bytes per token = 2 * 32 * 8 * 128 * 2 = 131,072 bytes (128 KiB).
  A request KV size is its total tokens times 131,072.
- Paged allocation: blocks of 16 tokens, allocated tokens are total
  rounded up to a whole block, waste is allocated minus total.
- Contiguous baseline slab rule: every request is rounded up to the next
  whole multiple of 256 tokens, however small it is.
- Batching simulation: all 12 requests are active from step 1. Each step
  has a budget of 32 generation tokens. Inside a step, repeated round
  robin passes in request order give 1 token to each active request per
  pass until the budget is used or all active requests finish. Only
  output tokens (thinking plus answer) are generated. Prompt tokens are
  prefill: they count in KV and block accounting, not in steps.
- Throughput is reported in tokens per step, never tokens per second.
  No wall time is measured or claimed.

## Results (simulated, deterministic)

Per-request table, from `run_learn.py` run on 2026-11-05:

| id | prompt | thinking | answer | output | total | kv_bytes | paged_blocks | paged_waste | contig_waste | steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | 120 | 200 | 80 | 280 | 400 | 52428800 | 25 | 0 | 112 | 58 |
| R02 | 60 | 0 | 45 | 45 | 105 | 13762560 | 7 | 7 | 151 | 15 |
| R03 | 200 | 350 | 120 | 470 | 670 | 87818240 | 42 | 2 | 98 | 72 |
| R04 | 45 | 80 | 30 | 110 | 155 | 20316160 | 10 | 5 | 101 | 32 |
| R05 | 90 | 0 | 60 | 60 | 150 | 19660800 | 10 | 10 | 106 | 20 |
| R06 | 300 | 500 | 150 | 650 | 950 | 124518400 | 60 | 10 | 74 | 79 |
| R07 | 30 | 0 | 25 | 25 | 55 | 7208960 | 4 | 9 | 201 | 9 |
| R08 | 150 | 120 | 90 | 210 | 360 | 47185920 | 23 | 8 | 152 | 53 |
| R09 | 200 | 0 | 100 | 100 | 300 | 39321600 | 19 | 4 | 212 | 32 |
| R10 | 55 | 250 | 70 | 320 | 375 | 49152000 | 24 | 9 | 137 | 66 |
| R11 | 180 | 90 | 110 | 200 | 380 | 49807360 | 24 | 4 | 132 | 53 |
| R12 | 40 | 0 | 35 | 35 | 75 | 9830400 | 5 | 5 | 181 | 17 |

Totals across the 12 requests:

- Prompt tokens 1,470. Thinking tokens 1,590. Answer tokens 915.
  Output tokens 2,505. Total tokens 3,975.
- Thinking is 63.5 percent of all generated tokens (1,590 of 2,505).
  That is the Learn lesson for the SPEC scope: if thinking is not logged
  apart, most of the generation cost is invisible.
- KV bytes total 521,011,200 for the example config.
- Paged allocation: 4,048 tokens allocated, 73 waste tokens.
- Contiguous slab allocation: 5,632 tokens allocated, 1,657 waste tokens.
  Paged blocks cut waste by 1,584 tokens on this set (73 vs 1,657).
- Simulation: 79 total steps for 2,505 output tokens, simulated
  throughput 31.708861 tokens per step against a budget of 32. The gap
  below 32 is the last step, where only R06 remains and finishes early.

Every request also prints a log line that separates the split, for
example: `request=R01 prompt=120 thinking=200 answer=80 output=280 total=400 reasoning=true`

## Tests

`tests/test_learn_serving.py`: 20 unit tests with hand-computed values
for KV bytes per token and per request, thinking and answer split sums,
paged vs contiguous waste, and simulation step totals for small cases
that can be checked by hand. `tests/test_smoke.py` still passes.

## Failure modes and limits

- No real model was served. Token counts in `requests.json` are a
  labelled teaching set, not traces from a real reasoning model.
- The simulator has no prefill cost, no memory bandwidth, no scheduling
  delay, no arrival times, and no sampling. Real vLLM numbers will
  differ and belong to Build, on BlackBox, with a real model.
- KV math assumes every token is kept for the whole request and uses
  the example config only. Grouped-query attention, KV quantization,
  and prefix caching change real sizes.
- The round robin step rule is a teaching model of continuous batching,
  not the vLLM scheduler. It is exact for its stated rule, which is why
  the tests can hold hand-computed step counts.
- Throughput in tokens per step cannot be converted to tokens per
  second without a real step time, which this level does not measure.

## Next

Build: serve one open reasoning model with real vLLM on BlackBox,
log thinking apart from answer on real requests, then run the
thinking-budget study from SPEC.md.
