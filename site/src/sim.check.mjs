// Known-answer check, run by `npm run build` before vite. Every computation the page does in
// the browser must reproduce the repo's own result files, or the build fails.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  kvBytesPerToken, kvBytesForRequest, pagedAllocation, contiguousAllocation, simulateBatch,
  budgetWhatIf, wilson, mcnemarExact, discordant, quantDeltas, kwarg, totalTokens,
} from './sim.js';

const read = (p) => fs.readFileSync(new URL(`../../${p}`, import.meta.url), 'utf8');
const learn = JSON.parse(read('level-1-learn/results_learn.json'));
const requests = JSON.parse(read('level-1-learn/requests.json'));
const real = JSON.parse(read('level-2-build/results_real.json'));
const quant = JSON.parse(read('level-3-deep/results_quant.json'));
const runReal = read('level-2-build/run_real.py');

// KV-cache calculator vs results_learn.json
const cfg = learn.model_config;
assert.equal(kvBytesPerToken(cfg), cfg.kv_bytes_per_token);
assert.equal(kvBytesPerToken(cfg), 131072);
const kvTotal = requests.reduce((s, r) => s + kvBytesForRequest(r, cfg), 0);
assert.equal(kvTotal, learn.totals.kv_bytes_total);
assert.equal(kvTotal, 521011200);

// Paged vs contiguous allocator, per request and in total
const byId = Object.fromEntries(learn.per_request.map((r) => [r.request_id, r]));
let pa = 0, pw = 0, ca = 0, cw = 0;
for (const r of requests) {
  const p = pagedAllocation(totalTokens(r), learn.block_size_tokens);
  const c = contiguousAllocation(totalTokens(r), learn.contiguous_slab_tokens);
  const want = byId[r.request_id];
  assert.equal(p.blocks, want.paged_blocks, r.request_id);
  assert.equal(p.waste, want.paged_waste_tokens, r.request_id);
  assert.equal(c.waste, want.contiguous_waste_tokens, r.request_id);
  pa += p.allocated; pw += p.waste; ca += c.allocated; cw += c.waste;
}
assert.deepEqual([pa, pw, ca, cw], [learn.totals.paged_allocated_tokens, learn.totals.paged_waste_tokens, learn.totals.contiguous_allocated_tokens, learn.totals.contiguous_waste_tokens]);
assert.deepEqual([pa, pw, ca, cw], [4048, 73, 5632, 1657]);

// Continuous-batching step simulator
const sim = simulateBatch(requests, learn.simulation.tokens_per_step);
assert.equal(sim.totalSteps, learn.simulation.total_steps);
assert.equal(sim.totalSteps, 79);
assert.equal(sim.throughput, learn.simulation.simulated_throughput_tokens_per_step);
assert.equal(sim.throughput, 31.708861);
assert.equal(sim.totalOutput, learn.simulation.total_output_tokens);
assert.deepEqual(sim.perRequestSteps, learn.simulation.per_request_steps);
assert.equal(sim.trace.length, sim.totalSteps);
assert.ok(sim.trace.at(-1).every((v) => v === 0));

// Budget what-if reproduces all three recorded budget points (tokens exact, correct exact here)
const rows384 = real.rows.filter((r) => r.budget === 384);
for (const b of [32, 128, 384]) {
  const w = budgetWhatIf(rows384, b);
  assert.equal(w.tokens, real.budgets[b].completion_tokens, `tokens at ${b}`);
  assert.equal(w.correct, real.budgets[b].correct, `correct at ${b}`);
}
assert.equal(budgetWhatIf(rows384, 384).tokensPerCorrect, real.budgets['384'].tokens_per_correct);

// Significance: the same hand-worked values as tests/test_significance.py
const [lo, hi] = wilson(8, 10);
assert.ok(Math.abs(lo - 0.4902) < 5e-5 && Math.abs(hi - 0.9433) < 5e-5, `wilson ${lo} ${hi}`);
assert.equal(wilson(0, 10)[0], 0);
assert.deepEqual(discordant(quant.rows.fp16, quant.rows.int4_bnb_nf4), [4, 1]);
assert.deepEqual(discordant(quant.rows.fp16, quant.rows.int8_bnb), [2, 0]);
assert.equal(mcnemarExact(4, 1), 0.375);
assert.equal(mcnemarExact(2, 0), 0.5);
assert.equal(mcnemarExact(0, 8), 2 / 256);
assert.equal(mcnemarExact(3, 3), 1);

// Quantization deltas computed in the browser match the comparison table in the file
const d = quantDeltas(quant.summaries);
for (const row of quant.comparison.filter((r) => r.config !== 'fp16')) {
  assert.equal(d[row.config].memorySavingPct, row.memory_saving_pct_vs_fp16, row.config);
  assert.equal(d[row.config].accuracyDelta, row.accuracy_delta_vs_fp16, row.config);
  assert.equal(d[row.config].throughputRatio, row.throughput_ratio_vs_fp16, row.config);
}

// The FP16 Deep rows reproduce the Build 384 rows, as REPORT.md (Deep) states
assert.deepEqual(quant.rows.fp16.map((r) => [r.question_id, r.extracted, r.completion_tokens]), rows384.map((r) => [r.question_id, r.extracted, r.completion_tokens]));

// Load-test settings come from the runner source
assert.equal(kwarg(runReal, 'concurrency'), 4);
assert.equal(kwarg(runReal, 'max_tokens'), 128);

console.log('sim ok', { kvBytesPerToken: kvBytesPerToken(cfg), kvTotal, paged: [pa, pw], contiguous: [ca, cw], steps: sim.totalSteps, throughput: sim.throughput });
