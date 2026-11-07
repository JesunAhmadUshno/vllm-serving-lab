// Everything the page computes in the browser. The serving functions are line-for-line
// ports of level-1-learn/serving_sim.py; the statistics mirror level-3-deep/significance.py.
// sim.check.mjs runs before every build and fails it if any port drifts from the result files.

export const totalTokens = (r) => r.prompt_tokens + r.thinking_tokens + r.answer_tokens;
export const outputTokens = (r) => r.thinking_tokens + r.answer_tokens;

// kv_bytes_per_token: 2 (K and V) * layers * kv_heads * head_dim * bytes_per_element
export const kvBytesPerToken = ({ layers, kv_heads, head_dim, bytes_per_element }) => 2 * layers * kv_heads * head_dim * bytes_per_element;
export const kvBytesForRequest = (r, cfg) => totalTokens(r) * kvBytesPerToken(cfg);

// paged_allocation: whole blocks of blockSize tokens
export function pagedAllocation(total, blockSize) {
  if (total <= 0) return { blocks: 0, allocated: 0, waste: 0 };
  const blocks = Math.ceil(total / blockSize);
  return { blocks, allocated: blocks * blockSize, waste: blocks * blockSize - total };
}

// contiguous_allocation: round up to the next whole slab
export function contiguousAllocation(total, slab) {
  if (total <= 0) return { slabs: 0, allocated: 0, waste: 0 };
  const slabs = Math.ceil(total / slab);
  return { slabs, allocated: slabs * slab, waste: slabs * slab - total };
}

// simulate_batch: round-robin passes inside each step until the tokens-per-step budget is
// spent. `trace[s][i]` is how many output tokens request i still needs after step s+1.
export function simulateBatch(records, tokensPerStep) {
  const remaining = records.map(outputTokens);
  const completion = records.map((r) => (outputTokens(r) === 0 ? 0 : null));
  const trace = [];
  let step = 0;
  while (remaining.some((v) => v > 0)) {
    step += 1;
    let budget = tokensPerStep;
    while (budget > 0 && remaining.some((v) => v > 0)) {
      for (let i = 0; i < records.length; i++) {
        if (budget <= 0) break;
        if (remaining[i] > 0) {
          remaining[i] -= 1; budget -= 1;
          if (remaining[i] === 0) completion[i] = step;
        }
      }
    }
    trace.push([...remaining]);
  }
  const totalOutput = records.reduce((s, r) => s + outputTokens(r), 0);
  return {
    totalSteps: step,
    totalOutput,
    throughput: step ? Math.round((totalOutput / step) * 1e6) / 1e6 : 0,
    perRequestSteps: Object.fromEntries(records.map((r, i) => [r.request_id, completion[i]])),
    trace,
  };
}

// Budget what-if over the recorded 384-token rows. Greedy decoding is deterministic, so a
// run capped at B produces the same tokens as the 384 run up to B. Answers that finished
// within B carry over exactly; longer ones are cut at exactly B tokens and counted wrong
// (a lower bound on correct, exact on tokens).
export function budgetWhatIf(rows384, cap) {
  let correct = 0, tokens = 0, cut = 0;
  for (const r of rows384) {
    const fits = r.completion_tokens <= cap && r.completion_tokens < 384;
    tokens += Math.min(r.completion_tokens, cap);
    if (!fits) cut += 1;
    if (fits && r.correct) correct += 1;
  }
  return { correct, tokens, cut, tokensPerCorrect: correct ? tokens / correct : null };
}

// Wilson score interval, same z as significance.py.
export function wilson(correct, n, z = 1.959964) {
  const p = correct / n, den = 1 + (z * z) / n, c = (p + (z * z) / (2 * n)) / den;
  const h = (z / den) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

const comb = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return r; };

// Exact two-sided McNemar: 2 * P(X <= min(b, c)), X ~ Binomial(b + c, 0.5), capped at 1.
export function mcnemarExact(b, c) {
  const n = b + c;
  if (n === 0) return 1;
  let tail = 0;
  for (let k = 0; k <= Math.min(b, c); k++) tail += comb(n, k);
  return Math.min(1, (2 * tail) / 2 ** n);
}

export function discordant(rowsA, rowsB) {
  const b = Object.fromEntries(rowsB.map((r) => [r.question_id, r.correct]));
  let onlyA = 0, onlyB = 0;
  for (const r of rowsA) {
    if (!(r.question_id in b)) continue;
    if (r.correct && !b[r.question_id]) onlyA++;
    if (!r.correct && b[r.question_id]) onlyB++;
  }
  return [onlyA, onlyB];
}

// Deltas against FP16, as level-3-deep/quant_compare.py reports them.
export function quantDeltas(summaries, base = 'fp16') {
  const b = summaries[base];
  return Object.fromEntries(Object.entries(summaries).filter(([k]) => k !== base).map(([k, s]) => [k, {
    memorySavingPct: Math.round((1 - s.gpu_bytes_after_load / b.gpu_bytes_after_load) * 1e4) / 100,
    accuracyDelta: Math.round((s.accuracy - b.accuracy) * 1e3) / 1e3,
    throughputRatio: Math.round((s.completion_tokens_per_second / b.completion_tokens_per_second) * 1e3) / 1e3,
  }]));
}

// Pull a keyword argument out of the committed runner source, so a config value shown on
// the page comes from the file that ran, not from prose.
export function kwarg(src, name) {
  const m = src.match(new RegExp(`${name}=(\\d+)`));
  if (!m) throw new Error(`${name} not found in runner source`);
  return +m[1];
}
