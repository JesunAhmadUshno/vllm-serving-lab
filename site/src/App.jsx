import { useMemo, useState, useEffect, useId } from 'react';
import learn from '../../level-1-learn/results_learn.json';
import requests from '../../level-1-learn/requests.json';
import real from '../../level-2-build/results_real.json';
import realStop from '../../level-2-build/results_real_stop.json';
import quant from '../../level-3-deep/results_quant.json';
import runRealSrc from '../../level-2-build/run_real.py?raw';
import reportMd from '../../REPORT.md?raw';
import readmeMd from '../../README.md?raw';
import {
  kvBytesPerToken, pagedAllocation, contiguousAllocation, simulateBatch, budgetWhatIf,
  wilson, mcnemarExact, discordant, quantDeltas, kwarg, totalTokens, outputTokens,
} from './sim.js';

const REPO = 'https://github.com/JesunAhmadUshno/vllm-serving-lab';
const blob = (p) => `${REPO}/blob/main/${p}`;
const fmt = (n, d = 0) => n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const mib = (b) => `${fmt(b / 2 ** 20, 1)} MiB`;
const gb = (b) => `${(b / 1e9).toFixed(2)} GB`;

const MODEL = real.model_id.split('/').pop();
const GPU = real.device.replace('NVIDIA GeForce ', '');
const ROWS384 = real.rows.filter((r) => r.budget === 384);
const BUDGETS = Object.keys(real.budgets).map(Number).sort((a, b) => a - b);
const MAX_BUDGET = BUDGETS.at(-1);
const N_Q = real.budgets[MAX_BUDGET].questions;
const CONCURRENCY = kwarg(runRealSrc, 'concurrency');
const LOAD_MAX_TOKENS = kwarg(runRealSrc, 'max_tokens');
const CONFIGS = quant.comparison.map((c) => ({ ...c, short: c.config.startsWith('fp16') ? 'FP16' : c.config.startsWith('int8') ? 'INT8' : 'INT4 NF4' }));
const QIDS = quant.rows.fp16.map((r) => r.question_id);
const THINK = requests.reduce((s, r) => s + r.thinking_tokens, 0);
const OUT = requests.reduce((s, r) => s + outputTokens(r), 0);
// The v1.1 backlog is the numbered list under REPORT.md's "### v1.1 backlog" heading; the verdict is README.md's Verdict line.
const BACKLOG_MD = (reportMd.split(/^### v1\.1 backlog.*$/m)[1] || '').split(/^##+ /m)[0];
const VERDICT = (readmeMd.match(/\*\*Verdict\.\*\*\s+([^.]+)\./) || [, 'see README'])[1];
const ROADMAP = BACKLOG_MD.split('\n').filter((l) => /^\d+\.\s/.test(l)).map((l) => l.replace(/^\d+\.\s+/, '').split(':')[0]);

const SECTIONS = [['top', 'Overview', '◎'], ['kv', 'KV cache', '▤'], ['paged', 'Paging', '▦'], ['batch', 'Batching', '⇶'], ['budget', 'Budget', '◷'], ['load', 'Load test', '≋'], ['quant', 'Quantization', '◫'], ['stats', 'Significance', '±'], ['rules', 'Rules', '§'], ['limits', 'Limits', '!']];

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function MenuBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(t); }, []);
  return (
    <div className="menubar" role="banner">
      <span className="apple" aria-hidden="true">◆</span>
      <a className="app" href="#top">vLLM Serving Lab</a>
      <a className="m" href={blob('README.md')}>README</a>
      <a className="m" href={blob('REPORT.md')}>Report</a>
      <a className="m" href={blob('PROPOSAL.md')}>Proposal</a>
      <span className="spacer" />
      <span className="status"><i /> Recorded on {GPU}</span>
      <a href={REPO}>GitHub ↗</a>
      <span className="clock">{now.toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' })}</span>
    </div>
  );
}

function Dock() {
  const [active, setActive] = useState('top');
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: '-45% 0px -50% 0px' });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  return (
    <nav className="dock" aria-label="Sections">
      {SECTIONS.map(([id, label, icon]) => (
        <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} aria-current={active === id ? 'true' : undefined}>
          <span className="ico" aria-hidden="true">{icon}</span><span className="lbl">{label}</span>
        </a>
      ))}
    </nav>
  );
}

// Row charts switch to a phone-sized viewBox so their labels stay readable at 390px.
function useNarrow() {
  const q = '(max-width: 760px)';
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(q).matches);
  useEffect(() => { const m = window.matchMedia(q); const f = () => setNarrow(m.matches); m.addEventListener('change', f); return () => m.removeEventListener('change', f); }, []);
  return narrow;
}

function Range({ label, value, min, max, step, onChange, show }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label} <span>{show}</span></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

const Tag = ({ kind }) => <span className={`pill ${kind === 'simulated' ? 'info' : kind === 'recorded' ? 'good' : 'warn'}`}>● {kind}</span>;

export default function App() {
  useReveal();
  return (
    <div className="wrap">
      <MenuBar />
      <div className="glow" aria-hidden="true" />
      <Hero />
      <KvCalc />
      <Paged />
      <Batch />
      <Budget />
      <LoadTest />
      <Quant />
      <Significance />
      <Rules />
      <Limits />
      <Dock />
      <footer>
        Code: Apache-2.0. Every number on this page is read from the <code>level-*/</code> result files at build time or computed in your browser from them.
        Learn numbers are simulated; Build and Deep numbers are recorded runs on {GPU}, replayed here.
      </footer>
    </div>
  );
}

function Hero() {
  const b = real.budgets, fp16 = quant.summaries.fp16, int4 = CONFIGS.find((c) => c.short === 'INT4 NF4');
  const lost = fp16.correct - int4.correct;
  return (
    <header className="card hero window" id="top" data-title="vllm-serving-lab — README.md">
      <div className="head">
        <p className="eyebrow">Three-level serving study · {MODEL} · {GPU}</p>
        <span className="pill">Verdict: {VERDICT}</span>
        <span className="pill warn">Transformers on PyTorch behind a vLLM-shaped API, not vLLM (Windows node, no WSL)</span>
      </div>
      <h1>On a laptop GPU, a thinking budget below the answer length bought <em>zero correct answers</em>.</h1>
      <p className="sub">
        {b[BUDGETS[0]].correct} of {N_Q} correct at {BUDGETS[0]} tokens, {b[BUDGETS[1]].correct} of {N_Q} at {BUDGETS[1]}, {b[MAX_BUDGET].correct} of {N_Q} at {MAX_BUDGET}.
        Separately, 4-bit quantization cut GPU memory {int4.memory_saving_pct_vs_fp16}% and cost {lost} questions net.
        The first three windows are a serving simulator you can drive; the rest replay the recorded GPU runs, with the uncertainty that {N_Q} questions leave.
      </p>
      <div className="actions">
        <a className="btn primary" href="#budget">Try the budget what-if</a>
        <a className="btn ghost" href="#batch">Run the batching simulator</a>
        <span className="hint">Reproduce the CPU parts with <code>python level-1-learn/run_learn.py</code></span>
      </div>
      <div className="tiles">
        <div className="tile"><div className="num">{b[MAX_BUDGET].correct}/{N_Q}</div><div className="cap">correct at a {MAX_BUDGET}-token budget, against {b[BUDGETS[1]].correct} at {BUDGETS[1]}</div></div>
        <div className="tile"><div className="num">{b[MAX_BUDGET].tokens_per_correct}</div><div className="cap">completion tokens per correct answer at {MAX_BUDGET}</div></div>
        <div className="tile"><div className="num">{int4.memory_saving_pct_vs_fp16}%</div><div className="cap">less GPU memory after load with INT4 NF4 vs FP16</div></div>
        <div className="tile"><div className="num">{fp16.completion_tokens_per_second}</div><div className="cap">tokens per second, FP16, single request, recorded</div></div>
      </div>
    </header>
  );
}

function KvCalc() {
  const d = learn.model_config;
  const maxReq = requests.reduce((m, r) => (totalTokens(r) > totalTokens(m) ? r : m));
  const [layers, setLayers] = useState(d.layers);
  const [heads, setHeads] = useState(d.kv_heads);
  const [dim, setDim] = useState(d.head_dim);
  const [bytes, setBytes] = useState(d.bytes_per_element);
  const [seq, setSeq] = useState(totalTokens(maxReq));
  const per = kvBytesPerToken({ layers, kv_heads: heads, head_dim: dim, bytes_per_element: bytes });
  const setTotal = requests.reduce((s, r) => s + totalTokens(r), 0) * per;
  const isDefault = per === d.kv_bytes_per_token;
  const narrow = useNarrow();
  const W = narrow ? 340 : 440, rowH = 22, H = requests.length * rowH + 24, l = 40;
  const max = Math.max(...requests.map(totalTokens)) * per;
  return (
    <section id="kv" className="card window reveal" data-title="kv_cache.calc — Learn">
      <div className="head">
        <div>
          <p className="eyebrow">Learn · live calculator</p>
          <h2>How much GPU memory one token of context costs.</h2>
          <p className="sub">
            Every token a request holds keeps a key and a value vector in every layer: bytes per token = 2 × L × H × D × B.
            The defaults are the repo’s example config ({d.name}), not the {MODEL} model that was served, so treat the result as a teaching number.
          </p>
        </div>
        <Tag kind="simulated" />
      </div>
      <div className="controls">
        <Range label="Layers (L)" value={layers} min={1} max={128} step={1} onChange={setLayers} show={layers} />
        <Range label="KV heads (H)" value={heads} min={1} max={64} step={1} onChange={setHeads} show={heads} />
        <Range label="Head dim (D)" value={dim} min={16} max={256} step={16} onChange={setDim} show={dim} />
        <Range label="Bytes per element (B)" value={bytes} min={1} max={4} step={1} onChange={setBytes} show={bytes === 1 ? '1 (8-bit)' : bytes === 2 ? '2 (fp16)' : bytes === 4 ? '4 (fp32)' : bytes} />
        <Range label="Sequence length" value={seq} min={1} max={32768} step={1} onChange={setSeq} show={`${fmt(seq)} tokens`} />
      </div>
      <p className="mono formula">2 × {layers} × {heads} × {dim} × {bytes} = {fmt(per)} bytes per token</p>
      <div className="grid2" style={{ marginTop: 16 }}>
        <div className="tiles" style={{ gridTemplateColumns: '1fr', marginTop: 0 }}>
          <div className="tile"><div className="num">{fmt(per)}</div><div className="cap">bytes per token {isDefault ? <span className="ok">= results_learn.json</span> : `(example config: ${fmt(d.kv_bytes_per_token)})`}</div></div>
          <div className="tile"><div className="num">{mib(seq * per)}</div><div className="cap">KV cache for one {fmt(seq)}-token sequence ({fmt(seq * per)} bytes)</div></div>
          <div className="tile"><div className="num">{mib(setTotal)}</div><div className="cap">KV for all {requests.length} labelled requests ({fmt(setTotal)} bytes{isDefault ? ', matches the recorded total' : ''})</div></div>
        </div>
        <div>
          <h3>KV bytes per request at this config</h3>
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="KV cache bytes for each of the labelled requests at the chosen model config">
            {requests.map((r, i) => {
              const v = totalTokens(r) * per;
              return (
                <g key={r.request_id} transform={`translate(0 ${i * rowH})`}>
                  <text x={l - 8} y={15} textAnchor="end">{r.request_id}</text>
                  <rect x={l} y={4} height={14} rx={4} width={Math.max(1, (v / max) * (W - l - 76))} fill="var(--primary)" opacity={0.75} />
                  <text x={l + (v / max) * (W - l - 76) + 6} y={15}>{mib(v)}</text>
                </g>
              );
            })}
            <text x={l} y={H - 4}>{maxReq.request_id} ({fmt(totalTokens(maxReq))} tokens) is the largest request</text>
          </svg>
        </div>
      </div>
    </section>
  );
}

function Paged() {
  const [block, setBlock] = useState(learn.block_size_tokens);
  const [slab, setSlab] = useState(learn.contiguous_slab_tokens);
  const rows = requests.map((r) => ({ id: r.request_id, used: totalTokens(r), p: pagedAllocation(totalTokens(r), block), c: contiguousAllocation(totalTokens(r), slab) }));
  const sum = (k, f) => rows.reduce((s, r) => s + r[k][f], 0);
  const pw = sum('p', 'waste'), cw = sum('c', 'waste'), pa = sum('p', 'allocated'), ca = sum('c', 'allocated');
  const isDefault = block === learn.block_size_tokens && slab === learn.contiguous_slab_tokens;
  const narrow = useNarrow();
  const W = narrow ? 340 : 760, rowH = 26, H = rows.length * rowH + 4, l = 40;
  const max = Math.max(...rows.map((r) => Math.max(r.c.allocated, r.p.allocated)));
  const x = (v) => (v / max) * (W - l - 12);
  return (
    <section id="paged" className="card window reveal" data-title="paged_alloc.sim — Learn">
      <div className="head">
        <div>
          <p className="eyebrow">Learn · live simulation</p>
          <h2>Paged blocks waste <em>tokens</em>, not slabs.</h2>
          <p className="sub">
            A contiguous allocator reserves a whole slab per request; a PagedAttention-style allocator hands out small fixed blocks and wastes at most one partial block.
            Same {requests.length} labelled requests, same arithmetic as <a href={blob('level-1-learn/serving_sim.py')}>serving_sim.py</a>.
          </p>
        </div>
        <Tag kind="simulated" />
      </div>
      <div className="controls">
        <Range label="Paged block size" value={block} min={1} max={64} step={1} onChange={setBlock} show={`${block} tokens`} />
        <Range label="Contiguous slab size" value={slab} min={16} max={1024} step={16} onChange={setSlab} show={`${slab} tokens`} />
      </div>
      <div className="tiles" style={{ marginTop: 0, marginBottom: 16 }}>
        <div className="tile"><div className="num">{fmt(pw)}</div><div className="cap">paged waste tokens of {fmt(pa)} allocated</div></div>
        <div className="tile"><div className="num">{fmt(cw)}</div><div className="cap">contiguous waste tokens of {fmt(ca)} allocated</div></div>
        <div className="tile"><div className="num">{pct(pw / pa)}</div><div className="cap">of paged memory wasted</div></div>
        <div className="tile"><div className="num">{pct(cw / ca)}</div><div className="cap">of contiguous memory wasted</div></div>
      </div>
      <svg className="chart rows" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Per-request tokens used, paged allocation and contiguous allocation">
        {rows.map((r, i) => (
          <g key={r.id} transform={`translate(0 ${i * rowH})`}>
            <text x={l - 8} y={16} textAnchor="end">{r.id}</text>
            <rect x={l} y={3} height={8} rx={3} width={x(r.c.allocated)} fill="var(--bad)" opacity={0.22} />
            <rect x={l} y={13} height={8} rx={3} width={x(r.p.allocated)} fill="var(--good)" opacity={0.3} />
            <rect x={l} y={3} height={18} rx={3} width={x(r.used)} fill="none" stroke="var(--text)" strokeWidth={1.2} />
          </g>
        ))}
      </svg>
      <div className="legend"><span><i style={{ background: 'var(--bad)', opacity: 0.4 }} />contiguous slabs</span><span><i style={{ background: 'var(--good)', opacity: 0.5 }} />paged blocks</span><span><i style={{ border: '1.5px solid var(--text)' }} />tokens actually used</span></div>
      <p className="hint">{isDefault ? `At the recorded settings this reproduces results_learn.json: ${learn.totals.paged_waste_tokens} vs ${fmt(learn.totals.contiguous_waste_tokens)} waste tokens.` : `Recorded settings: block ${learn.block_size_tokens}, slab ${learn.contiguous_slab_tokens}.`}</p>
    </section>
  );
}

function Batch() {
  const [tps, setTps] = useState(learn.simulation.tokens_per_step);
  const [step, setStep] = useState(null);
  const [split, setSplit] = useState(true);
  const sim = useMemo(() => simulateBatch(requests, tps), [tps]);
  const s = step == null ? Math.ceil(sim.totalSteps / 2) : Math.min(step, sim.totalSteps);
  const left = sim.trace[s - 1];
  const active = left.filter((v) => v > 0).length;
  const done = requests.length - active;
  const isDefault = tps === learn.simulation.tokens_per_step;
  const narrow = useNarrow();
  const W = narrow ? 340 : 760, rowH = 24, H = requests.length * rowH + 8, l = 40, r = 64;
  const maxOut = Math.max(...requests.map(outputTokens));
  const x = (v) => (v / maxOut) * (W - l - r);
  return (
    <section id="batch" className="card window reveal" data-title="batch_scheduler.sim — Learn">
      <div className="head">
        <div>
          <p className="eyebrow">Learn · live simulation</p>
          <h2>Continuous batching, one step at a time.</h2>
          <p className="sub">
            All {requests.length} requests start at step 1. Each step has a token budget; the scheduler hands one token to each unfinished request in turn until the budget is spent, so short requests finish early and free their share.
            Throughput is <b>tokens per step, never per second</b>: this is a deterministic model, not a timed run.
          </p>
        </div>
        <Tag kind="simulated" />
      </div>
      <div className="controls">
        <Range label="Tokens per step" value={tps} min={1} max={128} step={1} onChange={(v) => { setTps(v); setStep(null); }} show={tps} />
        <Range label="Step" value={s} min={1} max={sim.totalSteps} step={1} onChange={setStep} show={`${s} of ${sim.totalSteps}`} />
        <div className="field"><span className="flabel">Colour generated tokens</span>
          <div className="seg" role="group" aria-label="Colour generated tokens">
            <button aria-pressed={split} onClick={() => setSplit(true)}>Thinking vs answer</button>
            <button aria-pressed={!split} onClick={() => setSplit(false)}>Plain</button>
          </div>
        </div>
      </div>
      <div className="tiles" style={{ marginTop: 0, marginBottom: 16 }}>
        <div className="tile"><div className="num">{sim.totalSteps}</div><div className="cap">steps to finish {fmt(sim.totalOutput)} output tokens</div></div>
        <div className="tile"><div className="num">{sim.throughput.toFixed(2)}</div><div className="cap">tokens per step (budget {tps}){isDefault ? `, recorded ${learn.simulation.simulated_throughput_tokens_per_step}` : ''}</div></div>
        <div className="tile"><div className="num">{active}</div><div className="cap">requests still decoding after step {s}; {done} finished</div></div>
        <div className="tile"><div className="num">{pct(THINK / OUT)}</div><div className="cap">of generated tokens are thinking ({fmt(THINK)} of {fmt(OUT)})</div></div>
      </div>
      <svg className="chart rows" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Progress of each request after step ${s}`}>
        {requests.map((q, i) => {
          const gen = outputTokens(q) - left[i];
          const th = split ? Math.min(gen, q.thinking_tokens) : gen;
          const fin = left[i] === 0;
          return (
            <g key={q.request_id} transform={`translate(0 ${i * rowH})`}>
              <text x={l - 8} y={15} textAnchor="end">{q.request_id}</text>
              <rect x={l} y={4} height={14} rx={4} width={x(outputTokens(q))} fill="var(--primary-pale)" />
              <rect x={l} y={4} height={14} rx={4} width={x(th)} fill={split ? 'var(--warn)' : 'var(--primary)'} opacity={0.85} />
              {split && <rect x={l + x(th)} y={4} height={14} width={x(gen - th)} fill="var(--primary)" opacity={0.85} />}
              <text x={l + x(outputTokens(q)) + 6} y={15} style={fin ? { fill: 'var(--good)' } : undefined}>{fin ? `done @${sim.perRequestSteps[q.request_id]}` : `${gen}/${outputTokens(q)}`}</text>
            </g>
          );
        })}
      </svg>
      {split && <div className="legend"><span><i style={{ background: 'var(--warn)' }} />thinking tokens</span><span><i style={{ background: 'var(--primary)' }} />answer tokens</span><span><i style={{ background: 'var(--primary-pale)' }} />still to generate</span></div>}
      <p className="hint">{isDefault ? `At ${tps} tokens per step every completion step matches results_learn.json (checked at build time).` : `Recorded run: ${learn.simulation.tokens_per_step} tokens per step, ${learn.simulation.total_steps} steps.`} The thinking/answer split uses the hand-labelled counts in requests.json; thinking is drawn first.</p>
    </section>
  );
}

function Budget() {
  const [cap, setCap] = useState(256);
  const w = budgetWhatIf(ROWS384, cap);
  const curve = useMemo(() => Array.from({ length: MAX_BUDGET }, (_, i) => [i + 1, budgetWhatIf(ROWS384, i + 1)]), []);
  const W = 460, H = 220, pad = { l: 34, r: 10, t: 12, b: 26 };
  const sx = (b) => pad.l + (b / MAX_BUDGET) * (W - pad.l - pad.r);
  const sy = (c) => pad.t + (1 - c / N_Q) * (H - pad.t - pad.b);
  const step = curve.map(([b, v], i) => `${i ? 'L' : 'M'}${sx(b)},${sy(v.correct)}`).join(' ');
  const SW = 460, SH = 220, sp = { l: 34, r: 10, t: 12, b: 26 };
  const bx = (i) => sp.l + (i + 0.5) * ((SW - sp.l - sp.r) / ROWS384.length);
  const by = (t) => sp.t + (1 - t / MAX_BUDGET) * (SH - sp.t - sp.b);
  const recorded = BUDGETS.map((b) => real.budgets[b]);
  return (
    <section id="budget" className="card window reveal" data-title="budget_study.json — Build">
      <div className="head">
        <div>
          <p className="eyebrow">Build · recorded study + counterfactual cap</p>
          <h2>The answer needs about {Math.min(...ROWS384.map((r) => r.completion_tokens))} to {Math.max(...ROWS384.filter((r) => r.completion_tokens < MAX_BUDGET).map((r) => r.completion_tokens))} tokens. Cap it lower and you get <em>nothing</em>.</h2>
          <p className="sub">
            {MODEL}, greedy decoding, the same {N_Q} questions at budgets of {BUDGETS.join(', ')} tokens. Below, drag the cap: because greedy decoding is deterministic, a run capped at B emits the same tokens as the {MAX_BUDGET}-token run up to B.
            Answers that finished within B carry over exactly; longer ones are cut at B and counted wrong, a stated lower bound (every one of the {real.rows.filter((r) => r.budget < MAX_BUDGET).length} truncated runs at {BUDGETS.slice(0, -1).join(' and ')} scored zero).
          </p>
        </div>
        <span className="pill warn">● recorded data, counterfactual cap</span>
      </div>
      <div className="controls" style={{ gridTemplateColumns: '1fr' }}>
        <Range label="What-if token cap" value={cap} min={1} max={MAX_BUDGET} step={1} onChange={setCap} show={`${cap} tokens`} />
      </div>
      <div className="tiles" style={{ marginTop: 0, marginBottom: 16 }}>
        <div className="tile"><div className="num">≥ {w.correct}/{N_Q}</div><div className="cap">correct at cap {cap} (lower bound)</div></div>
        <div className="tile"><div className="num">{fmt(w.tokens)}</div><div className="cap">completion tokens spent (exact)</div></div>
        <div className="tile"><div className="num">{w.tokensPerCorrect ? `≤ ${w.tokensPerCorrect.toFixed(1)}` : '∞'}</div><div className="cap">tokens per correct answer</div></div>
        <div className="tile"><div className="num">{w.cut}</div><div className="cap">answers cut off by the cap</div></div>
      </div>
      <div className="grid2">
        <div>
          <h3>Correct answers vs cap</h3>
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Lower-bound correct answers against token cap, with recorded budget points">
            {[0, 2, 4, 6, 8, 10].filter((t) => t <= N_Q).map((t) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} /><text x={pad.l - 6} y={sy(t) + 4} textAnchor="end">{t}</text></g>)}
            <path d={step} fill="none" stroke="var(--primary)" strokeWidth="2.2" />
            <line x1={sx(cap)} x2={sx(cap)} y1={pad.t} y2={H - pad.b} stroke="var(--warn)" strokeDasharray="4 4" />
            {BUDGETS.map((b) => <g key={b}><circle cx={sx(b)} cy={sy(real.budgets[b].correct)} r="5.5" fill="var(--text)" /><text x={sx(b)} y={H - 8} textAnchor="middle">{b}</text></g>)}
            <text x={pad.l} y={H - 8} textAnchor="start">0</text>
          </svg>
          <p className="hint">Line: counterfactual lower bound from the {MAX_BUDGET} rows. Dots: the three recorded runs.</p>
        </div>
        <div>
          <h3>Per-question tokens at {MAX_BUDGET}</h3>
          <svg className="chart" viewBox={`0 0 ${SW} ${SH}`} role="img" aria-label={`Completion tokens per question at budget ${MAX_BUDGET}, coloured by correctness, with the cap line`}>
            {[0, 128, 256, 384].filter((t) => t <= MAX_BUDGET).map((t) => <g key={t}><line className="axis" x1={sp.l} x2={SW - sp.r} y1={by(t)} y2={by(t)} /><text x={sp.l - 6} y={by(t) + 4} textAnchor="end">{t}</text></g>)}
            {ROWS384.map((r, i) => (
              <g key={r.question_id}>
                <rect x={bx(i) - 12} width={24} y={by(r.completion_tokens)} height={by(0) - by(r.completion_tokens)} rx={4} fill={r.correct ? 'var(--good)' : 'var(--bad)'} opacity={r.completion_tokens <= cap && r.completion_tokens < MAX_BUDGET ? 0.85 : 0.25} />
                <text x={bx(i)} y={SH - 8} textAnchor="middle">{r.question_id}</text>
              </g>
            ))}
            <line x1={sp.l} x2={SW - sp.r} y1={by(cap)} y2={by(cap)} stroke="var(--warn)" strokeWidth="2" strokeDasharray="5 4" />
            <text x={SW - sp.r} y={by(cap) - 6} textAnchor="end" style={{ fill: 'var(--warn)' }}>cap {cap}</text>
          </svg>
          <p className="hint">Green: correct, red: wrong. Faded bars would be cut by the cap. {ROWS384.filter((r) => r.completion_tokens >= MAX_BUDGET).map((r) => r.question_id).join(', ')} hit the {MAX_BUDGET} cap itself.</p>
        </div>
      </div>
      <div style={{ overflowX: 'auto', marginTop: 12 }}>
        <table>
          <thead><tr><th>Recorded budget</th><th>Correct</th><th>Completion tokens</th><th>Tokens per correct</th><th>Seconds per correct</th></tr></thead>
          <tbody>{BUDGETS.map((b, i) => <tr key={b}><td>{b}</td><td>{recorded[i].correct} of {recorded[i].questions}</td><td>{fmt(recorded[i].completion_tokens)}</td><td>{recorded[i].tokens_per_correct ?? '∞'}</td><td>{recorded[i].seconds_per_correct ?? '∞'}</td></tr>)}</tbody>
        </table>
      </div>
    </section>
  );
}

function LoadTest() {
  const a = real.load_test, b = realStop.load_test;
  const same = realStop.rows.every((r, i) => r.completion_tokens === real.rows[i].completion_tokens);
  const metrics = [
    ['Requests per second', 'requests_per_second', 3, true], ['Completion tokens per second', 'completion_tokens_per_second', 2, true],
    ['p50 latency (s)', 'p50_latency_seconds', 4, false], ['p95 latency (s)', 'p95_latency_seconds', 4, false], ['Wall time (s)', 'wall_seconds', 3, false],
  ];
  return (
    <section id="load" className="card window reveal" data-title="load_test.log — Build">
      <div className="head">
        <div>
          <p className="eyebrow">Build · recorded replay</p>
          <h2>{a.requests} requests, concurrency {CONCURRENCY}, with and without a stop rule.</h2>
          <p className="sub">
            Max {LOAD_MAX_TOKENS} tokens per request against the OpenAI-compatible server, which generates one request at a time, so concurrency measures queueing, not batching.
            The second run added a stop rule at a “Final answer:” line. The model rarely writes that line, so the rule {same ? `saved zero tokens: all ${real.rows.length} study runs produced identical completion counts` : 'changed some completion counts'}, and its per-step check only slowed decoding.
            Only aggregates were saved, so there is no latency histogram to show.
          </p>
        </div>
        <Tag kind="recorded" />
      </div>
      <div className="bars">
        {metrics.map(([k, f, d, higherBetter]) => {
          const max = Math.max(a[f], b[f]);
          return (
            <div key={f} className="bar-row">
              <div className="lab"><span>{k}</span><b>{fmt(a[f], d)} → {fmt(b[f], d)}</b></div>
              <div className="track"><div className="fill" style={{ width: `${(a[f] / max) * 100}%`, background: 'var(--primary)' }} /></div>
              <div className="track" style={{ marginTop: 4 }}><div className="fill" style={{ width: `${(b[f] / max) * 100}%`, background: higherBetter === (b[f] > a[f]) ? 'var(--good)' : 'var(--bad)' }} /></div>
            </div>
          );
        })}
      </div>
      <div className="legend"><span><i style={{ background: 'var(--primary)' }} />no stop rule ({a.succeeded} of {a.requests} succeeded)</span><span><i style={{ background: 'var(--bad)' }} />with stop rule ({b.succeeded} of {b.requests} succeeded), red where it is worse</span></div>
    </section>
  );
}

function Quant() {
  const [cfg, setCfg] = useState('int4_bnb_nf4');
  const [q, setQ] = useState('q06');
  const deltas = quantDeltas(quant.summaries);
  const W = 460, H = 240, pad = { l: 40, r: 20, t: 20, b: 30 };
  const mems = CONFIGS.map((c) => c.gpu_bytes_after_load);
  const lo = Math.min(...mems) * 0.85, hi = Math.max(...mems) * 1.08;
  const sx = (m) => pad.l + ((m - lo) / (hi - lo)) * (W - pad.l - pad.r);
  const sy = (a) => pad.t + (1 - a) * (H - pad.t - pad.b);
  const maxTps = Math.max(...CONFIGS.map((c) => c.completion_tokens_per_second));
  const row = quant.rows[cfg].find((r) => r.question_id === q);
  const sel = CONFIGS.find((c) => c.config === cfg);
  return (
    <section id="quant" className="card window reveal" data-title="quant_compare.json — Deep">
      <div className="head">
        <div>
          <p className="eyebrow">Deep · recorded on {GPU}</p>
          <h2>Quantization saves memory, and its losses are <em>question-shaped</em>.</h2>
          <p className="sub">
            {MODEL} at FP16, bitsandbytes INT8 and INT4 NF4, budget {quant.budget}, same {N_Q} questions, one run each.
            Memory is torch.cuda memory_allocated after load. Pick a config or click any cell to see what the model extracted.
          </p>
        </div>
        <Tag kind="recorded" />
      </div>
      <div className="seg" role="group" aria-label="Select a quantization config" style={{ marginBottom: 12 }}>
        {CONFIGS.map((c) => <button key={c.config} aria-pressed={cfg === c.config} onClick={() => setCfg(c.config)}>{c.short}</button>)}
      </div>
      <div className="grid2">
        <div>
          <h3>Memory after load vs accuracy (bubble size: tokens per second)</h3>
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Scatter of GPU memory after load against accuracy for FP16, INT8 and INT4, sized by throughput">
            {[0, 0.25, 0.5, 0.75, 1].map((t) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} /><text x={pad.l - 6} y={sy(t) + 4} textAnchor="end">{t * 100}%</text></g>)}
            {CONFIGS.map((c) => (
              <g key={c.config} onClick={() => setCfg(c.config)} style={{ cursor: 'pointer' }}>
                <circle cx={sx(c.gpu_bytes_after_load)} cy={sy(c.accuracy)} r={8 + 22 * Math.sqrt(c.completion_tokens_per_second / maxTps)} fill="var(--primary)" opacity={cfg === c.config ? 0.55 : 0.18} stroke="var(--primary)" strokeWidth={cfg === c.config ? 2 : 1} />
                <text x={sx(c.gpu_bytes_after_load)} y={sy(c.accuracy) + 4} textAnchor="middle" style={{ fill: 'var(--text)', fontWeight: 600 }}>{c.short}</text>
                <text x={sx(c.gpu_bytes_after_load)} y={sy(c.accuracy) + 38} textAnchor="middle">{gb(c.gpu_bytes_after_load)} · {c.completion_tokens_per_second} tok/s</text>
              </g>
            ))}
            <text x={W - pad.r} y={H - 6} textAnchor="end">GPU memory after load →</text>
          </svg>
        </div>
        <div>
          <h3>Per-question correctness (tokens used)</h3>
          <div style={{ overflowX: 'auto' }}>
            <table className="qgrid">
              <thead><tr><th>Q</th>{CONFIGS.map((c) => <th key={c.config}>{c.short}</th>)}</tr></thead>
              <tbody>
                {QIDS.map((id) => (
                  <tr key={id}><td className="mono">{id}</td>
                    {CONFIGS.map((c) => {
                      const r = quant.rows[c.config].find((x) => x.question_id === id);
                      const on = cfg === c.config && q === id;
                      return <td key={c.config}><button className={`cell ${r.correct ? 'yes' : 'no'}${on ? ' on' : ''}`} aria-pressed={on} aria-label={`${c.short} ${id}: ${r.correct ? 'correct' : 'wrong'}, ${r.completion_tokens} tokens`} onClick={() => { setCfg(c.config); setQ(id); }}>{r.correct ? '✓' : '✗'} {r.completion_tokens}</button></td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="tip" aria-live="polite">
        <b>{sel.short} · {row.question_id}</b>: extracted {row.extracted ?? 'nothing'} against expected {row.expected}, {row.completion_tokens} tokens, {row.correct ? 'correct' : 'wrong'}. FP16 on the same question: {(() => { const f = quant.rows.fp16.find((r) => r.question_id === q); return `${f.extracted ?? 'nothing'} in ${f.completion_tokens} tokens, ${f.correct ? 'correct' : 'wrong'}`; })()}.
      </div>
      <h3 style={{ marginTop: 20 }}>Deltas against FP16, computed in your browser from the summaries</h3>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Config</th><th>Correct</th><th>Memory saving</th><th>Accuracy delta</th><th>Throughput ratio</th><th>Tokens per correct</th></tr></thead>
          <tbody>
            {CONFIGS.map((c) => (
              <tr key={c.config}><td>{c.label}</td><td>{c.correct} of {N_Q}</td>
                <td>{deltas[c.config] ? `${deltas[c.config].memorySavingPct}%` : 'baseline'}</td>
                <td className={deltas[c.config] ? 'no' : ''}>{deltas[c.config] ? deltas[c.config].accuracyDelta.toFixed(1) : 'baseline'}</td>
                <td>{deltas[c.config] ? `${deltas[c.config].throughputRatio}×` : 'baseline'}</td>
                <td>{c.tokens_per_correct}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">INT4’s lower tokens per correct is not an efficiency win: it stops earlier and scores less. See the significance window before reading any accuracy delta as real.</p>
    </section>
  );
}

function Significance() {
  const groups = [
    ...BUDGETS.map((b) => ({ k: `Budget ${b} (Build)`, rows: real.rows.filter((r) => r.budget === b) })),
    ...CONFIGS.map((c) => ({ k: `${c.short} (Deep)`, rows: quant.rows[c.config] })),
  ].map((g) => { const n = g.rows.length, c = g.rows.filter((r) => r.correct).length; return { ...g, n, c, ci: wilson(c, n) }; });
  const pairs = [
    ['FP16', 'INT8', quant.rows.fp16, quant.rows.int8_bnb], ['FP16', 'INT4 NF4', quant.rows.fp16, quant.rows.int4_bnb_nf4],
    ['INT8', 'INT4 NF4', quant.rows.int8_bnb, quant.rows.int4_bnb_nf4],
    [`Budget ${BUDGETS[1]}`, `Budget ${MAX_BUDGET}`, real.rows.filter((r) => r.budget === BUDGETS[1]), ROWS384],
  ].map(([a, b, ra, rb]) => { const [x, y] = discordant(ra, rb); return { a, b, x, y, p: mcnemarExact(x, y) }; });
  const narrow = useNarrow();
  const W = narrow ? 340 : 760, rowH = 28, H = groups.length * rowH + 26, l = narrow ? 118 : 150, r = 16;
  const sx = (v) => l + v * (W - l - r);
  return (
    <section id="stats" className="card window reveal" data-title="significance.stats — live">
      <div className="head">
        <div>
          <p className="eyebrow">Live statistics on recorded rows</p>
          <h2>With {N_Q} questions, the quantization gaps are <em>not separable</em>.</h2>
          <p className="sub">
            Wilson 95% intervals for each accuracy, and an exact two-sided McNemar test on the paired rows (same questions, two configs: only the questions where they disagree carry information).
            Computed in your browser, and again in <a href={blob('level-3-deep/significance.py')}>significance.py</a> with unit tests in CI.
          </p>
        </div>
        <Tag kind="recorded" />
      </div>
      <svg className="chart rows" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Wilson 95 percent intervals for each recorded accuracy">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => <g key={t}><line className="axis" x1={sx(t)} x2={sx(t)} y1={0} y2={H - 20} /><text x={sx(t)} y={H - 6} textAnchor="middle">{t * 100}%</text></g>)}
        {groups.map((g, i) => (
          <g key={g.k} transform={`translate(0 ${i * rowH + 14})`}>
            <text x={l - 10} y={4} textAnchor="end" style={{ fill: 'var(--text)' }}>{g.k}</text>
            <line x1={sx(g.ci[0])} x2={sx(g.ci[1])} y1={0} y2={0} stroke="var(--primary)" strokeWidth="4" strokeLinecap="round" opacity="0.45" />
            <circle cx={sx(g.c / g.n)} cy={0} r="5" fill="var(--primary)" />
          </g>
        ))}
      </svg>
      <div style={{ overflowX: 'auto', marginTop: 12 }}>
        <table>
          <thead><tr><th>Comparison</th><th>Only A right</th><th>Only B right</th><th>Exact McNemar p</th><th>Reading</th></tr></thead>
          <tbody>
            {pairs.map((p) => (
              <tr key={p.a + p.b}><td>{p.a} vs {p.b}</td><td>{p.x}</td><td>{p.y}</td><td>{p.p.toFixed(4)}</td>
                <td><span className={`pill ${p.p < 0.05 ? 'good' : 'warn'}`}>{p.p < 0.05 ? 'Separable at 5%' : 'Not separable'}</span></td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">The budget effect survives this test; the quantization deltas do not. They are real for this question set and are not population estimates.</p>
    </section>
  );
}

function Rules() {
  const b = real.budgets, lo = BUDGETS.slice(0, -1);
  const wasted = lo.reduce((s, k) => s + b[k].completion_tokens, 0);
  const natural = ROWS384.filter((r) => r.completion_tokens < MAX_BUDGET).length;
  const wrong = ROWS384.filter((r) => !r.correct);
  const rules = [
    ['A budget pays only above the natural answer length.', `${lo.map((k) => `${b[k].correct} of ${N_Q} at ${k}`).join(', ')}: ${fmt(wasted)} tokens for no correct answers.`, '#budget'],
    ['A budget is a cap, not a target.', `At ${MAX_BUDGET}, ${natural} of ${N_Q} answers stopped on their own; the run used ${fmt(b[MAX_BUDGET].completion_tokens)} tokens, not ${fmt(N_Q * MAX_BUDGET)}.`, '#budget'],
    ['More thinking does not fix a wrong method.', `The ${wrong.length} misses at ${MAX_BUDGET} (${wrong.map((r) => `${r.question_id}: ${r.extracted} vs ${r.expected}`).join('; ')}) are method errors, not truncation.`, '#quant'],
    ['Watch cost per correct answer, not per request.', `${b[MAX_BUDGET].tokens_per_correct} tokens and ${quant.summaries.fp16.seconds_per_correct} to ${b[MAX_BUDGET].seconds_per_correct} s per correct answer at ${MAX_BUDGET} across the two FP16 runs. A setting that scores zero is infinitely expensive.`, '#budget'],
    ['Stop rules save only when they match the real output.', `The unmatched stop rule left tokens per correct at ${realStop.budgets[MAX_BUDGET].tokens_per_correct} and cut load-test throughput from ${real.load_test.completion_tokens_per_second} to ${realStop.load_test.completion_tokens_per_second} tok/s.`, '#load'],
  ];
  return (
    <section id="rules" className="card window reveal" data-title="thinking_rule.md">
      <p className="eyebrow">The thinking rule</p>
      <h2>Five rules, each tied to the number it rests on.</h2>
      <ol className="steps" style={{ marginTop: 16 }}>
        {rules.map(([t, d, href], i) => <li key={t}><span className="n">{i + 1}</span><div><b>{t}</b><div className="hint">{d} <a href={href}>See the data</a></div></div></li>)}
      </ol>
      <p className="hint" style={{ marginTop: 12 }}>Rules distilled from <a href={blob('REPORT.md')}>REPORT.md</a> (Deep, the thinking rule). They are drawn from one model and {N_Q} questions.</p>
    </section>
  );
}

function Limits() {
  const limits = [
    ['No vLLM anywhere', 'Every recorded number is Transformers on PyTorch with sequential generation. No vLLM or SGLang number exists yet.'],
    ['Small sample, single run', `${N_Q} questions and one run per config. Accuracy deltas rest on a few questions, and throughput has no variance bounds.`],
    ['No real batching', 'The server decodes one request at a time, so concurrency results are a floor. The batching window is a simulation, not a measurement.'],
    ['Simulator not validated', `requests.json is a hand-labelled teaching set and the KV example config is not ${MODEL}.`],
    ['Thin quantization coverage', 'bitsandbytes on-the-fly only: no AWQ, GPTQ or KV-cache quantization; no speculative decoding.'],
    ['Scoring heuristic', 'Answers are extracted by rule (final-answer marker, else last number) with no human-checked error rate.'],
  ];
  const docs = [
    ['README.md', 'README', 'Question, headline, verdict'], ['PROPOSAL.md', 'Proposal', 'Scope and specs (not locked)'],
    ['REPORT.md', 'Report', 'Method, results, limits'], ['CHANGELOG.md', 'Changelog', 'What changed, when'],
    ['docs/decisions/0001-keep-level-folders.md', 'Decisions', 'Hard-to-reverse choices'], ['releases', 'Releases', 'Tagged versions'],
  ];
  return (
    <section id="limits" className="card window reveal" data-title="limits_and_roadmap.md">
      <p className="eyebrow">Limits and threats to validity</p>
      <h2>What this lab does <em>not</em> show yet.</h2>
      <div className="grid2" style={{ marginTop: 16 }}>
        {limits.map(([t, d]) => <div className="tile" key={t}><div className="head" style={{ marginBottom: 6 }}><h3>{t}</h3><span className="pill warn">Limit</span></div><p className="hint" style={{ margin: 0, fontSize: 13 }}>{d}</p></div>)}
      </div>
      <h3 style={{ marginTop: 24 }}>v1.1 backlog (from REPORT.md, none of it built)</h3>
      <ol className="backlog">{ROADMAP.map((t) => <li key={t}>{t}</li>)}</ol>
      <div className="docs" style={{ marginTop: 20 }}>
        {docs.map(([p, t, d]) => <a key={p} href={p === 'releases' ? `${REPO}/releases` : blob(p)}><b>{t}</b><small>{d}</small></a>)}
      </div>
    </section>
  );
}
