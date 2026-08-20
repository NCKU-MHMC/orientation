// Self-check for guidance-playground.html — extracts the CORE object via regex,
// evals it, and asserts the numeric core behaves per the unified guidance formula.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const htmlPath = fileURLToPath(new URL('./guidance-playground.html', import.meta.url));
const html = readFileSync(htmlPath, 'utf8');

const m = html.match(/\/\* CORE:BEGIN \*\/([\s\S]*?)\/\* CORE:END \*\//);
assert.ok(m, 'CORE:BEGIN/END block not found in HTML');
let block = m[1].trim();
block = block.replace(/^const CORE\s*=\s*/, '').replace(/;\s*$/, '');
const CORE = eval(block);

const approxEq = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

// (a) guided probabilities are always a legal distribution: non-negative, sums to 1.
for (const key of Object.keys(CORE.presets)) {
  const p = CORE.presets[key];
  for (const w of [-3, -1.5, 0, 1.5, 3]) {
    const guided = CORE.guidedLogits(p.base, p.A, p.B, w);
    const probs = CORE.softmax(guided);
    const sum = probs.reduce((s, x) => s + x, 0);
    assert.ok(probs.every(x => x >= 0), `[a] negative prob in preset ${key} at w=${w}`);
    assert.ok(approxEq(sum, 1), `[a] probs don't sum to 1 in preset ${key} at w=${w} (sum=${sum})`);
  }
}
console.log('(a) OK — guided probabilities are legal distributions for all presets/w');

// (b) temperature preset: entropy decreases monotonically as w increases.
{
  const p = CORE.presets.temperature;
  const ws = [-3, -2, -1, 0, 1, 2, 3];
  const entropies = ws.map(w => CORE.entropy(CORE.softmax(CORE.guidedLogits(p.base, p.A, p.B, w))));
  for (let i = 1; i < entropies.length; i++) {
    assert.ok(entropies[i] < entropies[i - 1],
      `[b] entropy not strictly decreasing at w=${ws[i]}: ${entropies[i - 1]} -> ${entropies[i]}`);
  }
}
console.log('(b) OK — temperature entropy strictly decreases as w increases');

// (c) log-space: guided logits are strictly linear in w (constant first difference).
{
  const p = CORE.presets.cfg;
  const g = w => CORE.guidedLogits(p.base, p.A, p.B, w);
  const g0 = g(0), g1 = g(1), g2 = g(2);
  for (let i = 0; i < g0.length; i++) {
    const d1 = g1[i] - g0[i], d2 = g2[i] - g1[i];
    assert.ok(approxEq(d1, d2), `[c] non-linear at token ${i}: d1=${d1} d2=${d2}`);
  }
}
console.log('(c) OK — guided logits are strictly linear in w (equal steps for equal Δw)');

// (d) prompt engineering: changing w does not change the output distribution.
{
  const p = CORE.presets.promptEng;
  const probsLo = CORE.softmax(CORE.guidedLogits(p.base, p.A, p.B, -3));
  const probsHi = CORE.softmax(CORE.guidedLogits(p.base, p.A, p.B, 3));
  for (let i = 0; i < probsLo.length; i++) {
    assert.ok(approxEq(probsLo[i], probsHi[i]),
      `[d] prompt-engineering distribution changed with w at token ${i}`);
  }
}
console.log('(d) OK — prompt engineering distribution is invariant to w');

console.log('ALL CHECKS PASSED');
