// 從 interface-contract.html 抽出 MODELS 區塊,在 node 跑同一份驗收檢查。
// 用法:node public/demos/interface-contract.check.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const html = readFileSync(fileURLToPath(new URL('./interface-contract.html', import.meta.url)), 'utf8');
const m = html.match(/\/\* ===== MODELS:START[\s\S]*?\*\/([\s\S]*?)\/\* ===== MODELS:END/);
if (!m) { console.error('找不到 MODELS 區塊'); process.exit(2); }

const MODELS = new Function(m[1] + '\nreturn MODELS;')();
const R = MODELS.selftest();
for (const r of R) console.log((r.ok ? 'PASS  ' : 'FAIL  ') + r.msg);
const bad = R.filter(r => !r.ok).length;
console.log(`\nselftest: ${R.length - bad}/${R.length} pass`);
process.exit(bad ? 1 : 0);
