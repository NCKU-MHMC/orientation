// node public/demos/normalizing-flow-2d-interactive.check.mjs
// 從 normalizing-flow-2d-interactive.html 抽出 CORE 區塊、eval 後跑驗收。
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import assert from 'node:assert/strict'

const HERE = dirname(fileURLToPath(import.meta.url))
const html = readFileSync(join(HERE, 'normalizing-flow-2d-interactive.html'), 'utf8')

const m = html.match(/\/\* ===CORE:BEGIN=== \*\/([\s\S]*?)\/\* ===CORE:END=== \*\//)
assert.ok(m, 'HTML 裡找不到 CORE 區塊(===CORE:BEGIN===/===CORE:END===)')
const CORE = new Function(`${m[1]}\nreturn CORE;`)()

// 固定種子亂數,驗收可重跑
let seed = 12345
const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 }

// ---- (a) round-trip x → z → x 最大誤差 < 1e-6(200 隨機點) ----
let worst = 0
for (let i = 0; i < 200; i++) {
  const x = [rand() * 6 - 3, rand() * 6 - 3]
  const { z } = CORE.inverse(x)
  const { x: xr } = CORE.forward(z)
  worst = Math.max(worst, Math.abs(xr[0] - x[0]), Math.abs(xr[1] - x[1]))
}
assert.ok(worst < 1e-6, `round-trip 最大誤差 ${worst.toExponential(2)} 應 < 1e-6`)
console.log(`round-trip 最大誤差(200 點): ${worst.toExponential(2)}`)

// z → x → z 同樣要閉合
worst = 0
for (let i = 0; i < 200; i++) {
  const z = [rand() * 4 - 2, rand() * 4 - 2]
  const { x } = CORE.forward(z)
  const { z: zr } = CORE.inverse(x)
  worst = Math.max(worst, Math.abs(zr[0] - z[0]), Math.abs(zr[1] - z[1]))
}
assert.ok(worst < 1e-6, `z→x→z 最大誤差 ${worst.toExponential(2)} 應 < 1e-6`)

// ---- (b) logdet 與數值 Jacobian 行列式一致(±1e-3) ----
const numLogdet = x => {
  const h = 1e-5
  const zc = d => CORE.inverse(d).z
  const zx1 = zc([x[0] + h, x[1]]), zx0 = zc([x[0] - h, x[1]])
  const zy1 = zc([x[0], x[1] + h]), zy0 = zc([x[0], x[1] - h])
  const j00 = (zx1[0] - zx0[0]) / (2 * h), j01 = (zy1[0] - zy0[0]) / (2 * h)
  const j10 = (zx1[1] - zx0[1]) / (2 * h), j11 = (zy1[1] - zy0[1]) / (2 * h)
  return Math.log(Math.abs(j00 * j11 - j01 * j10))
}
for (const x of [[0, 0], [1.2, 0.5], [-1.5, 0.8], [0.4, -1.1], [2.0, -0.3], [-0.7, -0.6]]) {
  const a = CORE.logdet(x), b = numLogdet(x)
  assert.ok(Math.abs(a - b) < 1e-3, `logdet(${x}) 解析 ${a.toFixed(6)} vs 數值 ${b.toFixed(6)} 應 ±1e-3`)
}
console.log('logdet 與數值 Jacobian 一致(6 點,±1e-3)')

// ---- (c) 密度網格積分 ≈ 1(±3e-2) ----
{
  const R = 6, N = 400, d = 2 * R / N
  let s = 0
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++)
    s += Math.exp(CORE.logprob([-R + (i + 0.5) * d, -R + (j + 0.5) * d]))
  const I = s * d * d
  assert.ok(Math.abs(I - 1) < 3e-2, `密度積分 ${I.toFixed(4)} 應在 1 ± 3e-2`)
  console.log(`密度網格積分([-6,6]²): ${I.toFixed(4)}`)
}

// ---- (d) 兩項分解自洽:base + logdet = total,且與 logprob 相等 ----
for (let i = 0; i < 50; i++) {
  const x = [rand() * 6 - 3, rand() * 6 - 3]
  const p = CORE.logprobParts(x)
  assert.ok(Math.abs(p.base + p.logdet - p.total) < 1e-12, 'logprobParts 兩項相加應等於總和')
  assert.equal(p.total, CORE.logprob(x), 'logprob 應等於 logprobParts().total')
}

// ---- (e) 層深插值:t=0 為恆等、t=L 等於完整 forward、對 t 連續 ----
for (let i = 0; i < 50; i++) {
  const z = [rand() * 4 - 2, rand() * 4 - 2]
  const p0 = CORE.partial(z, 0)
  assert.ok(Math.abs(p0[0] - z[0]) < 1e-12 && Math.abs(p0[1] - z[1]) < 1e-12, 'partial(z,0) 應為恆等')
  const pL = CORE.partial(z, CORE.L), xf = CORE.forward(z).x
  assert.ok(Math.abs(pL[0] - xf[0]) < 1e-12 && Math.abs(pL[1] - xf[1]) < 1e-12, 'partial(z,L) 應等於 forward')
  for (let k = 1; k < CORE.L; k++) { // 整數層深兩側極限一致(不撕裂的必要條件)
    const lo = CORE.partial(z, k - 1e-9), hi = CORE.partial(z, k + 1e-9)
    assert.ok(Math.hypot(lo[0] - hi[0], lo[1] - hi[1]) < 1e-6, `partial 在 t=${k} 應連續`)
  }
}
console.log('層深插值:t=0 恆等、t=L=forward、整數層深處連續')

// ---- (f) 擬合品質:雙月牙上的平均 logprob 應遠高於背景 ----
{
  let s2 = 20260816
  const r2 = () => { s2 = (s2 * 1664525 + 1013904223) >>> 0; return s2 / 4294967296 }
  const rn = () => { const u = r2() || 1e-12, v = r2(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) }
  let onMoons = 0, bg = 0
  const NPT = 300
  for (let i = 0; i < NPT; i++) {
    const t = r2() * Math.PI
    let mx, my
    if (i % 2 === 0) { mx = Math.cos(t); my = Math.sin(t) } else { mx = 1 - Math.cos(t); my = 0.5 - Math.sin(t) }
    onMoons += CORE.logprob([(mx - 0.5) * 1.7 + rn() * 0.07, (my - 0.25) * 1.7 + rn() * 0.07])
    bg += CORE.logprob([r2() * 6 - 3, r2() * 6 - 3])
  }
  onMoons /= NPT; bg /= NPT
  assert.ok(onMoons > -2.2, `雙月牙上的平均 logprob ${onMoons.toFixed(3)} 應 > -2.2`)
  assert.ok(onMoons - bg > 3, `雙月牙 (${onMoons.toFixed(2)}) 與均勻背景 (${bg.toFixed(2)}) 的平均 logprob 差應 > 3`)
  console.log(`擬合品質:平均 logprob 雙月牙 ${onMoons.toFixed(3)} vs 均勻背景 ${bg.toFixed(3)}`)
}

console.log('ok')
