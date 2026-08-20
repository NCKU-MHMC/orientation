// node public/demos/ebm-2d-interactive.check.mjs
// 從 ebm-2d-interactive.html 抽出 CORE 那段 script、eval 後跑 assert。
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import assert from 'node:assert/strict'

const HERE = dirname(fileURLToPath(import.meta.url))
const html = readFileSync(join(HERE, 'ebm-2d-interactive.html'), 'utf8')

const m = html.match(/\/\* ===CORE:BEGIN=== \*\/([\s\S]*?)\/\* ===CORE:END=== \*\//)
assert.ok(m, 'HTML 裡找不到 CORE 區塊(===CORE:BEGIN===/===CORE:END===)')
const CORE = new Function(`${m[1]}\nreturn CORE;`)()

// ---- (a) gradE 與數值梯度一致(數點,±1e-4) ----
const H = 1e-5
for (const kind of ['near', 'far', 'ring']) {
  for (const p of [[0.3, -0.7], [-1.2, 0.4], [1.9, 1.1], [-0.05, 2.0], [0.6, 0.6]]) {
    const g = CORE.gradE(p, kind)
    const gn = [
      (CORE.energy([p[0] + H, p[1]], kind) - CORE.energy([p[0] - H, p[1]], kind)) / (2 * H),
      (CORE.energy([p[0], p[1] + H], kind) - CORE.energy([p[0], p[1] - H], kind)) / (2 * H),
    ]
    for (const d of [0, 1]) {
      assert.ok(Math.abs(g[d] - gn[d]) < 1e-4,
        `${kind} 在 (${p}) 的解析梯度應與數值梯度一致(維度 ${d}:${g[d]} vs ${gn[d]})`)
    }
  }
}

// ---- (b) 近距雙峰:2000 步 × 20 鏈後,兩峰人口比例都 > 20% ----
// 固定種子,與 demo 相同的 η=0.02、T=1、均勻初始化於 [-3,3]²
const ETA = 0.02, T = 1
const rng = CORE.mulberry32(7)
let chains = Array.from({ length: 20 }, () => [rng() * 6 - 3, rng() * 6 - 3])
for (let t = 0; t < 2000; t++) chains = chains.map(x => CORE.langevinStep(x, ETA, T, 'near', rng))
const nL = chains.filter(x => x[0] < 0).length
assert.ok(nL / 20 > 0.2 && (20 - nL) / 20 > 0.2,
  `近距雙峰 2000 步後兩峰人口都應 > 20%(左 ${nL}/20、右 ${20 - nL}/20)`)
console.log(`近距雙峰 20 鏈 × 2000 步:左峰 ${nL}/20、右峰 ${20 - nL}/20`)

// ---- (c) 遠距雙峰:同設定單鏈 2000 步跨峰次數 ≤ 近距情形的十分之一 ----
const countCross = (kind, seed) => {
  const r = CORE.mulberry32(seed)
  let x = [r() * 6 - 3, r() * 6 - 3], well = 0, cross = 0
  for (let t = 0; t < 2000; t++) {
    x = CORE.langevinStep(x, ETA, T, kind, r)
    const w = CORE.wellOf(x, kind)
    if (w !== 0) { if (well !== 0 && w !== well) cross++; well = w }
  }
  return cross
}
const cNear = countCross('near', 42), cFar = countCross('far', 42)
assert.ok(cNear >= 10, `近距雙峰單鏈 2000 步應頻繁跨峰(得到 ${cNear} 次)`)
assert.ok(cFar <= cNear / 10, `遠距雙峰跨峰次數應 ≤ 近距的十分之一(近距 ${cNear}、遠距 ${cFar})`)
console.log(`單鏈 2000 步跨峰:近距 ${cNear} 次、遠距 ${cFar} 次`)

// ---- (d) e^{−E} 網格積分為有限正值(歸一化常數存在) ----
for (const kind of ['near', 'far', 'ring']) {
  const R0 = 4.5, N = 300, d = 2 * R0 / N
  let Z = 0
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    Z += Math.exp(-CORE.energy([-R0 + (i + 0.5) * d, -R0 + (j + 0.5) * d], kind))
  }
  Z *= d * d
  assert.ok(Number.isFinite(Z) && Z > 0, `${kind} 的 ∫e^{−E} 應為有限正值(得到 ${Z})`)
  console.log(`${kind}: ∫e^{−E} ≈ ${Z.toFixed(4)}`)
}

console.log('ok')
