// node public/demos/ar-2d-interactive.check.mjs
// 從 ar-2d-interactive.html 抽出 CORE 那段 script、eval 後跑 assert。
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import assert from 'node:assert/strict'

const HERE = dirname(fileURLToPath(import.meta.url))
const html = readFileSync(join(HERE, 'ar-2d-interactive.html'), 'utf8')

const m = html.match(/\/\* ===CORE:BEGIN=== \*\/([\s\S]*?)\/\* ===CORE:END=== \*\//)
assert.ok(m, 'HTML 裡找不到 CORE 區塊(===CORE:BEGIN===/===CORE:END===)')
const CORE = new Function(`${m[1]}\nreturn CORE;`)()

const data = CORE.makeData(1500, CORE.mulberry32(12345))
const model = CORE.fit(data)
const sum = a => a.reduce((x, y) => x + y, 0)

// ---- (a) 邊際與各條件直方歸一(±1e-9) ----
assert.ok(Math.abs(sum(model.p1) - 1) < 1e-9, '邊際 p(x1) 應歸一')
assert.ok(Math.abs(sum(model.p2) - 1) < 1e-9, '邊際 p(x2) 應歸一')
assert.ok(Math.abs(model.joint.reduce((s, r) => s + sum(r), 0) - 1) < 1e-9, 'joint 應歸一')
for (let k = 0; k < CORE.B; k++) {
  assert.ok(Math.abs(sum(model.c21[k]) - 1) < 1e-9, `條件 p(x2|x1∈bin ${k}) 應歸一`)
  assert.ok(Math.abs(sum(model.c12[k]) - 1) < 1e-9, `條件 p(x1|x2∈bin ${k}) 應歸一`)
}

// ---- (b) 兩種維度順序下,數個測試點的總 logprob 一致(±0.15) ----
const pts = [[0, 0], ...CORE.CENTERS, [2.5, 2.5], [-2.8, 0.5], [1.0, 1.0], [-0.7, 2.2]]
for (const p of pts) {
  const a = CORE.logprob(model, p, '12'), b = CORE.logprob(model, p, '21')
  assert.ok(Number.isFinite(a.total) && Number.isFinite(b.total), `logprob(${p}) 應為有限值`)
  assert.ok(Math.abs(a.total - b.total) < 0.15,
    `維度順序不應改變總 logprob:(${p}) 得 ${a.total} vs ${b.total}`)
  assert.ok(Math.abs(a.t1 + a.t2 - a.total) < 1e-12, '兩項之和應等於 total')
}
// 密度區讀數應高於空白區
assert.ok(CORE.logprob(model, CORE.CENTERS[0], '12').total >
  CORE.logprob(model, [2.8, 2.8], '12').total + 2, '密度區 logprob 應明顯高於空白區')

// ---- (c) 抽 2000 點,三群各自都有樣本(覆蓋性) ----
const rng = CORE.mulberry32(999)
const counts = [0, 0, 0]
for (let n = 0; n < 2000; n++) {
  let st = { step: 0, order: n % 2 ? '21' : '12' }
  st = CORE.sampleStep(model, st, rng)
  assert.equal(st.step, 1, '第一步後 state.step 應為 1')
  st = CORE.sampleStep(model, st, rng)
  assert.equal(st.step, 2, '第二步後 state.step 應為 2')
  const [x, y] = st.x
  assert.ok(x >= CORE.LO && x <= CORE.HI && y >= CORE.LO && y <= CORE.HI, '樣本應落在定義域內')
  CORE.CENTERS.forEach((c, k) => { if (Math.hypot(x - c[0], y - c[1]) < 0.8) counts[k]++ })
}
for (let k = 0; k < 3; k++)
  assert.ok(counts[k] > 50, `第 ${k} 群應有樣本覆蓋(得 ${counts[k]})`)
console.log(`三群樣本數(2000 抽): ${counts.join(' / ')}`)
console.log('ok')
