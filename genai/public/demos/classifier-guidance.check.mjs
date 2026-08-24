// node public/demos/classifier-guidance.check.mjs
// 從 classifier-guidance.html 抽出 CORE 那段 script、eval 還原成物件,
// 驗證 classifier guidance 的數值核心:w=0 是無條件、w=1 是貝氏後驗、w>1 是外插。
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import assert from 'node:assert/strict'

const htmlPath = fileURLToPath(new URL('./classifier-guidance.html', import.meta.url))
const html = readFileSync(htmlPath, 'utf8')

const m = html.match(/\/\* CORE:BEGIN \*\/([\s\S]*?)\/\* CORE:END \*\//)
assert.ok(m, 'HTML 裡找不到 CORE:BEGIN/END 標記')
const CORE = new Function(`${m[1]}\nreturn CORE;`)()

const N = 220
const maxAbs = (a, b) => a.reduce((s, v, i) => Math.max(s, Math.abs(v - b[i])), 0)
const l1 = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0)

// (a) 機率格點合法:非負、總和為 1
for (const w of [0, 0.5, 1, 2, 4]) {
  for (let k = 0; k < CORE.CLASSES.length; k++) {
    const p = CORE.guided(k, w, 120)
    assert.ok(p.every(v => v >= 0 && Number.isFinite(v)), `[a] k=${k} w=${w} 出現負值或非有限值`)
    assert.ok(Math.abs(p.reduce((s, v) => s + v, 0) - 1) < 1e-12, `[a] k=${k} w=${w} 總和不為 1`)
  }
}
console.log('(a) OK — 各 w、各類別的引導後格點都是合法機率分布')

// (b) w=0 回到無條件分布 p(x)
{
  const u = CORE.uncond(N)
  let worst = 0
  for (let k = 0; k < CORE.CLASSES.length; k++) worst = Math.max(worst, maxAbs(CORE.guided(k, 0, N), u))
  assert.ok(worst < 1e-15, `[b] w=0 與 p(x) 不同,最大差 ${worst}`)
  console.log(`(b) OK — w=0 即無條件分布(最大差 ${worst.toExponential(1)})`)
}

// (c) w=1 恰好是貝氏後驗 p(x|c):p(x)·p(c|x) ∝ π_c·N_c(x)
{
  let worst = 0
  for (let k = 0; k < CORE.CLASSES.length; k++)
    worst = Math.max(worst, maxAbs(CORE.guided(k, 1, N), CORE.conditional(k, N)))
  assert.ok(worst < 1e-15, `[c] w=1 與 p(x|c) 不同,最大差 ${worst}`)
  console.log(`(c) OK — w=1 即貝氏後驗 p(x|c)(最大差 ${worst.toExponential(1)})`)
}

// (d) 分類器就是比值項:log p(c|x) = log p(x|c) − log p(x) + log π_c
{
  let worst = 0
  for (let k = 0; k < CORE.CLASSES.length; k++) {
    const lpi = Math.log(CORE.CLASSES[k].pi)
    for (let j = 0; j < 40; j++) for (let i = 0; i < 40; i++) {
      const x = CORE.at(i, 40), y = CORE.at(j, 40)
      const lhs = CORE.logPost(x, y, k)
      const rhs = CORE.logNormal(x, y, CORE.CLASSES[k]) - CORE.logMix(x, y) + lpi
      worst = Math.max(worst, Math.abs(lhs - rhs))
    }
  }
  assert.ok(worst < 1e-9, `[d] 比值項恆等式不成立,最大差 ${worst}`)
  console.log(`(d) OK — log p(c|x) = log p(x|c) − log p(x) + log π_c(最大差 ${worst.toExponential(1)})`)
}

// (e) 另一種寫法:p_w ∝ p(x)^(1−w) · (π_c·N_c(x))^w,與分類器寫法逐點相同
{
  let worst = 0
  for (const w of [0.3, 1, 2.5]) {
    for (let k = 0; k < CORE.CLASSES.length; k++) {
      const a = CORE.guided(k, w, 120)
      const b = CORE.normalizeGrid(120, (x, y) =>
        (1 - w) * CORE.logMix(x, y) + w * CORE.logJoint(x, y, k))
      worst = Math.max(worst, maxAbs(a, b))
    }
  }
  assert.ok(worst < 1e-14, `[e] 幾何內插寫法與分類器寫法不一致,最大差 ${worst}`)
  console.log(`(e) OK — 等價於 p^(1−w)·(π_c N_c)^w 的幾何內插(最大差 ${worst.toExponential(1)})`)
}

// (f) w 越大越銳利:格點熵單調下降、目標類別質量與 E[log p(c|x)] 單調上升
{
  const ws = [0, 0.5, 1, 1.5, 2, 3, 4]
  for (let k = 0; k < CORE.CLASSES.length; k++) {
    const s = ws.map(w => CORE.stats(k, w, 140))
    for (let i = 1; i < ws.length; i++) {
      assert.ok(s[i].entropy < s[i - 1].entropy,
        `[f] k=${k} 熵未下降:w=${ws[i - 1]}→${ws[i]}(${s[i - 1].entropy}→${s[i].entropy})`)
      assert.ok(s[i].meanLogPost > s[i - 1].meanLogPost,
        `[f] k=${k} E[log p(c|x)] 未上升:w=${ws[i - 1]}→${ws[i]}`)
      assert.ok(s[i].classMass > s[i - 1].classMass,
        `[f] k=${k} 目標類別質量未上升:w=${ws[i - 1]}→${ws[i]}`)
    }
    const s0 = s[0], s1 = s[2], s4 = s[6]
    console.log(`    類別 ${CORE.CLASSES[k].name}:熵 ${s0.entropy.toFixed(2)} → ${s1.entropy.toFixed(2)} → ${s4.entropy.toFixed(2)} bits;`
      + ` 目標類別質量 ${(s0.classMass * 100).toFixed(1)}% → ${(s1.classMass * 100).toFixed(1)}% → ${(s4.classMass * 100).toFixed(1)}%`)
  }
  console.log('(f) OK — w 增大:熵單調下降,目標類別質量與 E[log p(c|x)] 單調上升')
}

// (g) 與貝氏後驗的 L1 距離在 w=1 觸底,兩側皆大於零
{
  for (let k = 0; k < CORE.CLASSES.length; k++) {
    const d = w => CORE.stats(k, w, 140).l1ToPosterior
    assert.ok(d(1) < 1e-14, `[g] k=${k} w=1 的 L1 距離不為 0`)
    for (const w of [0, 0.5, 0.9, 1.1, 2, 4])
      assert.ok(d(w) > 1e-6, `[g] k=${k} w=${w} 的 L1 距離應大於 0`)
    assert.ok(d(0.5) > d(0.9) && d(2) > d(1.1), `[g] k=${k} L1 距離未以 w=1 為谷底`)
  }
  console.log('(g) OK — ‖p_w − p(·|c)‖₁ 在 w=1 歸零,兩側遞增')
}

// (h) 大 w 的數值穩定性:遠尾不會產生 NaN / Inf
{
  for (const w of [8, 20]) {
    const p = CORE.guided(0, w, 100)
    assert.ok(p.every(Number.isFinite), `[h] w=${w} 出現非有限值`)
    assert.ok(Math.abs(p.reduce((s, v) => s + v, 0) - 1) < 1e-12, `[h] w=${w} 總和不為 1`)
  }
  console.log('(h) OK — w 拉到 20 仍在 log 空間穩定')
}

console.log('\nALL CHECKS PASSED — public/demos/classifier-guidance.html 的 CORE 數值核心正常')
