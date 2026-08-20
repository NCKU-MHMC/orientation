// node public/demos/mle-vs-ddo-gradient.check.mjs
// 從 demo HTML 抽出 CORE 那段 script,驗證兩種梯度的符號結構與訓練軌跡。
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const html = readFileSync(fileURLToPath(new URL('./mle-vs-ddo-gradient.html', import.meta.url)), 'utf8')
const m = html.match(/\/\* CORE-START[\s\S]*?\*\/([\s\S]*?)\/\* CORE-END \*\//)
assert.ok(m, 'HTML 裡找不到 CORE-START / CORE-END 區段')
const CORE = (0, eval)(m[1] + '\nCORE')

const p = CORE.pdata(), q0 = CORE.qinit()
const B = CORE.BETA, EPS = 1e-12
assert.ok(Math.abs(p.reduce((a, b) => a + b) - 1) < 1e-12, 'p_data 未正規化')
assert.ok(Math.abs(q0.reduce((a, b) => a + b) - 1) < 1e-12, 'q_init 未正規化')
assert.ok(p.every(v => v > 0), 'p_data 必須處處 > 0,否則 reverse KL 發散')
// 初始 q 必須在雙峰之間放多餘質量,否則第一幀看不到 cyan 箭頭
assert.ok(CORE.valley(q0) > 3 * CORE.valley(p), `初始 q 峰間質量不足:${CORE.valley(q0)} vs p ${CORE.valley(p)}`)

// ---- (a) MLE 梯度在所有 bin 上恆為推高,永遠不產生向下分量 ----
for (const [mode, lr, sp] of [['mle', 1, 0], ['mle', 3, 0], ['mle', 1, 50]]) {
  for (const h of CORE.run(mode, 500, lr, B, sp)) {
    assert.ok(h.c.every(v => v > 0), `MLE step ${h.step}(η=${lr})出現非正的梯度權重`)
  }
}
console.log('(a) MLE 500 步 × 3 組設定 × 64 bins:梯度權重全部 > 0,零個向下箭頭 ✓')

// ---- (b) DDO 梯度:q > p 壓低、q < p 推高 ----
let down = 0, up = 0
for (const h of CORE.run('ddo', 500, 1, B, 0)) {
  for (let i = 0; i < CORE.N; i++) {
    const d = h.q[i] - p[i]
    if (d > EPS) { assert.ok(h.c[i] < 0, `DDO step ${h.step} bin ${i}:q>p 卻沒有壓低`); down++ }
    else if (d < -EPS) { assert.ok(h.c[i] > 0, `DDO step ${h.step} bin ${i}:q<p 卻沒有推高`); up++ }
  }
}
assert.ok(down > 1000 && up > 1000, '樣本數不足,符號測試沒有真的測到')
console.log(`(b) DDO:q>p 的 ${down} 個 bin 全部壓低、q<p 的 ${up} 個 bin 全部推高 ✓`)
// 第 0 幀就要有 cyan 箭頭
const c0 = CORE.run('ddo', 0, 1, B, 0)[0].c
assert.ok(c0.filter(v => v < 0).length > 8, '第 0 步的 cyan 壓低箭頭太少')
console.log(`    第 0 步即有 ${c0.filter(v => v < 0).length} 支 cyan 箭頭 ✓`)

// ---- (c)(d) 500 步後的 KL 行為 ----
const mle = CORE.run('mle', 500, 1, B, 0), ddo = CORE.run('ddo', 500, 1, B, 0)
const f = h => [h[0].fwd, h.at(-1).fwd], r = h => [h[0].rev, h.at(-1).rev]
const [mf0, mf1] = f(mle), [mr0, mr1] = r(mle), [df0, df1] = f(ddo), [dr0, dr1] = r(ddo)
assert.ok(Math.abs(mf0 - df0) < 1e-12 && Math.abs(mr0 - dr0) < 1e-12, '兩邊起點必須相同才能比較')
console.log(`    起點 forward ${mf0.toFixed(4)} / reverse ${mr0.toFixed(4)},峰間質量 ${(CORE.valley(q0) * 100).toFixed(1)}%`)
console.log(`    500 步 MLE forward ${mf1.toFixed(4)} reverse ${mr1.toFixed(4)} 峰間 ${(CORE.valley(mle.at(-1).q) * 100).toFixed(2)}%`)
console.log(`    500 步 DDO forward ${df1.toFixed(4)} reverse ${dr1.toFixed(4)} 峰間 ${(CORE.valley(ddo.at(-1).q) * 100).toFixed(2)}%`)

// (d) 兩者的 forward KL 都下降
assert.ok(mf1 < mf0 * 0.1, `MLE forward KL 沒有明顯下降:${mf0} → ${mf1}`)
assert.ok(df1 < df0 * 0.1, `DDO forward KL 沒有明顯下降:${df0} → ${df1}`)
console.log('(d) 兩邊 forward KL 都降到起點的 10% 以下 ✓')

// (c) DDO 的 reverse KL 下降幅度大於 MLE(起點相同 ⇒ 等價於終點更低)
assert.ok(dr0 - dr1 > mr0 - mr1, `DDO reverse KL 下降幅度未超過 MLE:${dr0 - dr1} vs ${mr0 - mr1}`)
assert.ok(dr1 < mr1 * 0.6, `DDO 終點 reverse KL 未明顯低於 MLE:${dr1} vs ${mr1}`)
assert.ok(CORE.valley(ddo.at(-1).q) < CORE.valley(mle.at(-1).q), 'DDO 的峰間殘餘質量應少於 MLE')
console.log(`(c) reverse KL 下降幅度 DDO ${(dr0 - dr1).toFixed(4)} > MLE ${(mr0 - mr1).toFixed(4)};終點 ${(mr1 / dr1).toFixed(2)}× 低 ✓`)

// 穩健性:η 與 self-play 變動下結論不翻盤
for (const [lr, sp] of [[0.4, 0], [2, 0], [1, 50], [2, 50]]) {
  const a = CORE.run('mle', 500, lr, B, sp), b = CORE.run('ddo', 500, lr, B, sp)
  assert.ok(b.at(-1).rev < a.at(-1).rev, `η=${lr} selfplay=${sp}:DDO reverse KL 未低於 MLE`)
  assert.ok(a.at(-1).fwd < a[0].fwd && b.at(-1).fwd < b[0].fwd, `η=${lr} selfplay=${sp}:forward KL 沒降`)
  assert.ok(b.every(h => Number.isFinite(h.fwd) && Number.isFinite(h.rev)), `η=${lr} 出現非有限的 KL`)
}
console.log('    η∈{0.4,1,2} × self-play{off,每50步}:結論一致、數值有限 ✓')
console.log('\nAll checks passed.')
