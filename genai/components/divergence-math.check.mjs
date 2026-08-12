// node components/divergence-math.check.mjs
// 驗證投影片上那三條 q 曲線的參數,與實際最小化散度解出來的一致。
import assert from 'node:assert/strict'
import { FITS, bimodal, fitGaussian, separationCurve } from './divergence-math.js'

for (const [w, table] of Object.entries(FITS)) {
  const p = bimodal(Number(w))
  for (const [which, want] of Object.entries(table)) {
    const got = fitGaussian(which, p)
    console.log(`w=${w} ${which.padEnd(7)} got`, got.mu, got.sigma, '| FITS', want.mu, want.sigma)
    // w=0.5 對稱,reverse KL 有 ±1.6 兩個等價解,只比對絕對值
    assert.ok(Math.abs(Math.abs(got.mu) - Math.abs(want.mu)) < 0.05, `${w}/${which} mu 不符`)
    assert.ok(Math.abs(got.sigma - want.sigma) < 0.05, `${w}/${which} sigma 不符`)
  }
}

// 對稱時:forward 蓋住全部,reverse 鎖單峰
assert.equal(FITS[0.5].forward.mu, 0)
assert.ok(FITS[0.5].forward.sigma > 1.4)
assert.ok(Math.abs(FITS[0.5].reverse.mu) > 1.2 && FITS[0.5].reverse.sigma < 0.9)

// 不對稱時:JSD 翻邊成 mode-seeking,forward 仍然不翻
assert.deepEqual(FITS[0.3].jsd, FITS[0.3].reverse, 'w=0.3 時 JSD 應與 reverse KL 同解')
assert.ok(Math.abs(FITS[0.3].forward.mu) < 1.2, 'forward KL 永遠不翻邊')

// JSD 有界於 log 2,分離後飽和 → 梯度消失
const c = separationCurve()
const last = c.at(-1)
assert.ok(last.jsd <= Math.LN2 + 1e-6, 'JSD 上界是 log 2')
assert.ok(Math.abs(last.jsd - c.at(-6).jsd) < 1e-3, '分離後 JSD 應飽和')
assert.ok(last.forward > 10 * last.jsd, 'KL 無上界')

console.log('separation d=6 ->', last)
console.log('ok')
