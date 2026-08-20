// node public/demos/asr-noisy-channel.check.mjs
// 從 asr-noisy-channel.html 抽出 CORE 那段 script、eval 還原成物件,
// 驗證噪聲通道 n-best 重排名的數值核心沒有壞掉。
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import assert from 'node:assert/strict'

const HERE = dirname(fileURLToPath(import.meta.url))
const html = readFileSync(join(HERE, 'asr-noisy-channel.html'), 'utf8')

const m = html.match(/\/\/ === CORE:START ===([\s\S]*?)\/\/ === CORE:END ===/)
assert.ok(m, 'HTML 裡找不到 CORE:START/CORE:END 標記')
const CORE = new Function(`${m[1]}\nreturn CORE;`)()

const byId = id => CORE.nbest.find(h => h.id === id)
const correct = CORE.nbest.find(h => h.tag === 'correct')
const acousticBest = CORE.nbest.find(h => h.tag === 'acoustic-best')
const pathological = CORE.nbest.find(h => h.tag === 'pathological')
assert.ok(correct && acousticBest && pathological, 'nbest 缺少 correct / acoustic-best / pathological 三個標記之一')

// ---------------------------------------------------------------
// (a) 總分 = 聲學分數 + λ·語言分數 + γ·詞數,且全在對數空間(相加,不是相乘)
// ---------------------------------------------------------------
for (const h of CORE.nbest) {
  assert.ok(h.ac < 0 && h.lm < 0, `${h.id}: 聲學/LM 分數應為對數機率(<0),實際 ac=${h.ac} lm=${h.lm}`)
}
const samplePoints = [
  [0, 0], [0.3, 0], [1, 0.5], [2.5, -1.2], [1.7, 2],
]
for (const h of CORE.nbest) {
  for (const [lambda, gamma] of samplePoints) {
    const want = h.ac + lambda * h.lm + gamma * h.len
    const got = CORE.score(h, lambda, gamma)
    assert.ok(Math.abs(got - want) < 1e-9, `${h.id} λ=${lambda} γ=${gamma}: score() 沒有算出 ac+λ·lm+γ·len`)
    // 加法組合,不是乘法:乘法版本在這些非平凡係數下應該明顯不同
    const mulVersion = h.ac * (1 + lambda) * h.lm * (1 + gamma)
    assert.notEqual(got.toFixed(6), mulVersion.toFixed(6), `${h.id}: score() 疑似退化成乘法組合`)
  }
}
console.log('(a) OK — 總分是對數空間的線性組合 ac + λ·lm + γ·len')

// ---------------------------------------------------------------
// (b) λ=0 時排序完全由聲學分數決定
// ---------------------------------------------------------------
const byAcousticOnly = [...CORE.nbest].sort((a, b) => b.ac - a.ac).map(h => h.id)
const rankedLambda0 = CORE.rank(0, 0).map(h => h.id)
assert.deepEqual(rankedLambda0, byAcousticOnly, 'λ=0 的排序應與純聲學分數排序完全一致')
assert.equal(rankedLambda0[0], acousticBest.id, 'λ=0 首位應是聲學分數最高的候選句')
console.log('(b) OK — λ=0 時排序 = 純聲學分數排序,首位是', acousticBest.id, acousticBest.text)

// ---------------------------------------------------------------
// (c) 存在一個 λ 門檻,跨過後 top-1 從「聲學最像但語言不合理」換成「語言合理」
//     門檻由 CORE 自身的資料算出(不寫死常數),再用 rank() 驗證跨越前後的行為
// ---------------------------------------------------------------
const gamma0 = 0
// 令 score(acousticBest, λ) = score(correct, λ) 解出 λ*
const lambdaStar = (acousticBest.ac - correct.ac) / (correct.lm - acousticBest.lm)
assert.ok(lambdaStar > 0 && Number.isFinite(lambdaStar), `算出的 λ* 不合理: ${lambdaStar}`)

const eps = 1e-4
const below = CORE.rank(lambdaStar - eps, gamma0)[0]
const above = CORE.rank(lambdaStar + eps, gamma0)[0]
assert.equal(below.id, acousticBest.id, `λ* 之前 top-1 應仍是 ${acousticBest.id}(聲學最像),實際是 ${below.id}`)
assert.equal(above.id, correct.id, `λ* 之後 top-1 應換成 ${correct.id}(語意通順),實際是 ${above.id}`)
console.log(`(c) OK — LM weight 門檻 λ* = ${lambdaStar.toFixed(6)}: 跨過後 top-1 由「${acousticBest.text}」(同音錯字)換成「${correct.text}」(語意通順)`)

// 額外驗證:λ 由 0 拉到更高,途中還會出現第二次「病理性」翻轉(top-1 換成 pathological)
// 呼應計劃書驗收標準「至少兩次翻轉,其中一次為病理性翻轉」
let sawPathologicalFlip = false
let lambdaAtPathologicalFlip = null
for (let lambda = 0; lambda <= 5; lambda += 0.001) {
  if (CORE.rank(lambda, gamma0)[0].id === pathological.id) {
    sawPathologicalFlip = true
    lambdaAtPathologicalFlip = lambda
    break
  }
}
assert.ok(sawPathologicalFlip, 'λ 拉到 5 以內都沒有出現病理性翻轉(top-1 換成 pathological 候選句)')
console.log(`    附加檢查 OK — λ ≈ ${lambdaAtPathologicalFlip.toFixed(3)} 時 top-1 變成病理候選句「${pathological.text}」`)

// ---------------------------------------------------------------
// (d) insertion penalty 變動會系統性地偏好較短或較長的假說
// ---------------------------------------------------------------
const lens = CORE.nbest.map(h => h.len)
const minLen = Math.min(...lens), maxLen = Math.max(...lens)
const GAMMA_EXTREME = 3 // 與頁面滑桿上限一致

const shortWinsTop = CORE.rank(0, -GAMMA_EXTREME)[0]
assert.equal(shortWinsTop.len, minLen, `γ=-${GAMMA_EXTREME}(強懲罰長句)時 top-1 詞數應是最短的 ${minLen},實際是 ${shortWinsTop.len}(${shortWinsTop.id})`)

const longWinsTop = CORE.rank(0, GAMMA_EXTREME)[0]
assert.equal(longWinsTop.len, maxLen, `γ=+${GAMMA_EXTREME}(強獎勵長句)時 top-1 詞數應是最長的 ${maxLen},實際是 ${longWinsTop.len}(${longWinsTop.id})`)

// 更一般地:對任兩個詞數不同的候選句,γ 越大,長句對短句的分數差應嚴格遞增
const a = byId('H1'), b = byId('H7') // len 8 vs len 3
assert.ok(a.len !== b.len)
const diffLowGamma = CORE.score(a, 0, -1) - CORE.score(b, 0, -1)
const diffHighGamma = CORE.score(a, 0, 1) - CORE.score(b, 0, 1)
assert.ok(diffHighGamma > diffLowGamma, 'γ 增加時,長句相對短句的分數差應該系統性變大')
console.log(`(d) OK — γ=-${GAMMA_EXTREME} 時最短句(len=${minLen})勝出,γ=+${GAMMA_EXTREME} 時最長句(len=${maxLen})勝出;懲罰方向正確`)

console.log('\nALL CHECKS PASSED — public/demos/asr-noisy-channel.html 的 CORE 數值核心正常')
