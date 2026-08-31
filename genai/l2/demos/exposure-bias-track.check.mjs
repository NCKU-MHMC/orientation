// node public/demos/exposure-bias-track.check.mjs
// 從 exposure-bias-track.html 抽出 CORE 那段 script、eval 後跑 assert。
// CORE 只含真實系統、歷史視窗與 rollout(不含 TensorFlow.js 的訓練),因此可以在 node 裡直接檢查。
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import assert from 'node:assert/strict'

const HERE = dirname(fileURLToPath(import.meta.url))
const html = readFileSync(join(HERE, 'exposure-bias-track.html'), 'utf8')

const m = html.match(/\/\* ===CORE:BEGIN=== \*\/([\s\S]*?)\/\* ===CORE:END=== \*\//)
assert.ok(m, 'HTML 裡找不到 CORE 區塊(===CORE:BEGIN===/===CORE:END===)')
const CORE = new Function(`${m[1]}\nreturn CORE;`)()

const mulberry32 = seed => () => {
  seed |= 0; seed = seed + 0x6D2B79F5 | 0
  let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
  return ((t ^ t >>> 14) >>> 0) / 4294967296
}
const N = 1, STEPS = 300
const mean = a => Array.from(a).reduce((x, y) => x + y, 0) / a.length

const data = CORE.makeDataset(20000, mulberry32(9))

// ---- (a) 真實軌跡繞著曲線擺動,而且不會跑掉 ----
const devData = []
for (let t = 0; t < data.n; t++) devData.push(CORE.distToCurve(data.pts[2 * t], data.pts[2 * t + 1]))
const devMean = mean(devData)
assert.ok(devMean > 0.05 && devMean < 0.6,
  `真實軌跡應在曲線兩側擺動而非貼死或發散(實測平均離曲線 ${devMean.toFixed(3)})`)
assert.ok(Math.max(...devData) < 1.2, '真實軌跡不應離開畫布範圍')
assert.ok(data.stepLen > 0.05 && data.stepLen < 0.2, `每步位移應在合理範圍(${data.stepLen.toFixed(3)})`)

// ---- (b) 這個系統是非 Markov 的 —— k 滑桿的全部意義 ----
// 只給位置時仍有殘差(隱藏的速度);給到兩步就只剩過程雜訊。
const floorK1 = CORE.residualGivenPosition(data)
const floorK2 = 2 * CORE.XI * CORE.XI
assert.ok(floorK1 > 20 * floorK2,
  `只看位置的殘差 ${floorK1.toExponential(2)} 應遠大於過程雜訊 ${floorK2.toExponential(2)},否則 k 就沒有意義`)

// k=2 之後歷史不再帶新資訊:換掉更舊的幀,條件期望完全不變
{
  const P = 4, a = new Float32Array(P * 2), b = new Float32Array(P * 2)
  const tail = [1.3, 0.9, 1.38, 0.96]              // 最後兩幀相同
  for (let j = 0; j < P - 2; j++) {
    a[j * 2] = 0.2 * j;      a[j * 2 + 1] = -0.7 * j
    b[j * 2] = -1.9 + 0.5 * j; b[j * 2 + 1] = 2.4 * j
  }
  a.set(tail, (P - 2) * 2); b.set(tail, (P - 2) * 2)
  assert.deepEqual(Array.from(CORE.trueDeltaOfHist(a, P, 1)), Array.from(CORE.trueDeltaOfHist(b, P, 1)),
    '真實動力學是二階的:最後兩幀相同時,更舊的歷史不應改變預測')
}
// featurize:長度固定 KMAX 的序列,最近 k 幀靠右對齊,LSTM 只跑最後 k 格
{
  const P = CORE.KMAX, D = CORE.DSCALE, S = CORE.SDIM
  const hist = new Float32Array(P * 2)
  for (let j = 0; j < P; j++) { hist[2 * j] = j; hist[2 * j + 1] = 2 * j }   // 每步位移固定 (1,2)
  const slot = (f, j) => Array.from(f.slice(j * S, (j + 1) * S))             // 單筆:第 j 個時刻

  const f1 = CORE.featurize(hist, P, 1, 1)
  assert.equal(f1.length, CORE.KMAX * S, '序列長度固定為 KMAX')
  assert.deepEqual(slot(f1, P - 1), [P - 1, 2 * (P - 1), 0, 0],
    'k=1:最後一格是目前位置,看不到任何位移 —— 這正是 k=1 資訊不足的來源')
  for (let j = 0; j < P - 1; j++)
    assert.ok(slot(f1, j).every(v => v === 0), 'k=1 時更早的格子必須留白')

  const f3 = CORE.featurize(hist, P, 3, 1)
  assert.deepEqual(slot(f3, P - 1), [P - 1, 2 * (P - 1), 1 * D, 2 * D], 'k=3:最後一格帶位移(已乘 DSCALE)')
  assert.deepEqual(slot(f3, P - 2), [P - 2, 2 * (P - 2), 1 * D, 2 * D], 'k=3:中間那一格也帶位移')
  assert.deepEqual(slot(f3, P - 3), [P - 3, 2 * (P - 3), 0, 0],
    'k=3:最舊的可見幀沒有前一步可減,位移留白')
  for (let j = 0; j < P - 3; j++)
    assert.ok(slot(f3, j).every(v => v === 0), 'k=3 時更舊的格子仍應留白')

  // 靠右對齊:不論 k 多少,最後一格永遠是目前狀態
  for (const k of [1, 2, 5, CORE.KMAX]) {
    const f = CORE.featurize(hist, P, k, 1)
    assert.deepEqual(slot(f, P - 1).slice(0, 2), [P - 1, 2 * (P - 1)], `k=${k}:最後一格應是目前位置`)
  }

  // 串流模式:位移欄整排留白,模型只讀到一串位置
  const fs = CORE.featurize(hist, P, P, 1, true)
  for (let j = 0; j < P; j++)
    assert.deepEqual(slot(fs, j), [j, 2 * j, 0, 0], `串流第 ${j} 格應只有位置`)

  // 多列:排列是 [時刻][列][SDIM],同一時刻的各列相鄰
  const two = new Float32Array(P * 2 * 2)
  for (let j = 0; j < P; j++) for (let i = 0; i < 2; i++) {
    two[j * 4 + 2 * i] = hist[2 * j] + i; two[j * 4 + 2 * i + 1] = hist[2 * j + 1]
  }
  const fm = CORE.featurize(two, P, 3, 2)
  const cell = (j, i) => Array.from(fm.slice((j * 2 + i) * S, (j * 2 + i + 1) * S))
  assert.equal(cell(P - 1, 1)[0] - cell(P - 1, 0)[0], 1, '第二列的位置應該比第一列多 1')
  assert.deepEqual(cell(P - 1, 0).slice(2), cell(P - 1, 1).slice(2), '兩列位移相同')
  assert.ok(cell(P - 4, 0).every(v => v === 0), 'k=3 時第 P−4 格對每一列都留白')
}
// trueDeltaSeq:串流訓練一次拿到 P−1 個監督訊號,每一個都要等於在那一刻單獨算的答案
{
  const P = 6, n = 3
  const hist = CORE.sampleWindows(data, n, P, 0.2, mulberry32(21))
  const seq = CORE.trueDeltaSeq(hist, P, n)
  assert.equal(seq.length, (P - 1) * n * 2, '第 0 幀沒有前一步,不該有監督訊號')
  for (let j = 1; j < P; j++) {
    const one = CORE.trueDeltaOfHist(hist.subarray(0, (j + 1) * n * 2), j + 1, n)
    assert.deepEqual(Array.from(seq.slice((j - 1) * n * 2, j * n * 2)), Array.from(one),
      `第 ${j} 幀的監督訊號應等於單獨計算的條件期望`)
  }
}

// 點選起點:找得到資料集裡最接近的那一幀
{
  const t = 5000
  const got = CORE.nearestIndex(data, data.pts[2 * t] + 1e-4, data.pts[2 * t + 1] - 1e-4, CORE.KMAX)
  assert.equal(got, t, 'nearestIndex 應回傳最接近查詢座標的那一幀')
}

// ---- (c) 訓練窗格取自真實軌跡;σ 把分布撐開 ----
const spread = sig => {
  const P = 2, mN = 3000
  const h = CORE.sampleWindows(data, mN, P, sig, mulberry32(4))
  const d = []
  for (let i = 0; i < mN; i++) d.push(CORE.distToCurve(h[mN * 2 + 2 * i], h[mN * 2 + 2 * i + 1]))
  return mean(d)
}
const sp0 = spread(0), sp4 = spread(0.4)
assert.ok(Math.abs(sp0 - devMean) < 0.05, 'σ=0 時訓練窗格的分布就是真實軌跡的分布')
assert.ok(sp4 > 1.5 * sp0, 'σ 調大必須把訓練分布往資料流形外撐開(這就是那個補丁)')

// ---- (d) 預測正確就沒有 exposure bias:同一份雜訊下兩條軌跡完全重合 ----
for (const P of [2, 4]) {
  const pre = CORE.makePrefix(data, N, P)
  const nz = CORE.makeNoise(N, STEPS, mulberry32(13))
  const tru = CORE.rollout(CORE.trueDeltaOfHist, pre, N, P, STEPS, nz)
  const per = CORE.rollout(CORE.trueDeltaOfHist, pre, N, P, STEPS, nz)
  assert.equal(CORE.gapAt(per, tru, N, P + STEPS - 1), 0,
    `P=${P}:模型等於真實動力學時,free-running 應與真實軌跡完全一致`)
  assert.equal(Math.max(...CORE.teacherErrors(CORE.trueDeltaOfHist, tru, N, P, STEPS)), 0,
    `P=${P}:模型等於真實動力學時,teacher forcing 單步誤差應為 0`)
  for (let f = 0; f < P; f++) assert.equal(CORE.gapAt(per, tru, N, f), 0, '前綴必須共用')
}

// ---- (e) k=1 的失敗模式:只看位置的最佳預測是條件均值 → 生成軌跡過度平滑 ----
// 終端速度 A(x)/C 正是「忽略隱藏速度」時講得通的最佳位置函數;拿它當預測器 rollout,
// 擺動會被抹掉 —— 這就是 demo 裡 k=1 的粉紅軌跡比真實軌跡還「乾淨」的原因。
const P2 = 2
const prefix = CORE.makePrefix(data, N, P2)
const noise = CORE.makeNoise(N, STEPS, mulberry32(11))
const truth = CORE.rollout(CORE.trueDeltaOfHist, prefix, N, P2, STEPS, noise)
const meanPred = (hist, P, n) => CORE.termVel(CORE.currentOf(hist, P, n), n)
const smooth = CORE.rollout(meanPred, prefix, N, P2, STEPS, noise)
const devTruth = CORE.manifoldDev(truth, N, P2 + STEPS)
const devSmooth = CORE.manifoldDev(smooth, N, P2 + STEPS)
assert.ok(devSmooth < 0.6 * devTruth,
  `只看位置的條件均值預測器應把擺動抹掉(${devSmooth.toFixed(3)} vs 真實 ${devTruth.toFixed(3)})`)

// ---- (f) 帶偏誤的模型:單步誤差不大,連續生成卻離開資料流形,且 q 前綴上的誤差更大 ----
const biased = (hist, P, n) => {
  const d = CORE.trueDeltaOfHist(hist, P, n)
  for (let i = 0; i < n * 2; i++) d[i] *= 1.05        // 每步多走 5%
  return d
}
const free = CORE.rollout(biased, prefix, N, P2, STEPS, noise)
const tfErr = mean(CORE.teacherErrors(biased, truth, N, P2, STEPS))
const devFree = CORE.manifoldDev(free, N, P2 + STEPS)
assert.ok(devFree > 1.5 * devTruth, 'free-running 應離開真實軌跡待著的範圍')
assert.ok(CORE.gapAt(free, truth, N, P2 + STEPS - 1) > 20 * tfErr,
  '連續生成的終點誤差應遠大於 teacher forcing 的單步誤差')
const errOnP = mean(CORE.teacherErrors(biased, truth, N, P2, STEPS))
const errOnQ = mean(CORE.teacherErrors(biased, free, N, P2, STEPS))
assert.ok(errOnQ > errOnP, '在模型自己生成的前綴上,單步誤差應大於在真實前綴上(訓練從未量過前者)')

console.log(`真實軌跡離曲線的平均距離 ${devMean.toFixed(3)}(繞著曲線擺動),每步位移 ${data.stepLen.toFixed(3)}`)
console.log(`MSE 下限: k=1 ${floorK1.toExponential(2)}  k≥2 ${floorK2.toExponential(2)}  → 相差 ${(floorK1 / floorK2).toFixed(0)}×`)
console.log(`訓練分布離流形: σ=0 → ${sp0.toFixed(3)}   σ=0.4 → ${sp4.toFixed(3)}`)
console.log(`條件均值預測器把擺動抹掉: ${devSmooth.toFixed(3)} vs 真實 ${devTruth.toFixed(3)}`)
console.log(`帶偏誤的模型: 單步誤差 ${tfErr.toExponential(2)},離流形 ${devFree.toFixed(3)};p 前綴 → q 前綴 ${(errOnQ / errOnP).toFixed(1)}×`)

console.log('ok')
