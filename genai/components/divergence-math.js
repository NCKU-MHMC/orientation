// 投影片圖表共用的一維散度計算。
// 目的:B-1 那張圖上的三條 q 曲線是真的解出來的,不是手畫的。

const SQRT2PI = Math.sqrt(2 * Math.PI)
const EPS = 1e-12

export const gauss = (x, mu, sigma) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * SQRT2PI)

export const DX = 0.025
export const XS = Array.from({ length: 481 }, (_, i) => -6 + i * DX)

/** 雙峰資料分布:左峰權重 w,兩峰在 ∓1.6、寬度 0.55 */
export const bimodal = (w = 0.5) =>
  XS.map((x) => w * gauss(x, -1.6, 0.55) + (1 - w) * gauss(x, 1.6, 0.55))

const kl = (a, b) => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * Math.log((a[i] + EPS) / (b[i] + EPS))
  return s * DX
}

export const divergences = (p, q) => {
  const m = p.map((v, i) => 0.5 * (v + q[i]))
  return {
    forward: kl(p, q), // KL(p‖q):權重是 p
    reverse: kl(q, p), // KL(q‖p):權重是 q
    jsd: 0.5 * kl(p, m) + 0.5 * kl(q, m),
  }
}

/**
 * 用單一高斯 q 擬合 p,最小化指定散度。
 * ponytail: 粗網格窮舉。比寫最佳化器短,精度也足夠畫圖。
 * 一次要跑 ~7k 組合 × 481 點,所以投影片不即時呼叫,只用來產生/驗證下面的 FITS。
 */
export function fitGaussian(which, p) {
  let best = { mu: 0, sigma: 1, val: Infinity }
  for (let mu = -2.6; mu <= 2.6 + 1e-9; mu += 0.04) {
    for (let sigma = 0.3; sigma <= 2.6 + 1e-9; sigma += 0.04) {
      const q = XS.map((x) => gauss(x, mu, sigma))
      const val = divergences(p, q)[which]
      if (val < best.val) best = { mu: +mu.toFixed(2), sigma: +sigma.toFixed(2), val }
    }
  }
  return best
}

/**
 * fitGaussian 的結果,固定成常數讓投影片零成本渲染。
 * 由 divergence-math.check.mjs 驗證仍與實算一致。
 *
 * 有意思的地方在 w=0.3(右峰較重):兩峰一旦不對稱,JSD 就從「蓋住全部」翻邊成
 * 「鎖定主峰」,而 forward KL 永遠不翻、reverse KL 永遠翻。
 * 這才是「JSD 在中間」的精確意思。
 */
export const FITS = {
  0.5: {
    forward: { mu: 0, sigma: 1.7 },
    jsd: { mu: 0, sigma: 1.66 },
    reverse: { mu: 1.6, sigma: 0.58 },
  },
  0.3: {
    forward: { mu: 0.64, sigma: 1.58 },
    jsd: { mu: 1.6, sigma: 0.58 },
    reverse: { mu: 1.6, sigma: 0.58 },
  },
}

/**
 * 互動用:一次掃描同時解出三個散度的最佳高斯,依 w 快取。
 * ponytail: 網格比 fitGaussian 粗一倍(0.08)且三個散度共用同一次掃描,
 * 換到 ~30ms/次,滑桿拖到哪算到哪;要更高精度的定值用 FITS。
 * mu 以 0 為中心取格點,對稱情況才會剛好解到 mu=0。
 */
const fitCache = new Map()
export function fitAll(w) {
  const key = w.toFixed(2)
  if (fitCache.has(key)) return fitCache.get(key)
  const p = bimodal(w)
  const best = { forward: { val: Infinity }, jsd: { val: Infinity }, reverse: { val: Infinity } }
  for (let i = -32; i <= 32; i++) {
    const mu = +(i * 0.08).toFixed(2)
    for (let j = 0; j <= 29; j++) {
      const sigma = +(0.3 + j * 0.08).toFixed(2)
      const q = XS.map((x) => gauss(x, mu, sigma))
      const d = divergences(p, q)
      for (const k in best) if (d[k] < best[k].val) best[k] = { mu, sigma, val: d[k] }
    }
  }
  fitCache.set(key, best)
  return best
}

/** 兩個等寬高斯相距 d 時的三個散度 —— 用來畫 JSD 的飽和曲線 */
export function separationCurve(sigma = 0.5, dMax = 6, steps = 61) {
  return Array.from({ length: steps }, (_, i) => {
    const d = (i / (steps - 1)) * dMax
    const p = XS.map((x) => gauss(x, -d / 2, sigma))
    const q = XS.map((x) => gauss(x, d / 2, sigma))
    return { d, ...divergences(p, q) }
  })
}
