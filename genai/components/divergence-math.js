// 元件共用的機率計算。
export const LOG2 = Math.log(2)

export const gauss = (x, mu, s) =>
  Math.exp(-((x - mu) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI))

// comps: [[weight, mu, sigma], ...]
export const mix = (x, comps) => comps.reduce((a, [w, mu, s]) => a + w * gauss(x, mu, s), 0)

// 兩個標準差 s 的高斯相距 d 時的 JSD(數值積分)
export function jsdGauss(d, s = 1, lo = -14, hi = 14, n = 2800) {
  const h = (hi - lo) / n
  let acc = 0
  for (let i = 0; i <= n; i++) {
    const x = lo + i * h
    const p = gauss(x, -d / 2, s)
    const q = gauss(x, d / 2, s)
    const m = (p + q) / 2
    let t = 0
    if (p > 1e-300) t += 0.5 * p * Math.log(p / m)
    if (q > 1e-300) t += 0.5 * q * Math.log(q / m)
    acc += t * h * (i === 0 || i === n ? 0.5 : 1)
  }
  return acc
}

export function softmax(logits, T = 1) {
  const m = Math.max(...logits)
  const e = logits.map((l) => Math.exp((l - m) / T))
  const Z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / Z)
}

// 曲線座標轉 SVG path
export function pathOf(xs, ys, x0, x1, y0, y1, w, h, pad = 0) {
  const sx = (x) => pad + ((x - x0) / (x1 - x0)) * (w - 2 * pad)
  const sy = (y) => h - pad - ((y - y0) / (y1 - y0)) * (h - 2 * pad)
  return xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join('')
}
