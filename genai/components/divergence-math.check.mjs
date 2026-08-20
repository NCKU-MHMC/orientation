import { jsdGauss, softmax, LOG2, gauss, mix } from './divergence-math.js'

const close = (a, b, tol = 1e-3) => Math.abs(a - b) < tol
console.assert(close(jsdGauss(0), 0), 'JSD(p,p)=0')
console.assert(close(jsdGauss(12), LOG2, 1e-4), 'JSD 分離後趨近 log2')
console.assert(jsdGauss(1) < jsdGauss(2) && jsdGauss(2) < jsdGauss(4), 'JSD 隨距離單調')

const p = softmax([2, 1, 0])
console.assert(close(p.reduce((a, b) => a + b, 0), 1, 1e-9), 'softmax 歸一')
const flat = softmax([2, 1, 0], 10)
console.assert(flat[0] - flat[2] < p[0] - p[2], '高溫攤平')

// 混合密度積分約為 1
let acc = 0
for (let x = -12; x <= 12; x += 0.01) acc += mix(x, [[0.5, -2, 0.6], [0.5, 2, 0.6]]) * 0.01
console.assert(close(acc, 1, 1e-2), '混合密度歸一')
console.assert(close(gauss(0, 0, 1), 0.39894, 1e-4), '標準高斯峰值')
console.log('divergence-math: OK')
