<script setup>
// GAN 改進的共同結構:成因在評估基準,不在架構。
// 左:JSD 在不重疊時飽和,D 地景是一道階梯,G 站在平台上沒有坡可爬。
// 右:Lipschitz 約束(WGAN / GP / SN / R1)把地景壓平緩,任何位置都有方向。
import { typeScale } from './chart-style.js'

const W = 700, H = 200
const FS = typeScale(W)

const PW = 320, BASE = 132, TOP = 40
const sx = (o) => (t) => o + t * PW              // t ∈ [0,1]
const sy = (v) => BASE - v * (BASE - TOP)        // v ∈ [0,1]

// 左:近似階梯(sigmoid 拉很陡)
const stepPath = (o) => {
  const X = sx(o)
  return Array.from({ length: 81 }, (_, i) => {
    const t = i / 80
    const v = 1 / (1 + Math.exp(-40 * (t - 0.52)))
    return `${i ? 'L' : 'M'}${X(t).toFixed(1)},${sy(v).toFixed(1)}`
  }).join('')
}
// 右:平緩斜坡
const rampPath = (o) => {
  const X = sx(o)
  return Array.from({ length: 81 }, (_, i) => {
    const t = i / 80
    const v = 0.12 + 0.76 * t
    return `${i ? 'L' : 'M'}${X(t).toFixed(1)},${sy(v).toFixed(1)}`
  }).join('')
}

const L = 22, R = 368
const ballT = 0.16
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <defs>
      <marker id="gf-a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#ffb454" />
      </marker>
    </defs>

    <!-- 左:飽和的 JSD 地景 -->
    <text :x="L" y="22" fill="#ff6b9d" :font-size="FS.title">原始 GAN · D 訓到最優</text>
    <path :d="stepPath(L)" stroke="#ff6b9d" stroke-width="2.2" fill="none" />
    <circle :cx="sx(L)(ballT)" :cy="sy(0.02) - 7" r="6.5" fill="#e8edf6" />
    <text :x="sx(L)(ballT) + 12" :y="sy(0.02) - 4" fill="#8fa0bc" :font-size="FS.small">p_g 在這裡</text>
    <text :x="sx(L)(0.14)" :y="sy(0.02) + 22" fill="#ff6b9d" :font-size="FS.label">地景是平的 → 梯度 ≈ 0</text>
    <line :x1="L" :y1="BASE" :x2="L + PW" :y2="BASE" stroke="#243350" />

    <!-- 右:Lipschitz 約束後 -->
    <text :x="R" y="22" fill="#5edfff" :font-size="FS.title">WGAN / GP / SN / R1 · 約束 D 的 Lipschitz 常數</text>
    <path :d="rampPath(R)" stroke="#5edfff" stroke-width="2.2" fill="none" />
    <circle :cx="sx(R)(ballT)" :cy="sy(0.12 + 0.76 * ballT) - 8" r="6.5" fill="#e8edf6" />
    <line :x1="sx(R)(ballT) + 14" :y1="sy(0.12 + 0.76 * ballT) - 4"
          :x2="sx(R)(ballT) + 62" :y2="sy(0.12 + 0.76 * ballT) + 16"
          stroke="#ffb454" stroke-width="2" marker-end="url(#gf-a)" />
    <text :x="sx(R)(0.14)" :y="sy(0.02) + 22" fill="#5edfff" :font-size="FS.label">任何位置都有坡可爬</text>
    <line :x1="R" :y1="BASE" :x2="R + PW" :y2="BASE" stroke="#243350" />

    <line :x1="W / 2 + 4" y1="14" :x2="W / 2 + 4" :y2="BASE + 8" stroke="#243350" stroke-dasharray="4 4" />

    <text :x="L" :y="H - 22" fill="#8fa0bc" :font-size="FS.small">
      W₁(p,q) = sup 在 ‖f‖_L ≤ 1 之下的 E_p[f] − E_q[f]:兩個分布不重疊也給得出「差多遠」,而不是只回答「完全不同」。
    </text>
    <text :x="L" :y="H - 5" fill="#ffb454" :font-size="FS.small">
      共同動作:不換架構,而是換掉評估基準,或限制這個基準的地景不得過陡。
    </text>
  </svg>
</template>
