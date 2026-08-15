<script setup>
// ② 的核心證明:guidance 係數 w 就是那條軸上的座標。
// 曲線是數值算出來的,不是示意圖:p_guided ∝ p(x|c) · (p(x|c)/p(x))^w,再重新歸一化。
import { typeScale } from './chart-style.js'

const W = 700, H = 205
const FS = typeScale(W)

const N = 481, LO = -4.5, HI = 4.5
const xs = Array.from({ length: N }, (_, i) => LO + (i * (HI - LO)) / (N - 1))
const g = (x, m, s) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / s

// 無條件分布:兩個模式等重;條件分布:同樣兩個模式,但右峰被條件抬起來
const uncond = xs.map((x) => 0.5 * g(x, -1.5, 0.95) + 0.5 * g(x, 1.5, 0.95))
const cond = xs.map((x) => 0.34 * g(x, -1.5, 0.85) + 0.66 * g(x, 1.5, 0.85))

function guided(w) {
  const raw = xs.map((_, i) => Math.exp(Math.log(cond[i]) + w * (Math.log(cond[i]) - Math.log(uncond[i]))))
  const z = raw.reduce((a, b) => a + b, 0) * ((HI - LO) / (N - 1))
  return raw.map((v) => v / z)
}

const CURVES = [
  { w: 0, c: '#5edfff', label: 'w = 0(就是 p(x|c) 本身)' },
  { w: 1.5, c: '#ffb454', label: 'w = 1.5' },
  { w: 4, c: '#ff6b9d', label: 'w = 4' },
].map((d) => ({ ...d, y: guided(d.w) }))

const PAD = { l: 44, r: 16, t: 26, b: 34 }
const PW = W - PAD.l - PAD.r, PH = H - PAD.t - PAD.b
const YMAX = Math.max(...CURVES.flatMap((c) => c.y)) * 1.08
const px = (x) => PAD.l + ((x - LO) / (HI - LO)) * PW
const py = (y) => PAD.t + PH - (y / YMAX) * PH
const path = (ys) => ys.map((y, i) => `${i ? 'L' : 'M'}${px(xs[i]).toFixed(1)},${py(y).toFixed(1)}`).join('')
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <line :x1="PAD.l" :y1="PAD.t + PH" :x2="W - PAD.r" :y2="PAD.t + PH" stroke="#243350" />

    <path :d="path(CURVES[0].y)" fill="none" stroke="#5edfff" stroke-width="2" />
    <path :d="path(CURVES[1].y)" fill="none" stroke="#ffb454" stroke-width="2" />
    <path :d="path(CURVES[2].y)" fill="none" stroke="#ff6b9d" stroke-width="2" />

    <!-- 左峰被逐步抹掉的位置 -->
    <line :x1="px(-1.5)" :y1="PAD.t + 4" :x2="px(-1.5)" :y2="PAD.t + PH"
          stroke="#8fa0bc" stroke-dasharray="3 3" opacity="0.45" />
    <text :x="px(-1.5) - 6" :y="PAD.t + 14" fill="#8fa0bc" :font-size="FS.small" text-anchor="end">
      條件較不偏好的那個模式
    </text>

    <g :font-size="FS.small" fill="#8fa0bc">
      <rect v-for="(c, i) in CURVES" :key="'k' + i" :x="PAD.l + 4 + i * 176" :y="H - 16" width="16" height="3"
            :fill="c.c" />
      <text v-for="(c, i) in CURVES" :key="'t' + i" :x="PAD.l + 26 + i * 176" :y="H - 10">{{ c.label }}</text>
    </g>

    <text :x="W - PAD.r" y="16" fill="#ff6b9d" :font-size="FS.title" text-anchor="end">
      w 越大 → 質量從次要模式移走 → 越往 mode-seeking 端
    </text>
  </svg>
</template>
