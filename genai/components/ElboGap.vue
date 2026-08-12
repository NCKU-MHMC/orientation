<script setup>
// ELBO 與 log p(x) 的關係:log p(x) = ELBO + KL(q ‖ 真後驗)。
// 間隙不是誤差項,是 q_φ 偏離真後驗的程度。q 越靈活,下界越緊。
import { typeScale } from './chart-style.js'

const W = 700, H = 225
const FS = typeScale(W)

const TOP = 44, BASE = 158, FULL = BASE - TOP  // log p(x) 的總高度
const BW = 108
const CASES = [
  { x: 96, gap: 0.42, t: '對角高斯 q', s: '每維獨立,無法表達後驗的相關性' },
  { x: 306, gap: 0.16, t: '較靈活的 q', s: 'normalizing flow / 更深的 encoder' },
  { x: 516, gap: 0.0, t: 'q = 真後驗', s: '下界收緊成等式(實際做不到)' },
]
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <defs>
      <pattern id="eg-hatch" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
        <line x1="0" y1="0" x2="0" y2="6" stroke="#ff6b9d" stroke-width="2.4" stroke-opacity="0.55" />
      </pattern>
    </defs>

    <!-- 三根柱子共用同一個天花板:log p(x) 與 q 無關 -->
    <line x1="60" :y1="TOP" :x2="W - 24" :y2="TOP" stroke="#b48cff" stroke-width="1.2" stroke-dasharray="5 4" />
    <text x="60" :y="TOP - 9" fill="#b48cff" :font-size="FS.title">log p_θ(x) · 與 q 無關,固定在這條線上</text>

    <g v-for="c in CASES" :key="c.t">
      <!-- 間隙 = KL(q ‖ p(z|x)) -->
      <rect :x="c.x" :y="TOP" :width="BW" :height="FULL * c.gap"
            fill="url(#eg-hatch)" stroke="#ff6b9d" stroke-width="1.2" />
      <!-- ELBO -->
      <rect :x="c.x" :y="TOP + FULL * c.gap" :width="BW" :height="FULL * (1 - c.gap)"
            fill="#5edfff" fill-opacity="0.2" stroke="#5edfff" stroke-width="1.6" />

      <text v-if="c.gap > 0.02" :x="c.x + BW / 2" :y="TOP + (FULL * c.gap) / 2 + 4"
            fill="#ff6b9d" :font-size="FS.small" text-anchor="middle">間隙</text>
      <text :x="c.x + BW / 2" :y="TOP + FULL * c.gap + (FULL * (1 - c.gap)) / 2 + 4"
            fill="#5edfff" :font-size="FS.title" text-anchor="middle">ELBO</text>

      <text :x="c.x + BW / 2" :y="BASE + 20" fill="#e8edf6" :font-size="FS.title" text-anchor="middle">{{ c.t }}</text>
      <text :x="c.x + BW / 2" :y="BASE + 37" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">{{ c.s }}</text>
    </g>

    <line x1="60" :y1="BASE" :x2="W - 24" :y2="BASE" stroke="#243350" />

    <text x="60" :y="H - 8" fill="#ffb454" :font-size="FS.small">
      間隙 = KL(q_φ(z|x) ‖ p_θ(z|x)) ≥ 0。所以 ELBO 恆為下界,而且間隙本身無法直接量測。
    </text>
  </svg>
</template>
