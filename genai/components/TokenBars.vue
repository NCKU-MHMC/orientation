<script setup>
// forward KL 在 token 層級的實作:目標是 one-hot,但模型永遠不敢輸出 0。
import { typeScale } from './chart-style.js'

const TOKENS = [
  { t: '巴黎', target: 1, q: 0.62 },
  { t: '里昂', target: 0, q: 0.14 },
  { t: '法國', target: 0, q: 0.11 },
  { t: '那', target: 0, q: 0.07 },
  { t: '香蕉', target: 0, q: 0.04 },
  { t: '≈0 的一長串', target: 0, q: 0.02 },
]
const W = 700, H = 205
const FS = typeScale(W)
const BW = 78, BASE = 152
const x0 = (i) => 56 + i * 112
const barH = (v) => v * 96
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <text x="40" y="22" fill="#ffb454" :font-size="FS.title">
      任何一個 token 被壓到 0,只要語料裡出現過一次 → 懲罰 +∞
    </text>
    <text x="40" y="40" fill="#8fa0bc" :font-size="FS.small">
      所以尾巴永遠留著質量。這是 forward KL 的結構性後果,與模型容量無關。
    </text>

    <line x1="40" :y1="BASE" :x2="W - 20" :y2="BASE" stroke="#243350" />

    <g v-for="(tk, i) in TOKENS" :key="tk.t">
      <!-- 語料的目標:one-hot -->
      <rect :x="x0(i)" :y="BASE - barH(tk.target)" :width="BW * 0.42" :height="barH(tk.target)"
            fill="#b48cff" fill-opacity="0.35" stroke="#b48cff" />
      <!-- 模型 q -->
      <rect :x="x0(i) + BW * 0.5" :y="BASE - barH(tk.q)" :width="BW * 0.42" :height="barH(tk.q)"
            fill="#5edfff" fill-opacity="0.35" stroke="#5edfff" />
      <text :x="x0(i) + BW * 0.46" :y="BASE + 16" fill="#8fa0bc" :font-size="FS.label" text-anchor="middle">
        {{ tk.t }}
      </text>
      <text v-if="tk.target === 0" :x="x0(i) + BW * 0.71" :y="BASE - barH(tk.q) - 6"
            fill="#5edfff" :font-size="FS.small" text-anchor="middle" opacity="0.85">&gt; 0</text>
    </g>

    <g :font-size="FS.small">
      <rect x="40" y="182" width="14" height="10" fill="#b48cff" fill-opacity="0.35" stroke="#b48cff" />
      <text x="60" y="191" fill="#8fa0bc">語料的目標(one-hot)</text>
      <rect x="240" y="182" width="14" height="10" fill="#5edfff" fill-opacity="0.35" stroke="#5edfff" />
      <text x="260" y="191" fill="#8fa0bc">模型 q(· |「法國的首都是」)</text>
    </g>
  </svg>
</template>
