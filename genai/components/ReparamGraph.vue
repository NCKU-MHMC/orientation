<script setup>
// 重參數化:把隨機性搬到計算圖外面。
// 上排是可微的實線路徑,ε 掛在外面用虛線接進來,反向傳播不需要穿過它。
import { typeScale } from './chart-style.js'

const W = 700, H = 200
const FS = typeScale(W)

const Y = 66
const BOX = [
  { x: 14, w: 62, t: 'x', s: '輸入' },
  { x: 116, w: 96, t: 'Encoder φ', s: '' },
  { x: 252, w: 96, t: 'μ(x), σ(x)', s: '' },
  { x: 388, w: 118, t: 'z = μ + σ⊙ε', s: '' },
  { x: 546, w: 96, t: 'Decoder θ', s: '' },
]
const link = (i) => ({ x1: BOX[i].x + BOX[i].w, x2: BOX[i + 1].x })
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <defs>
      <marker id="rp-fwd" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#5edfff" />
      </marker>
      <marker id="rp-back" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#ffb454" />
      </marker>
      <marker id="rp-noise" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#8fa0bc" />
      </marker>
    </defs>

    <!-- 前向:實線,可微 -->
    <line v-for="i in [0, 1, 2, 3]" :key="'l' + i"
          :x1="link(i).x1" :y1="Y + 17" :x2="link(i).x2 - 3" :y2="Y + 17"
          stroke="#5edfff" stroke-width="1.8" marker-end="url(#rp-fwd)" />

    <g v-for="b in BOX" :key="b.t">
      <rect :x="b.x" :y="Y" :width="b.w" height="34" rx="6"
            fill="#0c1220" stroke="#5edfff" stroke-width="1.5" />
      <text :x="b.x + b.w / 2" :y="Y + 22" fill="#e8edf6" :font-size="FS.title" text-anchor="middle">{{ b.t }}</text>
    </g>

    <!-- 輸出 -->
    <line x1="642" :y1="Y + 17" x2="676" :y2="Y + 17" stroke="#5edfff" stroke-width="1.8" marker-end="url(#rp-fwd)" />
    <text x="682" :y="Y + 22" fill="#5edfff" :font-size="FS.title">x̂</text>

    <!-- ε:掛在圖外面,虛線接進來 -->
    <ellipse cx="447" cy="146" rx="62" ry="19" fill="#151d2e" stroke="#8fa0bc" stroke-width="1.4" stroke-dasharray="4 3" />
    <text x="447" y="151" fill="#8fa0bc" :font-size="FS.title" text-anchor="middle">ε ~ N(0, I)</text>
    <line x1="447" y1="127" x2="447" :y2="Y + 40" stroke="#8fa0bc" stroke-width="1.4"
          stroke-dasharray="4 3" marker-end="url(#rp-noise)" />
    <text x="462" y="118" fill="#8fa0bc" :font-size="FS.small">外部噪聲,視為常數</text>

    <!-- 反向:梯度沿實線一路回到 encoder -->
    <path d="M646,42 H160" stroke="#ffb454" stroke-width="1.6" fill="none"
          stroke-dasharray="6 4" marker-end="url(#rp-back)" />
    <text x="404" y="34" fill="#ffb454" :font-size="FS.title" text-anchor="middle">
      梯度沿 μ、σ 直通回 encoder
    </text>

    <text x="14" :y="H - 8" fill="#8fa0bc" :font-size="FS.small">
      不對「抽樣」這個動作微分,而是把骰子丟到模型外面,模型只負責平移(μ)與縮放(σ)骰子的結果。
    </text>
  </svg>
</template>
