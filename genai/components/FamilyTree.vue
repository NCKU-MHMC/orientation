<script setup>
// ③ 的分類樹(Goodfellow, NIPS 2016 Tutorial 的骨架)。
// 分岔點只有一個問題:p_θ(x) 這個數值,你寫不寫得出來。
import { typeScale } from './chart-style.js'

const W = 700, H = 250
const FS = typeScale(W)

const NODES = [
  { id: 'root', x: 8, y: 108, w: 118, h: 44, c: '#b48cff', t: '分布逼近', s: 'min D(p_data ‖ p_θ)' },

  { id: 'exp', x: 168, y: 52, w: 118, h: 44, c: '#5edfff', t: '顯式密度', s: '寫得出 p_θ(x)' },
  { id: 'imp', x: 168, y: 176, w: 118, h: 44, c: '#ff6b9d', t: '隱式密度', s: '只能取樣' },

  { id: 'exact', x: 330, y: 14, w: 112, h: 40, c: '#5edfff', t: '可精確計算', s: '' },
  { id: 'approx', x: 330, y: 96, w: 112, h: 40, c: '#5edfff', t: '取下界 / 近似', s: '' },

  { id: 'ar', x: 486, y: 4, w: 206, h: 30, c: '#5edfff', t: 'Autoregressive', s: 'GPT · PixelCNN · WaveNet' },
  { id: 'flow', x: 486, y: 40, w: 206, h: 30, c: '#5edfff', t: 'Normalizing Flow', s: 'RealNVP · Glow' },
  { id: 'vae', x: 486, y: 86, w: 206, h: 30, c: '#5edfff', t: 'VAE', s: '變分下界 ELBO', hi: true },
  { id: 'dpm', x: 486, y: 122, w: 206, h: 30, c: '#5edfff', t: 'Diffusion / Flow Matching', s: '同一框架的不同路徑參數化' },
  { id: 'gan', x: 486, y: 183, w: 206, h: 30, c: '#ff6b9d', t: 'GAN', s: '判別器當評估基準', hi: true },
]
const N = Object.fromEntries(NODES.map((n) => [n.id, n]))
const EDGES = [
  ['root', 'exp'], ['root', 'imp'],
  ['exp', 'exact'], ['exp', 'approx'],
  ['exact', 'ar'], ['exact', 'flow'],
  ['approx', 'vae'], ['approx', 'dpm'],
  ['imp', 'gan'],
]

// 直角折線:從 a 的右緣拉到 b 的左緣,中點轉折
const elbow = ([ai, bi]) => {
  const a = N[ai], b = N[bi]
  const x1 = a.x + a.w, y1 = a.y + a.h / 2
  const x2 = b.x, y2 = b.y + b.h / 2
  const mx = x1 + (x2 - x1) / 2
  return `M${x1},${y1} H${mx} V${y2} H${x2}`
}
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <path v-for="e in EDGES" :key="e.join()" :d="elbow(e)"
          fill="none" stroke="#3a4a68" stroke-width="1.3" />

    <g v-for="n in NODES" :key="n.id">
      <rect :x="n.x" :y="n.y" :width="n.w" :height="n.h" rx="7"
            :fill="n.hi ? n.c : '#151d2e'" :fill-opacity="n.hi ? 0.16 : 1"
            :stroke="n.c" :stroke-width="n.hi ? 2 : 1.3" />
      <text :x="n.x + n.w / 2" :y="n.s ? n.y + n.h / 2 - 2 : n.y + n.h / 2 + 4"
            :fill="n.hi ? n.c : '#e8edf6'" :font-size="FS.title" text-anchor="middle"
            :font-weight="n.hi ? 600 : 400">{{ n.t }}</text>
      <text v-if="n.s" :x="n.x + n.w / 2" :y="n.y + n.h / 2 + 12"
            fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">{{ n.s }}</text>
    </g>

    <text x="150" y="130" fill="#ffb454" :font-size="FS.small" text-anchor="middle">分岔點</text>
    <text x="310" y="80" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">積分算得動?</text>

    <text :x="W - 4" :y="H - 4" fill="#8fa0bc" :font-size="FS.small" text-anchor="end" opacity="0.7">
      唯一的分類依據:p_θ(x) 這個數值寫不寫得出來
    </text>
  </svg>
</template>
