<script setup>
// ③ 第 2 層:同一組 logits,四種取樣設定。四個面板的機率都是算出來的。
// 這張圖替代互動版 demo:重點是 T 與 top-p 一個拉平分布、一個切掉尾巴,方向相同、手法不同。
import { typeScale } from './chart-style.js'

const W = 700, H = 200
const FS = typeScale(W)

const TOK = ['很', '不', '相', '有', '挺', '略', '超', '尚']
const LOGITS = [3.4, 2.6, 2.1, 1.5, 0.9, 0.3, -0.4, -1.2]

const softmax = (z, T) => {
  const e = z.map((v) => Math.exp(v / T))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}
const topP = (p, thr) => {
  const idx = p.map((v, i) => i).sort((a, b) => p[b] - p[a])
  let acc = 0
  const keep = new Set()
  for (const i of idx) { keep.add(i); acc += p[i]; if (acc >= thr) break }
  const z = idx.filter((i) => keep.has(i)).reduce((a, i) => a + p[i], 0)
  return p.map((v, i) => (keep.has(i) ? v / z : 0))
}

const base = softmax(LOGITS, 1)
const PANEL = [
  { title: 'T = 1.5', sub: '分布被拉平', p: softmax(LOGITS, 1.5), c: '#5edfff' },
  { title: 'T = 1', sub: '模型原本的分布', p: base, c: '#8fa0bc' },
  { title: 'T = 0.6', sub: '分布被拉尖', p: softmax(LOGITS, 0.6), c: '#ff6b9d' },
  { title: 'top-p = 0.8', sub: '尾巴直接歸零', p: topP(base, 0.8), c: '#ffb454' },
]

const PW = 165, GAP = 12, X0 = 14
const BASE = 152, MAXH = 88
const bx = (pi, i) => X0 + pi * (PW + GAP) + 10 + i * 18
const bh = (v) => v * MAXH * 1.9
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <g v-for="(pn, pi) in PANEL" :key="pn.title">
      <rect :x="X0 + pi * (PW + GAP)" y="8" :width="PW" :height="H - 26" rx="7"
            fill="#151d2e" :stroke="pn.c" stroke-opacity="0.45" />
      <text :x="X0 + pi * (PW + GAP) + 10" y="28" :fill="pn.c" :font-size="FS.title">{{ pn.title }}</text>
      <text :x="X0 + pi * (PW + GAP) + 10" y="44" fill="#8fa0bc" :font-size="FS.small">{{ pn.sub }}</text>

      <line :x1="X0 + pi * (PW + GAP) + 8" :y1="BASE" :x2="X0 + pi * (PW + GAP) + PW - 8" :y2="BASE"
            stroke="#243350" />

      <g v-for="(v, i) in pn.p" :key="i">
        <rect :x="bx(pi, i)" :y="BASE - bh(v)" width="13" :height="bh(v)"
              :fill="pn.c" fill-opacity="0.35" :stroke="pn.c" />
        <text v-if="v === 0" :x="bx(pi, i) + 6.5" :y="BASE - 4" fill="#8fa0bc"
              :font-size="FS.small" text-anchor="middle" opacity="0.7">×</text>
        <text :x="bx(pi, i) + 6.5" :y="BASE + 14" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">
          {{ TOK[i] }}
        </text>
      </g>
    </g>

    <text :x="W / 2" :y="H - 4" fill="#ffb454" :font-size="FS.title" text-anchor="middle">
      T 連續地重新分配質量,top-p 直接截斷尾部;兩者都降低分布的熵
    </text>
  </svg>
</template>
