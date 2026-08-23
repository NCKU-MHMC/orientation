<script setup>
// 圖 B-1:單一高斯 q 擬合雙峰 p,三種散度給出三種解。
import { fs } from './chart-style.js'
import { palette } from './chart-style.js'
import { gauss, mix, pathOf } from './divergence-math.js'

const P = [[0.5, -2, 0.6], [0.5, 2, 0.6]]
const xs = []
for (let x = -5; x <= 5.001; x += 0.05) xs.push(x)
const pYs = xs.map((x) => mix(x, P))

// forward KL 的最優解為動差匹配;reverse KL 鎖定單峰;JSD 取折衷(示意值)
const panels = [
  { name: 'forward KL', mu: 0, s: 2.09, note: '覆蓋兩峰,質量填進峰間空隙' },
  { name: 'JSD', mu: 1.1, s: 1.25, note: '介於兩者之間的折衷' },
  { name: 'reverse KL', mu: 2, s: 0.6, note: '鎖定單一眾數,放棄另一峰' },
]
const W = 300, H = 225, PAD = 10
// 三個面板共用上緣才可比較;取 p 與各 q 的峰值(高斯峰在 mu 處)加留白,免得最窄的 q 被裁掉
const yMax = Math.max(...pYs, ...panels.map((pn) => gauss(pn.mu, pn.mu, pn.s))) * 1.18
const pPath = pathOf(xs, pYs, -5, 5, 0, yMax, W, H, PAD)
const qPath = (mu, s) => pathOf(xs, xs.map((x) => gauss(x, mu, s)), -5, 5, 0, yMax, W, H, PAD)
</script>

<template>
  <div class="fit-row">
    <div v-for="pn in panels" :key="pn.name" class="fit-panel">
      <svg :width="W" :height="H + 8" :viewBox="`0 0 ${W} ${H + 8}`">
        <line :x1="PAD" :y1="H - PAD" :x2="W - PAD" :y2="H - PAD" :stroke="palette.grid" stroke-width="1.5" />
        <path :d="pPath + `L${W - PAD},${H - PAD}L${PAD},${H - PAD}Z`" :fill="palette.p" fill-opacity="0.15" />
        <path :d="pPath" :stroke="palette.p" stroke-width="2" fill="none" />
        <path :d="qPath(pn.mu, pn.s)" :stroke="palette.q" stroke-width="2.5" fill="none" />
        <text :x="W / 2" :y="20" text-anchor="middle" :fill="palette.ink" font-weight="600"
          :style="{ fontSize: fs('label') }">{{ pn.name }}</text>
      </svg>
      <div class="fit-note">{{ pn.note }}</div>
    </div>
  </div>
  <div class="fit-legend">
    <span><i :style="{ background: palette.p }" /> 雙峰目標 p</span>
    <span><i :style="{ background: palette.q }" /> 單一高斯 q(該散度下的解)</span>
  </div>
</template>

<style scoped>
.fit-row { display: flex; gap: 0.6rem; justify-content: center; }
.fit-panel { text-align: center; }
.fit-panel svg { margin: 0 auto; }
.fit-note { font-size: 0.8rem; color: var(--muted); margin-top: 0.1rem; }
.fit-legend { display: flex; gap: 1.6rem; justify-content: center; margin-top: 0.7rem; font-size: 0.85rem; color: var(--ink); }
.fit-legend i { display: inline-block; width: 0.9em; height: 0.35em; border-radius: 2px; margin-right: 0.35em; vertical-align: middle; }
</style>
