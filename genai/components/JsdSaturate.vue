<script setup>
// JSD 隨兩分布距離的變化:有界於 log 2,支撐集分離後飽和、梯度消失。
import { fs, palette } from './chart-style.js'
import { jsdGauss, pathOf, LOG2 } from './divergence-math.js'

const ds = []
for (let d = 0; d <= 10.001; d += 0.2) ds.push(d)
const ys = ds.map((d) => jsdGauss(d))
const W = 640, H = 260, PAD = 40
const curve = pathOf(ds, ys, 0, 10, 0, 0.78, W, H, PAD)
const sy = (y) => H - PAD - (y / 0.78) * (H - 2 * PAD)
const sx = (x) => PAD + (x / 10) * (W - 2 * PAD)
</script>

<template>
  <div class="jsd-sat">
    <svg :width="W" :height="H" :viewBox="`0 0 ${W} ${H}`">
      <line :x1="PAD" :y1="H - PAD" :x2="W - PAD" :y2="H - PAD" :stroke="palette.ink" stroke-width="1.2" />
      <line :x1="PAD" :y1="H - PAD" :x2="PAD" :y2="PAD - 14" :stroke="palette.ink" stroke-width="1.2" />
      <line :x1="PAD" :y1="sy(LOG2)" :x2="W - PAD" :y2="sy(LOG2)" :stroke="palette.bad" stroke-width="1.4"
        stroke-dasharray="6 4" />
      <text :x="W - PAD" :y="sy(LOG2) - 8" text-anchor="end" :fill="palette.bad"
        :style="{ fontSize: fs('note') }">上界 log 2</text>
      <path :d="curve" :stroke="palette.accent" stroke-width="2.6" fill="none" />
      <text :x="sx(7.6)" :y="sy(0.66)" text-anchor="middle" :fill="palette.muted"
        :style="{ fontSize: fs('note') }">曲線變平:梯度趨近 0</text>
      <text :x="sx(1.6)" :y="sy(0.1)" text-anchor="middle" :fill="palette.accent"
        :style="{ fontSize: fs('note') }">重疊時仍有斜率</text>
      <text :x="W / 2" :y="H - 8" text-anchor="middle" :fill="palette.ink"
        :style="{ fontSize: fs('tick') }">兩個高斯的峰間距 d(標準差 = 1)</text>
      <text :x="14" :y="PAD - 4" :fill="palette.ink" :style="{ fontSize: fs('tick') }">JSD</text>
    </svg>
    <div class="jsd-cap">數值積分結果:兩個單位高斯相距 d 時的 JSD。支撐集實際分離後,再拉遠也幾乎不再增加。</div>
  </div>
</template>

<style scoped>
.jsd-sat { text-align: center; }
.jsd-sat svg { margin: 0 auto; }
.jsd-cap { font-size: 0.85rem; color: var(--ink-2); margin-top: 0.25rem; }
</style>
