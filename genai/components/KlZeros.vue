<script setup>
// forward / reverse KL 的懲罰位置:積分權重決定哪一側的錯誤被計入。
import { fs, palette } from './chart-style.js'
import { gauss, mix, pathOf } from './divergence-math.js'

const props = defineProps({
  mode: { type: String, default: 'forward' }, // 'forward' | 'reverse'
})

const P = [[0.5, -2, 0.6], [0.5, 2, 0.6]]
const xs = []
for (let x = -5; x <= 5.001; x += 0.05) xs.push(x)
const W = 620, H = 230, PAD = 14
const pYs = xs.map((x) => mix(x, P))
// forward:q 只蓋右峰 → 左峰處 p>0, q≈0,懲罰無界
// reverse:同一個 q → 左峰被忽略,無懲罰
const qYs = xs.map((x) => gauss(x, 2, 0.7))
// 上緣取兩條曲線的最大值加一成半留白,免得較窄的 q 峰頂被裁掉
const yMax = Math.max(...pYs, ...qYs) * 1.15
const pPath = pathOf(xs, pYs, -5, 5, 0, yMax, W, H, PAD)
const qPath = pathOf(xs, qYs, -5, 5, 0, yMax, W, H, PAD)
const sx = (x) => PAD + ((x + 5) / 10) * (W - 2 * PAD)
const isFwd = props.mode === 'forward'
</script>

<template>
  <div class="klz">
    <svg :width="W" :height="H" :viewBox="`0 0 ${W} ${H}`">
      <line :x1="PAD" :y1="H - PAD" :x2="W - PAD" :y2="H - PAD" :stroke="palette.grid" stroke-width="1.5" />
      <rect v-if="isFwd" :x="sx(-3.6)" y="18" :width="sx(-0.4) - sx(-3.6)" :height="H - PAD - 18"
        :fill="palette.bad" fill-opacity="0.1" />
      <path :d="pPath" :stroke="palette.p" stroke-width="2.2" fill="none" />
      <path :d="qPath" :stroke="palette.q" stroke-width="2.2" fill="none" stroke-dasharray="none" />
      <text :x="sx(-2)" :y="34" text-anchor="middle" :fill="isFwd ? palette.bad : palette.muted"
        :style="{ fontSize: fs('note') }">
        {{ isFwd ? 'p>0 而 q→0：懲罰無界' : 'p 的這一峰被 q 忽略：無懲罰' }}
      </text>
      <text :x="sx(-2)" :y="H - 24" text-anchor="middle" :fill="palette.p" :style="{ fontSize: fs('tick') }">p 的左峰</text>
      <text :x="sx(2.9)" :y="60" text-anchor="start" :fill="palette.q" :style="{ fontSize: fs('tick') }">q</text>
    </svg>
    <div class="klz-cap">
      <template v-if="isFwd">積分由 p 加權：p 有質量而 q 沒有的區域主導損失，q 被迫覆蓋 p 的整個支撐集(zero-avoiding)</template>
      <template v-else>積分由 q 加權：q 沒去的地方不進入積分，丟掉整個峰不付代價(zero-forcing)</template>
    </div>
  </div>
</template>

<style scoped>
.klz { text-align: center; }
.klz svg { margin: 0 auto; }
.klz-cap { font-size: 0.88rem; color: var(--ink-2); margin-top: 0.3rem; }
</style>
