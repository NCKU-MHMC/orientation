<script setup>
// temperature 與 top-p 對 token 分布的作用:連續攤平/銳化 vs 硬截斷。
import { fs, palette } from './chart-style.js'
import { softmax } from './divergence-math.js'

const logits = [3.1, 2.4, 1.9, 1.2, 0.8, 0.3, -0.2, -0.8]
const labels = ['t₁', 't₂', 't₃', 't₄', 't₅', 't₆', 't₇', 't₈']

const panels = [
  { title: 'T = 1', probs: softmax(logits, 1), cut: -1 },
  { title: 'T = 2(攤平)', probs: softmax(logits, 2), cut: -1 },
  { title: 'T = 0.5(銳化)', probs: softmax(logits, 0.5), cut: -1 },
  { title: 'top-p = 0.8(截斷)', probs: topP(softmax(logits, 1), 0.8), cut: cutIndex(softmax(logits, 1), 0.8) },
]

function cutIndex(p, thr) {
  let acc = 0
  for (let i = 0; i < p.length; i++) {
    acc += p[i]
    if (acc >= thr) return i
  }
  return p.length - 1
}
function topP(p, thr) {
  const k = cutIndex(p, thr)
  const Z = p.slice(0, k + 1).reduce((a, b) => a + b, 0)
  return p.map((v, i) => (i <= k ? v / Z : 0))
}

const W = 225, H = 170, PAD = 8
const bw = (W - 2 * PAD) / logits.length
</script>

<template>
  <div class="ttp-row">
    <div v-for="pn in panels" :key="pn.title" class="ttp-panel">
      <svg :width="W" :height="H" :viewBox="`0 0 ${W} ${H}`">
        <text :x="W / 2" :y="16" text-anchor="middle" :fill="palette.ink" font-weight="600"
          :style="{ fontSize: fs('tick') }">{{ pn.title }}</text>
        <g v-for="(p, i) in pn.probs" :key="i">
          <rect :x="PAD + i * bw + 2" :y="H - 22 - p * 210" :width="bw - 4" :height="p * 210"
            :fill="pn.cut >= 0 && i > pn.cut ? palette.grid : palette.p" rx="2" />
          <text :x="PAD + i * bw + bw / 2" :y="H - 8" text-anchor="middle" :fill="palette.muted"
            :style="{ fontSize: fs('tick') }">{{ labels[i] }}</text>
        </g>
        <line v-if="pn.cut >= 0" :x1="PAD + (pn.cut + 1) * bw" :y1="H - 100" :x2="PAD + (pn.cut + 1) * bw"
          :y2="H - 20" :stroke="palette.bad" stroke-width="1.5" stroke-dasharray="4 3" />
      </svg>
    </div>
  </div>
  <div class="ttp-cap">同一組 logits。temperature 連續改變熵;top-p 把尾部質量直接移除後再正規化。</div>
</template>

<style scoped>
.ttp-row { display: flex; gap: 0.4rem; justify-content: center; }
.ttp-cap { text-align: center; font-size: 0.85rem; color: var(--ink-2); margin-top: 0.3rem; }
</style>
