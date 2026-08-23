<script setup>
// 介面契約的動作示範:sample() 由密度抽點落到軸上,logprob(x) 由軸上的 x 讀曲線高度。
import { ref, onMounted, onBeforeUnmount } from 'vue'
import gsap from 'gsap'
import { fs, palette } from './chart-style.js'
import { gauss, mix } from './divergence-math.js'

const P = [[0.6, -1.6, 0.75], [0.4, 1.8, 0.6]]
const W = 620, H = 250, L = 34, R = 16, T = 46, B = 38
const AX = H - B
const yMax = 0.45
const sx = (x) => L + ((x + 5) / 10) * (W - L - R)
const sy = (y) => AX - (y / yMax) * (AX - T)

let d = ''
for (let x = -5; x <= 5.001; x += 0.05)
  d += `${d ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(mix(x, P)).toFixed(1)}`

// 依密度取的六個樣本(固定值,講述時每次播放一致)
const samples = [-2.3, -1.5, 1.9, -0.8, 1.6, -2.0].map((x) => ({ x, y: sy(mix(x, P)) }))

// 被查詢的點:不落在任何樣本上,說明任意 x 都能問
const qx = -0.6
const qLog = Math.log(mix(qx, P)).toFixed(2)

const root = ref(null)
const phase = ref('sample')
let ctx

onMounted(() => {
  ctx = gsap.context((self) => {
    const dots = self.selector('.s-dot')
    const ticks = self.selector('.s-tick')
    const tl = gsap.timeline({ repeat: -1, repeatDelay: 0.8, defaults: { ease: 'power2.out' } })
    tl.call(() => (phase.value = 'sample'))
      .set('.g-log', { opacity: 0 })
      .set('.g-sample', { opacity: 1 })
    dots.forEach((el, i) => {
      const t = 0.2 + i * 0.42
      tl.fromTo(el, { opacity: 0, attr: { cy: samples[i].y } }, { opacity: 1, duration: 0.2 }, t)
        .to(el, { attr: { cy: AX }, duration: 0.45, ease: 'power1.in' }, t + 0.2)
        .fromTo(ticks[i], { opacity: 0 }, { opacity: 1, duration: 0.2 }, t + 0.62)
        .to(el, { opacity: 0.35, duration: 0.3 }, t + 0.65)
    })
    tl.to('.g-sample', { opacity: 0.28, duration: 0.4 }, '+=0.6')
      .call(() => (phase.value = 'logprob'))
      .set('.g-log', { opacity: 1 })
      .fromTo('.q-mark', { opacity: 0 }, { opacity: 1, duration: 0.3 })
      .fromTo('.q-vline', { attr: { y2: AX } }, { attr: { y2: sy(mix(qx, P)) }, duration: 0.55 })
      .fromTo('.q-dot', { opacity: 0 }, { opacity: 1, duration: 0.25 })
      .fromTo('.q-hline', { attr: { x2: sx(qx) } }, { attr: { x2: L }, duration: 0.45 })
      .fromTo('.q-read', { opacity: 0 }, { opacity: 1, duration: 0.3 })
      .to({}, { duration: 1.6 })
      .to('.g-sample, .g-log', { opacity: 0, duration: 0.4 })
  }, root.value)
})
onBeforeUnmount(() => ctx && ctx.revert())
</script>

<template>
  <div class="canim" ref="root">
    <svg :width="W" :height="H" :viewBox="`0 0 ${W} ${H}`">
      <text :x="L" y="22" :fill="palette.muted" :style="{ fontSize: fs('note') }">
        p 的密度曲線(模型內部持有的函數)
      </text>
      <line :x1="L" :y1="AX" :x2="W - R" :y2="AX" :stroke="palette.grid" stroke-width="1.5" />
      <path :d="d" :stroke="palette.p" stroke-width="2.2" fill="none" />

      <g class="g-sample">
        <line v-for="(s, i) in samples" :key="`t${i}`" class="s-tick" :x1="sx(s.x)" :y1="AX - 7"
          :x2="sx(s.x)" :y2="AX + 7" :stroke="palette.q" stroke-width="2" :style="{ opacity: 0 }" />
        <circle v-for="(s, i) in samples" :key="`d${i}`" class="s-dot" :cx="sx(s.x)" :cy="s.y" r="5"
          :fill="palette.q" :style="{ opacity: 0 }" />
      </g>

      <g class="g-log">
        <line class="q-vline" :x1="sx(qx)" :y1="AX" :x2="sx(qx)" :y2="AX" :stroke="palette.q"
          stroke-width="1.6" stroke-dasharray="4 3" />
        <line class="q-hline" :x1="sx(qx)" :y1="sy(mix(qx, P))" :x2="sx(qx)" :y2="sy(mix(qx, P))"
          :stroke="palette.q" stroke-width="1.6" stroke-dasharray="4 3" />
        <circle class="q-dot" :cx="sx(qx)" :cy="sy(mix(qx, P))" r="5" :fill="palette.q"
          :style="{ opacity: 0 }" />
        <g class="q-mark" :style="{ opacity: 0 }">
          <polygon :points="`${sx(qx)},${AX - 6} ${sx(qx) - 6},${AX + 8} ${sx(qx) + 6},${AX + 8}`"
            :fill="palette.q" />
          <text :x="sx(qx)" :y="AX + 26" text-anchor="middle" :fill="palette.ink"
            :style="{ fontSize: fs('tick') }">給定的 x</text>
        </g>
        <text class="q-read" :x="L + 6" :y="sy(mix(qx, P)) - 10" :fill="palette.ink"
          :style="{ fontSize: fs('label'), opacity: 0 }">log p(x) = {{ qLog }}</text>
      </g>
    </svg>
    <div class="cap">
      <span :class="{ on: phase === 'sample' }">
        <code>p.sample()</code> 抽出一個 x,落點的疏密正比於密度
      </span>
      <span :class="{ on: phase === 'logprob' }">
        <code>p.logprob(x)</code> 給定一個 x,回報該處密度的對數
      </span>
    </div>
  </div>
</template>

<style scoped>
.canim { text-align: center; }
.canim svg { margin: 0 auto; }
.cap {
  display: flex;
  justify-content: center;
  gap: 2.2rem;
  margin-top: 0.3rem;
  font-size: 0.88rem;
  color: var(--muted);
  transition: color 0.3s;
}
.cap span { opacity: 0.45; transition: opacity 0.3s; }
.cap span.on { opacity: 1; color: var(--ink-2); }
.cap code {
  background: var(--paper-3);
  border-radius: 4px;
  padding: 0.1rem 0.4rem;
}
</style>
