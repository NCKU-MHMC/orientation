<script setup>
// JSD 為什麼會梯度消失:把兩個分布拉開,看兩個散度各自怎麼變。
import { computed, ref } from 'vue'
import { XS, gauss, divergences, separationCurve } from './divergence-math.js'
import { CONTENT, typeScale } from './chart-style.js'

const d = ref(1.2)
const curve = separationCurve()

// 兩張圖並排(flex gap-3 = 12px),各佔內容區的一半。
// 兩邊 viewBox 寬度必須相同,否則同一個 font-size 會被縮放成不同大小。
const VB_W = 340, VB_H = 150
const HALF = (CONTENT.w - 12) / 2
const FS = typeScale(VB_W, HALF)

// 左圖:兩個高斯
const lx = (x) => 10 + ((x + 4) / 8) * (VB_W - 20)
const ly = (y) => VB_H - 22 - (y / 0.85) * (VB_H - 40)
const line = (mu) =>
  XS.filter((x) => x >= -4 && x <= 4)
    .map((x, i) => `${i ? 'L' : 'M'}${lx(x).toFixed(1)},${ly(gauss(x, mu, 0.5)).toFixed(1)}`)
    .join('')
const pPath = computed(() => line(-d.value / 2))
const qPath = computed(() => line(d.value / 2))

// 右圖:散度 vs 距離,y 軸取對數。
//
// 為什麼要對數:KL 從 2e-2 一路到 26.9,JSD 從 5e-3 到 0.693 就停住。
// 這是三個數量級以上的跨度,線性刻度只能二選一 —— 要嘛把 KL 削頂(原本的做法,
// 一削就看不出「KL 還在漲」),要嘛讓 JSD 整條貼在底線上看不出它先漲後平。
// 取對數兩條都看得見,而且「變平 = 梯度為 0」在任何刻度下都還是一條水平線。
const VMIN = 1e-3, VMAX = 30
const PLOT_T = 26, PLOT_B = VB_H - 22
const SPAN = Math.log10(VMAX) - Math.log10(VMIN)
const rx = (v) => 34 + (v / 6) * (VB_W - 46)
const ry = (v) => {
  const c = Math.min(Math.max(v, VMIN), VMAX)
  return PLOT_B - ((Math.log10(c) - Math.log10(VMIN)) / SPAN) * (PLOT_B - PLOT_T)
}
const TICKS = [
  { v: 0.001, t: '0.001' },
  { v: 0.01, t: '0.01' },
  { v: 0.1, t: '0.1' },
  { v: 1, t: '1' },
  { v: 10, t: '10' },
]

// 濾掉 d=0(兩個散度都恰好是 0,對數下沒有位置)。
// 不濾的話會在左緣多出一條假的垂直線。
const trace = (key) =>
  curve
    .filter((c) => c[key] > 0)
    .map((c, i) => `${i ? 'L' : 'M'}${rx(c.d).toFixed(1)},${ry(c[key]).toFixed(1)}`)
    .join('')
const traces = [
  { key: 'forward', c: '#5edfff', label: 'KL' },
  { key: 'jsd', c: '#ffb454', label: 'JSD' },
]
const now = computed(() => {
  const p = XS.map((x) => gauss(x, -d.value / 2, 0.5))
  const q = XS.map((x) => gauss(x, d.value / 2, 0.5))
  return divergences(p, q)
})
const LOG2 = Math.LN2
</script>

<template>
  <div>
    <div class="flex gap-3 items-start">
      <svg :viewBox="`0 0 ${VB_W} ${VB_H}`" class="flex-1">
        <path :d="pPath" stroke="#b48cff" stroke-width="2.2" fill="#b48cff" fill-opacity="0.12" />
        <path :d="qPath" stroke="#ff6b9d" stroke-width="2.2" fill="#ff6b9d" fill-opacity="0.12" />
        <line x1="10" :y1="ly(0)" :x2="VB_W - 10" :y2="ly(0)" stroke="#243350" />
        <text x="12" y="16" fill="#8fa0bc" :font-size="FS.label">兩個分布,距離 d</text>
      </svg>

      <svg :viewBox="`0 0 ${VB_W} ${VB_H}`" class="flex-1">
        <!-- 十倍刻度線 -->
        <g v-for="t in TICKS" :key="t.v">
          <line :x1="rx(0)" :y1="ry(t.v)" :x2="rx(6)" :y2="ry(t.v)" stroke="#1b2740" stroke-width="1" />
          <text x="30" :y="ry(t.v) + 3" fill="#8fa0bc" :font-size="FS.small" text-anchor="end" opacity="0.8">
            {{ t.t }}
          </text>
        </g>

        <!-- JSD 的上界 -->
        <line :x1="rx(0)" :y1="ry(LOG2)" :x2="rx(6)" :y2="ry(LOG2)"
              stroke="#ffb454" stroke-width="1" stroke-dasharray="4 4" opacity="0.7" />
        <text :x="rx(6)" :y="ry(LOG2) - 4" fill="#ffb454" :font-size="FS.small" text-anchor="end">log 2 上界</text>

        <path v-for="t in traces" :key="t.key" :d="trace(t.key)" :stroke="t.c" stroke-width="2.2" fill="none" />

        <!-- 目前的 d -->
        <line :x1="rx(d)" :y1="PLOT_T" :x2="rx(d)" :y2="PLOT_B" stroke="#e8edf6" stroke-width="1" opacity="0.35" />
        <circle v-for="t in traces" :key="'p' + t.key" :cx="rx(d)" :cy="ry(now[t.key])" r="3" :fill="t.c" />

        <line :x1="rx(0)" :y1="PLOT_B" :x2="rx(6)" :y2="PLOT_B" stroke="#243350" />
        <text x="6" y="14" fill="#8fa0bc" :font-size="FS.small">散度值 · 對數刻度</text>
        <text :x="rx(6)" :y="VB_H - 6" fill="#8fa0bc" :font-size="FS.small" text-anchor="end">距離 d →</text>
      </svg>
    </div>

    <div class="flex items-center gap-4 mt-1 text-xs" style="font-family: var(--mono)">
      <input type="range" min="0" max="6" step="0.05" v-model.number="d" class="flex-1" />
      <span style="color: var(--muted)">d = {{ d.toFixed(2) }}</span>
      <span style="color: #5edfff">KL = {{ now.forward.toFixed(2) }}</span>
      <span style="color: #ffb454">JSD = {{ now.jsd.toFixed(4) }}</span>
      <span style="color: var(--muted)">
        KL / JSD = {{ now.jsd > 1e-6 ? (now.forward / now.jsd).toFixed(1) + '×' : '—' }}
      </span>
    </div>
  </div>
</template>
