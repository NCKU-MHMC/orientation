<script setup>
// 第二堂的目錄:六層介入點,各自釘在生成流程的哪一段。
// 層號是「介入成本」的順序,不是 pipeline 的順序 —— 第 2 層(取樣)實際發生在第 3 層(logits)之後。
import { typeScale } from './chart-style.js'

const props = defineProps({
  active: { type: Array, default: () => [1, 2, 3, 4, 5, 6] }, // 要點亮的層
})

const W = 700, H = 232
const FS = typeScale(W)

const BOX = [
  { x: 20, w: 78, label: '條件 c' },
  { x: 128, w: 96, label: 'p_θ 權重', tall: true },
  { x: 254, w: 76, label: 'logits' },
  { x: 360, w: 70, label: '取樣' },
  { x: 460, w: 82, label: '樣本 × n' },
  { x: 572, w: 70, label: '聚合' },
]
const Y = 138, BH = 34

// 上方掛的層:連到 pipeline 的某一格
const ABOVE = [
  { n: 1, at: 59, y: 58, text: '改條件', sub: 'prompt · few-shot · RAG · memory', c: '#5edfff' },
  { n: 3, at: 292, y: 92, text: '改 logits', sub: 'constrained · contrastive · CFG', c: '#b48cff' },
  { n: 2, at: 395, y: 58, text: '改取樣', sub: 'T · top-k/p · min-p · beam', c: '#ffb454' },
  { n: 4, at: 607, y: 92, text: '改聚合', sub: 'self-consistency · MBR · best-of-n', c: '#ff6b9d' },
]
// 下方掛的層:都指向權重那一格
const BELOW = [
  { n: 5, at: 128, text: '改權重', sub: 'SFT · LoRA · RLHF / DPO · DDO', c: '#ff6b9d' },
  { n: 6, at: 400, text: '改表徵', sub: 'activation steering', c: '#b48cff' },
]
const on = (n) => props.active.includes(n)
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <defs>
      <marker id="gl-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#8fa0bc" />
      </marker>
    </defs>

    <!-- pipeline 主軸 -->
    <g>
      <line v-for="(b, i) in BOX.slice(0, 5)" :key="'a' + i"
            :x1="b.x + b.w" :y1="Y + BH / 2" :x2="BOX[i + 1].x - 4" :y2="Y + BH / 2"
            stroke="#8fa0bc" stroke-width="1.4" marker-end="url(#gl-arrow)" opacity="0.7" />
      <line :x1="642" :y1="Y + BH / 2" :x2="680" :y2="Y + BH / 2"
            stroke="#8fa0bc" stroke-width="1.4" marker-end="url(#gl-arrow)" opacity="0.7" />
    </g>

    <g v-for="b in BOX" :key="b.label">
      <rect :x="b.x" :y="b.tall ? Y - 8 : Y" :width="b.w" :height="b.tall ? BH + 16 : BH"
            rx="6" fill="#151d2e" :stroke="b.tall ? '#b48cff' : '#243350'" stroke-width="1.4" />
      <text :x="b.x + b.w / 2" :y="Y + BH / 2 + 5" fill="#e8edf6" :font-size="FS.label" text-anchor="middle">
        {{ b.label }}
      </text>
    </g>

    <!-- 上方四層 -->
    <g v-for="a in ABOVE" :key="'ab' + a.n" :opacity="on(a.n) ? 1 : 0.22">
      <line :x1="a.at" :y1="a.y + 26" :x2="a.at" :y2="Y - (a.n === 3 ? 2 : 2)"
            :stroke="a.c" stroke-width="1.2" stroke-dasharray="3 3" />
      <circle :cx="a.at" :cy="a.y" r="10" fill="#0c1220" :stroke="a.c" stroke-width="1.6" />
      <text :x="a.at" :y="a.y + 4" :fill="a.c" :font-size="FS.small" text-anchor="middle">{{ a.n }}</text>
      <text :x="a.at + 16" :y="a.y - 1" :fill="a.c" :font-size="FS.title">{{ a.text }}</text>
      <text :x="a.at + 16" :y="a.y + 13" fill="#8fa0bc" :font-size="FS.small">{{ a.sub }}</text>
    </g>

    <!-- 下方兩層:都指向權重 -->
    <g v-for="b in BELOW" :key="'be' + b.n" :opacity="on(b.n) ? 1 : 0.22">
      <path :d="`M${b.at},198 L${b.at},186 L176,186 L176,${Y + BH + 10}`"
            :stroke="b.c" stroke-width="1.2" stroke-dasharray="3 3" fill="none" />
      <circle :cx="b.at" cy="208" r="10" fill="#0c1220" :stroke="b.c" stroke-width="1.6" />
      <text :x="b.at" y="212" :fill="b.c" :font-size="FS.small" text-anchor="middle">{{ b.n }}</text>
      <text :x="b.at + 16" y="205" :fill="b.c" :font-size="FS.title">{{ b.text }}</text>
      <text :x="b.at + 16" y="219" fill="#8fa0bc" :font-size="FS.small">{{ b.sub }}</text>
    </g>

    <text x="20" y="18" fill="#8fa0bc" :font-size="FS.small">
      編號依介入成本排序;取樣(2)在 logits(3)計算之後才執行
    </text>
  </svg>
</template>
