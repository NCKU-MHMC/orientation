<script setup>
// ③ 第 1 層:ICL 作為隱式貝氏推論。三個面板都是同一個後驗 p(task | prompt),
// 差別只在 prompt 裡放了什麼。memory agent 的理論框架就是中間與右邊的對比。
import { typeScale } from './chart-style.js'

const W = 700, H = 190
const FS = typeScale(W)

const TASK = ['翻譯', '摘要', '情感', '問答', '改寫']
const PANEL = [
  { title: '只有指令', sub: '後驗接近先驗', p: [0.22, 0.2, 0.2, 0.2, 0.18], c: '#8fa0bc' },
  { title: '+ 三個對的示例', sub: '後驗集中到正確任務', p: [0.05, 0.04, 0.82, 0.05, 0.04], c: '#5edfff' },
  { title: '+ 4000 token 無關 context', sub: '證據被稀釋,後驗重新變平', p: [0.14, 0.19, 0.36, 0.17, 0.14], c: '#ff6b9d' },
]

// 三格排滿 700:8 + 2×(218+12) + 218 = 686,右緣留 14 給描邊
const PW = 218, GAP = 12, X0 = 8
const BASE = 142, MAXH = 92
const bx = (pi, i) => X0 + pi * (PW + GAP) + 22 + i * 38
const bh = (v) => v * MAXH
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <g v-for="(pn, pi) in PANEL" :key="pn.title">
      <rect :x="X0 + pi * (PW + GAP)" y="6" :width="PW" :height="H - 24" rx="7"
            fill="#151d2e" :stroke="pn.c" stroke-opacity="0.45" />
      <text :x="X0 + pi * (PW + GAP) + 12" y="26" :fill="pn.c" :font-size="FS.title">{{ pn.title }}</text>
      <text :x="X0 + pi * (PW + GAP) + 12" y="42" fill="#8fa0bc" :font-size="FS.small">{{ pn.sub }}</text>

      <line :x1="X0 + pi * (PW + GAP) + 12" :y1="BASE" :x2="X0 + pi * (PW + GAP) + PW - 12" :y2="BASE"
            stroke="#243350" />

      <g v-for="(v, i) in pn.p" :key="i">
        <rect :x="bx(pi, i)" :y="BASE - bh(v)" width="26" :height="bh(v)"
              :fill="pn.c" fill-opacity="0.35" :stroke="pn.c" />
        <text :x="bx(pi, i) + 13" :y="BASE + 14" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">
          {{ TASK[i] }}
        </text>
      </g>
    </g>

    <text :x="W / 2" :y="H - 4" fill="#ffb454" :font-size="FS.title" text-anchor="middle">
      縱軸為 p(task | prompt);記憶系統的作用是選擇進入這個後驗的證據
    </text>
  </svg>
</template>
