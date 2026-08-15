<script setup>
// ⑤:predictive entropy 與 semantic entropy 測的不是同一件事。
// 同樣 10 個樣本,左邊按 token 序列算、右邊先按語意聚類再算,結論相反。
import { typeScale } from './chart-style.js'

const W = 700, H = 218
const FS = typeScale(W)

const SAMPLES = [
  { s: '巴黎', k: 0 }, { s: '法國巴黎', k: 0 }, { s: '是巴黎', k: 0 },
  { s: 'Paris', k: 0 }, { s: '巴黎市', k: 0 }, { s: '首都是巴黎', k: 0 },
  { s: '巴黎。', k: 0 }, { s: '里昂', k: 1 }, { s: '法國里昂', k: 1 },
  { s: '應該是里昂', k: 1 },
]
const CK = ['#5edfff', '#ff6b9d']

// 左:每個序列各自一個類別 → 均勻 → H = log 10
const HTOK = Math.log(10)
// 右:語意聚類後 7/3 → H = −(.7 log .7 + .3 log .3)
const HSEM = -(0.7 * Math.log(0.7) + 0.3 * Math.log(0.3))

const LX = 16, RX = 362, PW = 322
const cell = (i) => ({ x: 14 + (i % 4) * 76, y: 66 + Math.floor(i / 4) * 30 })
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <!-- 左:token 層級 -->
    <rect :x="LX" y="8" :width="PW" height="196" rx="7" fill="#151d2e" stroke="#8fa0bc" stroke-opacity="0.35" />
    <text :x="LX + 14" y="30" fill="#e8edf6" :font-size="FS.title">predictive entropy · 按 token 序列</text>
    <text :x="LX + 14" y="48" fill="#8fa0bc" :font-size="FS.small">10 個樣本 = 10 個相異序列 → 視為 10 個結果</text>
    <g v-for="(sm, i) in SAMPLES" :key="'l' + i">
      <rect :x="LX + cell(i).x" :y="cell(i).y" width="68" height="22" rx="11"
            fill="#0c1220" stroke="#8fa0bc" stroke-opacity="0.45" />
      <text :x="LX + cell(i).x + 34" :y="cell(i).y + 15" fill="#8fa0bc"
            :font-size="FS.small" text-anchor="middle">{{ sm.s }}</text>
    </g>
    <text :x="LX + 14" y="192" fill="#ffb454" :font-size="FS.title">
      H = log 10 = {{ HTOK.toFixed(2) }} nats(高不確定性)
    </text>

    <!-- 右:語意層級 -->
    <rect :x="RX" y="8" :width="PW" height="196" rx="7" fill="#151d2e" stroke="#b48cff" stroke-opacity="0.5" />
    <text :x="RX + 14" y="30" fill="#e8edf6" :font-size="FS.title">semantic entropy · 先聚類再算</text>
    <text :x="RX + 14" y="48" fill="#8fa0bc" :font-size="FS.small">雙向蘊涵判定 → 只剩兩個語意類別</text>
    <g v-for="(sm, i) in SAMPLES" :key="'r' + i">
      <rect :x="RX + cell(i).x" :y="cell(i).y" width="68" height="22" rx="11"
            fill="#0c1220" :stroke="CK[sm.k]" stroke-opacity="0.8" />
      <text :x="RX + cell(i).x + 34" :y="cell(i).y + 15" :fill="CK[sm.k]"
            :font-size="FS.small" text-anchor="middle">{{ sm.s }}</text>
    </g>
    <text :x="RX + 14" y="192" fill="#ffb454" :font-size="FS.title">
      H = {{ HSEM.toFixed(2) }} nats(7 : 3,低不確定性)
    </text>

    <text :x="W / 2" :y="H - 2" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">
      同一批樣本與同一個公式,差別在於「同一個結果」的判定方式
    </text>
  </svg>
</template>
