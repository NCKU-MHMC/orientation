<script setup>
// VAE 的四個缺陷不是四件獨立的事,它們各自可以追回 ELBO 的某一項。
// 講法:先指式子,再指缺陷。
import { typeScale } from './chart-style.js'

const W = 700, H = 232
const FS = typeScale(W)

const TERM = [
  { x: 62, w: 210, c: '#5edfff', t: '重建項', s: 'E_q[ log p_θ(x | z) ]' },
  { x: 400, w: 210, c: '#ffb454', t: '正則項', s: 'KL( q_φ(z|x) ‖ p(z) )' },
]
const CARD = [
  { x: 8, c: '#5edfff', n: '① 樣本糊', from: 62 + 105,
    s: '高斯 likelihood = MSE。一個 z 對應多個合理輸出時,最優解是平均。' },
  { x: 180, c: '#5edfff', n: '② 下界間隙', from: 62 + 105,
    s: '優化的是 ELBO 不是 log p(x)。q 家族太簡單,間隙就大。' },
  { x: 352, c: '#ffb454', n: '③ posterior collapse', from: 400 + 105,
    s: 'decoder 夠強時 q 塌回先驗,z 被忽略,KL 項歸零。' },
  { x: 524, c: '#ffb454', n: '④ prior hole', from: 400 + 105,
    s: '逐樣本壓向先驗,不保證整團 q(z) 蓋滿先驗。採樣踩到洞就生出垃圾。' },
]
const CW = 168, CY = 108
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <defs>
      <marker id="vd-a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#8fa0bc" />
      </marker>
    </defs>

    <!-- ELBO 的兩項 -->
    <text x="14" y="34" fill="#b48cff" :font-size="FS.title">ELBO =</text>
    <text x="330" y="34" fill="#e8edf6" :font-size="FS.title" text-anchor="middle">−</text>
    <g v-for="t in TERM" :key="t.t">
      <rect :x="t.x" y="12" :width="t.w" height="42" rx="7"
            :fill="t.c" fill-opacity="0.12" :stroke="t.c" stroke-width="1.7" />
      <text :x="t.x + t.w / 2" y="30" :fill="t.c" :font-size="FS.title" text-anchor="middle">{{ t.t }}</text>
      <text :x="t.x + t.w / 2" y="46" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">{{ t.s }}</text>
    </g>

    <!-- 連到各自造成的缺陷 -->
    <path v-for="c in CARD" :key="'p' + c.n"
          :d="`M${c.from},54 V${CY - 22} H${c.x + CW / 2} V${CY - 4}`"
          fill="none" stroke="#8fa0bc" stroke-width="1.2" opacity="0.7" marker-end="url(#vd-a)" />

    <!-- 缺陷卡 -->
    <g v-for="c in CARD" :key="c.n">
      <rect :x="c.x" :y="CY" :width="CW" height="96" rx="7"
            fill="#151d2e" :stroke="c.c" stroke-opacity="0.55" stroke-width="1.4" />
      <text :x="c.x + 10" :y="CY + 20" :fill="c.c" :font-size="FS.title">{{ c.n }}</text>
      <foreignObject :x="c.x + 8" :y="CY + 28" :width="CW - 16" height="64">
        <div xmlns="http://www.w3.org/1999/xhtml" class="d">{{ c.s }}</div>
      </foreignObject>
    </g>

    <text x="14" :y="H - 6" fill="#ffb454" :font-size="FS.small">
      ①② 來自「用什麼衡量重建」,③④ 來自「怎麼壓潛在分布」。改進的前提是先分清楚問題出在哪一項。
    </text>
  </svg>
</template>

<style scoped>
.d {
  font-size: 9.3px;         /* = 11.5px ÷ (868/700):與其他圖的 small 同一個實際大小 */
  line-height: 1.45;
  color: #8fa0bc;
  font-family: 'Noto Sans TC', sans-serif;
}
</style>
