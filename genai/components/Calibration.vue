<script setup>
// ⑤:reliability diagram。ECE 是圖上那些垂直落差、以各 bin 的樣本比例加權後的平均,
// 不是另一個獨立的指標。溫度縮放只動 logits 的尺度,不動排序,所以 accuracy 完全不變。
import { typeScale } from './chart-style.js'

const W = 700, H = 212
const FS = typeScale(W)

const CONF = [0.55, 0.65, 0.75, 0.85, 0.95]
const WGT = [0.10, 0.14, 0.20, 0.26, 0.30]
const RAW = [0.48, 0.53, 0.58, 0.64, 0.72]
const CAL = [0.54, 0.63, 0.72, 0.83, 0.92]

const ece = (acc) => acc.reduce((s, a, i) => s + WGT[i] * Math.abs(a - CONF[i]), 0)
const ECE_RAW = ece(RAW), ECE_CAL = ece(CAL)

const X0 = 56, X1 = 316, Y0 = 172, Y1 = 24
const px = (v) => X0 + ((v - 0.5) / 0.5) * (X1 - X0)
const py = (v) => Y0 - ((v - 0.4) / 0.6) * (Y0 - Y1)
const poly = (acc) => acc.map((a, i) => `${px(CONF[i]).toFixed(1)},${py(a).toFixed(1)}`).join(' ')
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <!-- 座標框 -->
    <line :x1="X0" :y1="Y0" :x2="X1" :y2="Y0" stroke="#243350" />
    <line :x1="X0" :y1="Y0" :x2="X0" :y2="Y1" stroke="#243350" />
    <text :x="(X0 + X1) / 2" :y="Y0 + 26" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">
      模型自報的 confidence
    </text>
    <text x="14" :y="(Y0 + Y1) / 2" fill="#8fa0bc" :font-size="FS.small">實際</text>
    <text x="14" :y="(Y0 + Y1) / 2 + 14" fill="#8fa0bc" :font-size="FS.small">accuracy</text>
    <text :x="X0 - 6" :y="Y0 + 4" fill="#8fa0bc" :font-size="FS.small" text-anchor="end">0.4</text>
    <text :x="X0 - 6" :y="Y1 + 4" fill="#8fa0bc" :font-size="FS.small" text-anchor="end">1.0</text>

    <!-- 完美校準線 -->
    <line :x1="px(0.5)" :y1="py(0.5)" :x2="px(1)" :y2="py(1)" stroke="#8fa0bc"
          stroke-dasharray="4 4" opacity="0.6" />
    <text :x="px(0.93)" :y="py(0.99)" fill="#8fa0bc" :font-size="FS.small" text-anchor="end">完美校準</text>

    <!-- 落差 -->
    <line v-for="(a, i) in RAW" :key="'g' + i" :x1="px(CONF[i])" :y1="py(a)"
          :x2="px(CONF[i])" :y2="py(CONF[i])" stroke="#ff6b9d" stroke-width="1.2" opacity="0.55" />

    <polyline :points="poly(RAW)" fill="none" stroke="#ff6b9d" stroke-width="2" />
    <polyline :points="poly(CAL)" fill="none" stroke="#5edfff" stroke-width="2" />
    <circle v-for="(a, i) in RAW" :key="'r' + i" :cx="px(CONF[i])" :cy="py(a)" r="3.5" fill="#ff6b9d" />
    <circle v-for="(a, i) in CAL" :key="'c' + i" :cx="px(CONF[i])" :cy="py(a)" r="3.5" fill="#5edfff" />

    <!-- 右側說明 -->
    <g>
      <rect x="348" y="20" width="338" height="60" rx="7" fill="#151d2e" stroke="#ff6b9d" stroke-opacity="0.5" />
      <text x="362" y="40" fill="#ff6b9d" :font-size="FS.title">原始模型 · ECE = {{ ECE_RAW.toFixed(3) }}</text>
      <text x="362" y="58" fill="#8fa0bc" :font-size="FS.small">整條曲線落在對角線下方 = 系統性過度自信</text>
      <text x="362" y="73" fill="#8fa0bc" :font-size="FS.small">RLHF 之後普遍出現這個方向的偏移</text>

      <rect x="348" y="92" width="338" height="60" rx="7" fill="#151d2e" stroke="#5edfff" stroke-opacity="0.5" />
      <text x="362" y="112" fill="#5edfff" :font-size="FS.title">溫度縮放後 · ECE = {{ ECE_CAL.toFixed(3) }}</text>
      <text x="362" y="130" fill="#8fa0bc" :font-size="FS.small">只在驗證集上擬合一個純量 T,logits 全體除以它</text>
      <text x="362" y="145" fill="#8fa0bc" :font-size="FS.small">排序不變,accuracy 不受影響</text>

      <text x="348" y="176" fill="#ffb454" :font-size="FS.title">
        ECE = 各 bin 落差以樣本比例加權的平均
      </text>
      <text x="348" y="194" fill="#8fa0bc" :font-size="FS.small">
        ECE 低不蘊含區辨力高;它與 accuracy 相互獨立
      </text>
    </g>
  </svg>
</template>
