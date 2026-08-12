<script setup>
// 板書 B-2:訓練軌道與推論軌道的分岔。第二堂 DDO 段會回放這張。
import { typeScale } from './chart-style.js'

// H 從 268 壓到 240:這一頁還要放標題與一個說明框,471px 的內容高度剛好卡住。
const W = 700, H = 240
const FS = typeScale(W)
const X = [70, 200, 330, 460, 590]
const TOP = 80
const INF = [80, 104, 130, 156, 180] // 推論軌道逐步下沉
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <defs>
      <marker id="tt-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#8fa0bc" />
      </marker>
    </defs>

    <!-- 落差:虛線 -->
    <g stroke="#8fa0bc" stroke-dasharray="3 3" opacity="0.5">
      <line v-for="i in [1, 2, 3]" :key="'g' + i" :x1="X[i]" :y1="TOP + 9" :x2="X[i]" :y2="INF[i] - 9" />
    </g>

    <!-- 訓練軌道:水平 -->
    <text x="58" y="58" fill="#5edfff" :font-size="FS.title">訓練軌道 · 前綴取自 p_data(teacher forcing)</text>
    <line :x1="X[0]" :y1="TOP" :x2="X[4] + 24" :y2="TOP" stroke="#5edfff" stroke-width="2"
          marker-end="url(#tt-arrow)" opacity="0.85" />
    <circle v-for="(x, i) in X" :key="'t' + i" :cx="x" :cy="TOP" r="7"
            fill="#0c1220" stroke="#5edfff" stroke-width="2" />

    <!-- 推論軌道:逐步下沉 -->
    <path :d="`M${X[0]},${TOP} L${X[1]},${INF[1]} L${X[2]},${INF[2]} L${X[3]},${INF[3]} L${X[4] + 22},${INF[4] + 6}`"
          stroke="#ff6b9d" stroke-width="2" fill="none" marker-end="url(#tt-arrow)" />
    <circle v-for="i in [1, 2, 3, 4]" :key="'i' + i" :cx="X[i]" :cy="INF[i]" r="7"
            fill="#0c1220" stroke="#ff6b9d" stroke-width="2" />
    <text :x="X[4]" y="204" fill="#ff6b9d" :font-size="FS.title" text-anchor="end">
      推論軌道 · 前綴取自模型輸出的 q_θ
    </text>

    <text x="40" y="105" fill="#8fa0bc" :font-size="FS.small">共用起點 x₁</text>
    <text :x="X[3] + 12" y="132" fill="#8fa0bc" :font-size="FS.small">落差逐步累積 = exposure bias</text>

    <!-- 標語 -->
    <rect x="40" y="210" :width="W - 80" height="28" rx="7" fill="#151d2e" stroke="#ffb454" stroke-opacity="0.5" />
    <text :x="W / 2" y="228" fill="#ffb454" :font-size="FS.title" text-anchor="middle">
      訓練 loss 只在上面這條軌道上計算,下面這條從來沒被量過
    </text>
  </svg>
</template>
