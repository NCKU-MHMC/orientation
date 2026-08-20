<script setup>
// 圖 B-2:訓練與推論的軌跡分歧。訓練期 prefix 來自 p(資料),推論期來自 q(模型),
// 目標函數從未度量右側軌跡。
import { fs, palette } from './chart-style.js'

const W = 780, H = 268
// 兩條軌跡:同起點,推論軌跡逐步偏離
const train = 'M60,140 C220,120 420,132 720,126'
const infer = 'M60,140 C220,150 420,196 720,236'
</script>

<template>
  <div class="eb">
    <svg :width="W" :height="H" :viewBox="`0 0 ${W} ${H}`">
      <defs>
        <marker id="eb-arr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path d="M0,0L8,4.5L0,9Z" :fill="palette.p" />
        </marker>
        <marker id="eb-arr-q" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path d="M0,0L8,4.5L0,9Z" :fill="palette.q" />
        </marker>
      </defs>
      <text x="60" y="118" :fill="palette.muted" :style="{ fontSize: fs('tick') }">t = 0</text>
      <text x="700" y="100" text-anchor="end" :fill="palette.muted" :style="{ fontSize: fs('tick') }">t 增加 →</text>

      <path :d="train" fill="none" :stroke="palette.p" stroke-width="2.6" marker-end="url(#eb-arr)" />
      <path :d="infer" fill="none" :stroke="palette.q" stroke-width="2.6" stroke-dasharray="8 5"
        marker-end="url(#eb-arr-q)" />

      <!-- 兩軌間距標注 -->
      <line x1="608" y1="132" x2="608" y2="218" :stroke="palette.bad" stroke-width="1.6" stroke-dasharray="4 3" />
      <text x="622" y="180" :fill="palette.bad" :style="{ fontSize: fs('note') }">誤差逐步累積</text>

      <text x="330" y="112" :fill="palette.p" font-weight="600" :style="{ fontSize: fs('label') }">
        訓練:prefix 取自 p(teacher forcing)</text>
      <text x="300" y="228" :fill="palette.q" font-weight="600" :style="{ fontSize: fs('label') }">
        推論:prefix 取自 q(自己生成)</text>
      <circle cx="60" cy="140" r="5.5" :fill="palette.ink" />
      <text x="60" y="164" text-anchor="middle" :fill="palette.muted" :style="{ fontSize: fs('tick') }">同一起點</text>
    </svg>
    <div class="eb-cap">訓練目標只在藍色軌跡上取期望;橘色軌跡上的條件分布從未被度量。</div>
  </div>
</template>

<style scoped>
.eb { text-align: center; }
.eb svg { margin: 0 auto; }
.eb-cap { font-size: 0.85rem; color: #475569; margin-top: 0.2rem; }
</style>
