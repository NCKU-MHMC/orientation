<script setup>
// 圖 B-3:DDO 機制。真實樣本與參考模型樣本進入隱式判別器,虛線為 self-play 回饋。
import { fs, palette } from './chart-style.js'
</script>

<template>
  <div class="ddo">
    <svg width="860" height="300" viewBox="0 0 860 300">
      <defs>
        <marker id="ddo-arr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path d="M0,0L8,4.5L0,9Z" :fill="palette.muted" />
        </marker>
        <marker id="ddo-arr-b" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path d="M0,0L8,4.5L0,9Z" :fill="palette.bad" />
        </marker>
      </defs>

      <!-- 左:兩個樣本來源 -->
      <rect x="30" y="40" width="215" height="64" rx="9" :fill="palette.soft" :stroke="palette.p" stroke-width="1.8" />
      <text x="137" y="66" text-anchor="middle" :fill="palette.p" font-weight="600"
        :style="{ fontSize: fs('label') }">真實樣本</text>
      <text x="137" y="88" text-anchor="middle" :fill="palette.muted"
        :style="{ fontSize: fs('tick') }">x ~ p_data(標記 1)</text>

      <rect x="30" y="180" width="215" height="64" rx="9" :fill="palette.soft" :stroke="palette.q" stroke-width="1.8" />
      <text x="137" y="206" text-anchor="middle" :fill="palette.q" font-weight="600"
        :style="{ fontSize: fs('label') }">參考模型樣本</text>
      <text x="137" y="228" text-anchor="middle" :fill="palette.muted"
        :style="{ fontSize: fs('tick') }">x ~ p_ref(標記 0)</text>

      <!-- 中:隱式判別器 -->
      <rect x="330" y="102" width="250" height="80" rx="9" fill="#fff" :stroke="palette.ink" stroke-width="2" />
      <text x="455" y="132" text-anchor="middle" :fill="palette.ink" font-weight="600"
        :style="{ fontSize: fs('label') }">隱式判別器</text>
      <text x="455" y="158" text-anchor="middle" :fill="palette.ink"
        :style="{ fontSize: fs('note') }">d(x) = σ( β · log pθ(x) / p_ref(x) )</text>

      <!-- 右:損失 -->
      <rect x="665" y="112" width="165" height="60" rx="9" :fill="palette.soft" :stroke="palette.accent" stroke-width="1.8" />
      <text x="747" y="138" text-anchor="middle" :fill="palette.accent" font-weight="600"
        :style="{ fontSize: fs('label') }">判別損失</text>
      <text x="747" y="158" text-anchor="middle" :fill="palette.muted"
        :style="{ fontSize: fs('tick') }">標準 BCE</text>

      <line x1="245" y1="72" x2="322" y2="118" :stroke="palette.muted" stroke-width="1.8" marker-end="url(#ddo-arr)" />
      <line x1="245" y1="212" x2="322" y2="166" :stroke="palette.muted" stroke-width="1.8" marker-end="url(#ddo-arr)" />
      <line x1="580" y1="142" x2="657" y2="142" :stroke="palette.muted" stroke-width="1.8" marker-end="url(#ddo-arr)" />

      <!-- 虛線:self-play 回饋(pθ 定期成為新的 p_ref) -->
      <path d="M455,182 C455,268 240,268 160,248" fill="none" :stroke="palette.bad" stroke-width="1.8"
        stroke-dasharray="7 5" marker-end="url(#ddo-arr-b)" />
      <text x="430" y="284" text-anchor="middle" :fill="palette.bad"
        :style="{ fontSize: fs('note') }">self-play:訓練後的 pθ 成為下一輪的 p_ref</text>

      <text x="455" y="86" text-anchor="middle" :fill="palette.muted"
        :style="{ fontSize: fs('note') }">判別器由 pθ 的 logprob 直接組成,沒有獨立網路</text>
    </svg>
  </div>
</template>

<style scoped>
.ddo { text-align: center; }
.ddo svg { margin: 0 auto; }
</style>
