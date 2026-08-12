<script setup>
// DPM / Flow:同一個 forward KL,換一個方向做鏈鎖分解。
import { typeScale } from './chart-style.js'

const W = 700, H = 175
const FS = typeScale(W)
const seq = [0, 1, 2, 3, 4]
const noise = [0, 1, 2, 3, 4]
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <defs>
      <marker id="da-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#8fa0bc" />
      </marker>
    </defs>

    <!-- 左:沿序列拆(AR) -->
    <g>
      <text x="20" y="20" fill="#5edfff" :font-size="FS.title">AR:沿「序列」拆</text>
      <line x1="30" y1="100" x2="300" y2="100" stroke="#8fa0bc" stroke-width="1.4" marker-end="url(#da-arrow)" />
      <g v-for="i in seq" :key="'s' + i">
        <rect :x="34 + i * 52" y="76" width="40" height="26" rx="5"
              fill="#0c1220" stroke="#5edfff" stroke-width="1.6" />
        <text :x="54 + i * 52" y="94" fill="#e8edf6" :font-size="FS.label" text-anchor="middle">x{{ i + 1 }}</text>
      </g>
      <text x="30" y="130" fill="#8fa0bc" :font-size="FS.small">每一步:預測下一個 token</text>
      <text x="30" y="148" fill="#8fa0bc" :font-size="FS.small">Σₜ KL(p(·|x&lt;ₜ) ‖ q(·|x&lt;ₜ))</text>
    </g>

    <line x1="345" y1="26" x2="345" y2="156" stroke="#243350" stroke-dasharray="4 4" />

    <!-- 右:沿噪聲尺度拆(DPM / Flow) -->
    <g>
      <text x="372" y="20" fill="#b48cff" :font-size="FS.title">DPM / Flow:沿「噪聲尺度」拆</text>
      <line x1="382" y1="100" x2="666" y2="100" stroke="#8fa0bc" stroke-width="1.4" marker-end="url(#da-arrow)" />
      <g v-for="i in noise" :key="'n' + i">
        <circle :cx="400 + i * 56" cy="89" :r="9 + i * 2.4"
                fill="#b48cff" :fill-opacity="0.05 + i * 0.05" stroke="#b48cff" stroke-width="1.6"
                :stroke-opacity="1 - i * 0.13" />
        <text :x="400 + i * 56" y="122" fill="#8fa0bc" :font-size="FS.label" text-anchor="middle">σ{{ i }}</text>
      </g>
      <text x="382" y="142" fill="#8fa0bc" :font-size="FS.small">每一步:從噪聲量回推乾淨訊號 → 退化成回歸</text>
      <text x="382" y="160" fill="#8fa0bc" :font-size="FS.small">所以訓練跟 AR 一樣穩定</text>
    </g>
  </svg>
</template>
