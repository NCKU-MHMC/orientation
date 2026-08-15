<script setup>
// 板書 B-3:DDO 的機制。關鍵在「判別器沒有自己的網路」——它是由 p_θ 與 p_ref 的
// 似然比直接參數化出來的,所以不需要交替訓練,也不需要對取樣過程反向傳播。
import { typeScale } from './chart-style.js'

const W = 700, H = 236
const FS = typeScale(W)
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <defs>
      <marker id="ddo-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#8fa0bc" />
      </marker>
      <marker id="ddo-arrow-v" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#b48cff" />
      </marker>
    </defs>

    <!-- 左:兩個樣本來源 -->
    <g>
      <rect x="16" y="34" width="150" height="46" rx="7" fill="#151d2e" stroke="#5edfff" stroke-opacity="0.6" />
      <text x="91" y="55" fill="#5edfff" :font-size="FS.title" text-anchor="middle">真實樣本</text>
      <text x="91" y="71" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">x ~ p_data · 標籤 1</text>

      <rect x="16" y="112" width="150" height="46" rx="7" fill="#151d2e" stroke="#ff6b9d" stroke-opacity="0.6" />
      <text x="91" y="133" fill="#ff6b9d" :font-size="FS.title" text-anchor="middle">參考模型樣本</text>
      <text x="91" y="149" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">x ~ p_ref · 標籤 0</text>
    </g>

    <line x1="166" y1="57" x2="252" y2="82" stroke="#5edfff" stroke-width="1.4"
          marker-end="url(#ddo-arrow)" opacity="0.75" />
    <line x1="166" y1="135" x2="252" y2="110" stroke="#ff6b9d" stroke-width="1.4"
          marker-end="url(#ddo-arrow)" opacity="0.75" />

    <!-- 中:隱式判別器 -->
    <rect x="258" y="60" width="216" height="72" rx="8" fill="#0c1220" stroke="#b48cff" stroke-width="1.8" />
    <text x="366" y="80" fill="#b48cff" :font-size="FS.title" text-anchor="middle">隱式判別器</text>
    <text x="366" y="102" fill="#e8edf6" :font-size="FS.title" text-anchor="middle" font-family="var(--mono)">
      d_θ = σ( β·log p_θ / p_ref )
    </text>
    <text x="366" y="121" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">
      沒有額外網路 · 參數就是 θ 本身
    </text>

    <line x1="474" y1="96" x2="546" y2="96" stroke="#8fa0bc" stroke-width="1.4" marker-end="url(#ddo-arrow)" />

    <!-- 右:判別 loss -->
    <rect x="552" y="60" width="132" height="72" rx="8" fill="#151d2e" stroke="#ffb454" stroke-opacity="0.6" />
    <text x="618" y="84" fill="#ffb454" :font-size="FS.title" text-anchor="middle">判別 loss</text>
    <text x="618" y="102" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">標準 GAN 的 BCE</text>
    <text x="618" y="118" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">最佳解 p_θ = p_data</text>

    <!-- 回饋:self-play -->
    <path d="M618,132 L618,190 L91,190 L91,158" stroke="#b48cff" stroke-width="1.4"
          stroke-dasharray="4 4" fill="none" marker-end="url(#ddo-arrow-v)" />
    <text x="356" y="184" fill="#b48cff" :font-size="FS.small" text-anchor="middle">
      每一輪結束:參考模型 ← 本輪的最佳模型(self-play)
    </text>

    <text x="16" y="20" fill="#8fa0bc" :font-size="FS.small">
      對照 GAN:判別器是另一張網路、要交替訓練、生成器的梯度得穿過取樣過程
    </text>
    <rect x="16" y="204" :width="W - 32" height="26" rx="7" fill="#151d2e" stroke="#ffb454" stroke-opacity="0.5" />
    <text :x="W / 2" y="221" fill="#ffb454" :font-size="FS.title" text-anchor="middle">
      likelihood-based 模型已能計算 log p_θ,判別器因此不需要額外參數
    </text>
  </svg>
</template>
