<script setup>
// GAN 的迴路。重點只有一件事:G 的唯一學習訊號是穿過 D 傳回來的梯度。
// 沒有 encoder、沒有重建項、沒有任何 pixel 級的目標。
import { typeScale } from './chart-style.js'

const W = 700, H = 215
const FS = typeScale(W)
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <defs>
      <marker id="ad-a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#8fa0bc" />
      </marker>
      <marker id="ad-g" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#ffb454" />
      </marker>
    </defs>

    <!-- 上路:z → G → 假樣本,在 x=300 轉下來進 D 的左緣 y=96 -->
    <ellipse cx="52" cy="62" rx="42" ry="19" fill="#151d2e" stroke="#8fa0bc" stroke-width="1.3" />
    <text x="52" y="67" fill="#8fa0bc" :font-size="FS.label" text-anchor="middle">z ~ N(0,I)</text>
    <line x1="96" y1="62" x2="152" y2="62" stroke="#8fa0bc" stroke-width="1.5" marker-end="url(#ad-a)" />

    <rect x="158" y="44" width="104" height="36" rx="7" fill="#ff6b9d" fill-opacity="0.18" stroke="#ff6b9d" stroke-width="1.8" />
    <text x="210" y="67" fill="#ff6b9d" :font-size="FS.title" text-anchor="middle" font-weight="600">Generator G</text>
    <path d="M262,62 H300 V96 H326" stroke="#8fa0bc" stroke-width="1.5" fill="none" marker-end="url(#ad-a)" />
    <text x="281" y="54" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">G(z)</text>

    <!-- 下路:真資料,同樣在 x=300 轉上來進 y=114。兩段垂直不重疊(96 < 114) -->
    <rect x="158" y="130" width="104" height="36" rx="7" fill="#0c1220" stroke="#b48cff" stroke-width="1.5" />
    <text x="210" y="153" fill="#b48cff" :font-size="FS.title" text-anchor="middle">真資料 x</text>
    <path d="M262,148 H300 V114 H326" stroke="#8fa0bc" stroke-width="1.5" fill="none" marker-end="url(#ad-a)" />

    <!-- D -->
    <rect x="330" y="86" width="112" height="38" rx="7" fill="#ffb454" fill-opacity="0.16" stroke="#ffb454" stroke-width="1.8" />
    <text x="386" y="110" fill="#ffb454" :font-size="FS.title" text-anchor="middle" font-weight="600">Discriminator D</text>

    <line x1="444" y1="105" x2="500" y2="105" stroke="#8fa0bc" stroke-width="1.5" marker-end="url(#ad-a)" />
    <rect x="506" y="87" width="100" height="36" rx="7" fill="#0c1220" stroke="#243350" stroke-width="1.3" />
    <text x="556" y="110" fill="#e8edf6" :font-size="FS.title" text-anchor="middle">真 or 假?</text>

    <!-- 梯度回傳:G 唯一的訊號 -->
    <path d="M556,87 V22 H210 V40" stroke="#ffb454" stroke-width="1.8" fill="none"
          stroke-dasharray="6 4" marker-end="url(#ad-g)" />
    <text x="386" y="16" fill="#ffb454" :font-size="FS.title" text-anchor="middle">
      G 的唯一學習訊號:穿過 D 傳回來的梯度
    </text>

    <!-- 註記 -->
    <text x="14" :y="H - 26" fill="#8fa0bc" :font-size="FS.small">
      D 是逐點評分器:它一次只看一個樣本,回答「這一個像不像真的」。
    </text>
    <text x="14" :y="H - 8" fill="#ff6b9d" :font-size="FS.small">
      「你漏了一整群」是分布層級的性質,這條迴路上沒有任何地方能傳遞它。
    </text>
  </svg>
</template>
