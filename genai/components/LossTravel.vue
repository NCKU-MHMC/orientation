<script setup>
// 收束圖:同一個核心問題,兩種評估基準,各自長出一個家族,再各自出走到別的領域。
// 這張圖的用意是讓學生記住「基準可攜」,而不是記住模型名字。
import { typeScale } from './chart-style.js'

const W = 700, H = 262
const FS = typeScale(W)

const LANE = [
  {
    y: 52, c: '#5edfff',
    ruler: '固定的基準', sub: 'KL / MLE / ELBO',
    family: 'AR · VAE · Diffusion',
    behav: 'mode-covering',
    travel: ['知識蒸餾', 'RLHF 的 β·KL', 'TRPO / PPO 信賴域', 'label smoothing'],
    say: '「我有一個參考分布,請新分布別離它太遠。」',
  },
  {
    y: 158, c: '#ff6b9d',
    ruler: '學出來的基準', sub: '判別器 / JSD',
    family: 'GAN',
    behav: 'mode-seeking',
    travel: ['DANN 域適應', 'pix2pix / CycleGAN', 'SRGAN / ESRGAN', 'HiFi-GAN vocoder'],
    say: '「我寫不出『像真的』的公式,那就訓練一個分類器當基準,然後騙過它。」',
  },
]
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
    <defs>
      <marker id="lt-a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#8fa0bc" />
      </marker>
    </defs>

    <!-- 核心問題 -->
    <rect x="6" y="86" width="104" height="76" rx="8" fill="#151d2e" stroke="#b48cff" stroke-width="1.8" />
    <text x="58" y="112" fill="#b48cff" :font-size="FS.title" text-anchor="middle">核心問題</text>
    <text x="58" y="130" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">縮短 p_θ 與</text>
    <text x="58" y="145" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">p_data 的差異</text>

    <g v-for="l in LANE" :key="l.ruler">
      <path :d="`M110,124 H136 V${l.y + 20} H154`" fill="none" stroke="#8fa0bc" stroke-width="1.4" marker-end="url(#lt-a)" />

      <!-- 基準 -->
      <rect x="158" :y="l.y" width="112" height="40" rx="7"
            :fill="l.c" fill-opacity="0.14" :stroke="l.c" stroke-width="1.7" />
      <text x="214" :y="l.y + 18" :fill="l.c" :font-size="FS.title" text-anchor="middle">{{ l.ruler }}</text>
      <text x="214" :y="l.y + 33" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">{{ l.sub }}</text>

      <line x1="272" :y1="l.y + 20" x2="300" :y2="l.y + 20" stroke="#8fa0bc" stroke-width="1.4" marker-end="url(#lt-a)" />

      <!-- 家族 -->
      <rect x="304" :y="l.y" width="140" height="40" rx="7" fill="#0c1220" :stroke="l.c" stroke-width="1.4" />
      <text x="374" :y="l.y + 18" fill="#e8edf6" :font-size="FS.title" text-anchor="middle">{{ l.family }}</text>
      <text x="374" :y="l.y + 33" :fill="l.c" :font-size="FS.small" text-anchor="middle">{{ l.behav }}</text>

      <line x1="446" :y1="l.y + 20" x2="474" :y2="l.y + 20" stroke="#8fa0bc" stroke-width="1.4" marker-end="url(#lt-a)" />

      <!-- 出走 -->
      <g v-for="(t, i) in l.travel" :key="t">
        <rect x="478" :y="l.y - 12 + i * 21" width="216" height="18" rx="9"
              fill="#0c1220" :stroke="l.c" stroke-opacity="0.45" />
        <text x="586" :y="l.y + 1 + i * 21" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">{{ t }}</text>
      </g>
    </g>

    <text x="158" y="132" fill="#ffb454" :font-size="FS.small">共同句型:</text>
    <text x="228" y="132" fill="#8fa0bc" :font-size="FS.small">{{ LANE[0].say }}</text>
    <text x="158" :y="H - 6" fill="#8fa0bc" :font-size="FS.small">{{ LANE[1].say }}</text>
  </svg>
</template>
