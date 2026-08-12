<script setup>
// Generative Learning Trilemma(Xiao, Kreis & Vahdat, ICLR 2022)。
// 三個頂點只能同時抓兩個。模型放在它拿到的那條邊上,對面的頂點就是它放棄的。
//
// 尺寸與位置從外面控制:
//   <Trilemma />                                  預設置中,最大 700px
//   <Trilemma style="--tri-max: 780px" />         放大
//   <Trilemma style="--tri-w: 55%" />             縮小
//   <Trilemma style="--tri-mx: 0" />              靠左(取消置中)
//   <div class="w-1/2"><Trilemma /></div>         包一層也可以
//
// 字級不受上面任何一種縮放影響:元件用 ResizeObserver 量自己實際被渲染成多寬,
// 再回推 viewBox 單位。所以放大圖形時文字仍然維持 chart-style.js 定的 14 / 12.5 / 11.5 px。
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { CONTENT, typeScale } from './chart-style.js'

const W = 560, H = 285

// 量測實際渲染寬度。首次繪製與 SSG 期間沒有 ResizeObserver,先用預設值。
// 預設 700px:這張投影片扣掉標題與說明文字後,圖最多放得下約 380px 高,
// 700 × 285/560 = 356px,還留 30px 餘裕。要再放大就得同時縮短說明文字。
const root = ref(null)
const renderW = ref(700)
let ro
onMounted(() => {
  if (!root.value || typeof ResizeObserver === 'undefined') return
  ro = new ResizeObserver(([e]) => {
    const w = e.contentRect.width
    if (w > 0) renderW.value = w
  })
  ro.observe(root.value)
})
onBeforeUnmount(() => ro?.disconnect())

const FS = computed(() => typeScale(W, Math.min(renderW.value, CONTENT.w)))

const CX = W / 2, CY = 158, R = 122
const V = [
  { k: 'q', t: '樣本品質', a: -90, c: '#5edfff' },
  { k: 's', t: '取樣速度', a: 30, c: '#ffb454' },
  { k: 'c', t: 'mode 覆蓋', a: 150, c: '#ff6b9d' },
]
const pt = (a, r = R) => [
  CX + r * Math.cos((a * Math.PI) / 180),
  CY + r * Math.sin((a * Math.PI) / 180),
]
const P = Object.fromEntries(V.map((v) => [v.k, pt(v.a)]))
const mid = (a, b) => [(P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2]

// 每個模型坐在它拿到的那兩個頂點之間;drop = 它放棄的頂點。
// slide:這個位置不是固定的,框架內的設計選擇會把它沿邊推向某個頂點。
const MODELS = [
  { n: 'GAN', edge: ['q', 's'], drop: 'mode 覆蓋', c: '#ff6b9d' },
  { n: 'VAE', edge: ['s', 'c'], drop: '樣本品質', c: '#5edfff' },
  {
    n: 'Diffusion / FM', edge: ['q', 'c'], drop: '取樣速度', c: '#b48cff',
    slide: { to: 's', label: '少步化' },
  },
]

// 標籤往三角形外面推,避免壓在邊上
const OUT = 30
const label = (m) => {
  const [mx, my] = mid(...m.edge)
  const [dx, dy] = [mx - CX, my - CY]
  const len = Math.hypot(dx, dy) || 1
  return [mx + (dx / len) * OUT, my + (dy / len) * OUT]
}

const slideArrow = (m) => {
  const [x1, y1] = mid(...m.edge)
  const [vx, vy] = P[m.slide.to]
  const L = 0.36 // 只畫一小段,表示方向而不是終點
  return { x1, y1, x2: x1 + (vx - x1) * L, y2: y1 + (vy - y1) * L }
}
const arrowTip = (m) => {
  const a = slideArrow(m)
  return [a.x2, a.y2]
}
</script>

<template>
  <svg ref="root" :viewBox="`0 0 ${W} ${H}`" class="tri">
    <defs>
      <marker id="tri-slide" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#ffb454" />
      </marker>
    </defs>

    <text x="8" y="16" fill="#8fa0bc" :font-size="FS.small">坐在一條邊上 = 放棄對面那個頂點</text>

    <polygon :points="V.map((v) => P[v.k].join(',')).join(' ')"
             fill="#151d2e" stroke="#243350" stroke-width="1.6" />

    <!-- 位置可以沿邊移動:路徑、取樣器、步數的選擇會把它推向某個頂點 -->
    <g v-for="m in MODELS.filter((x) => x.slide)" :key="'s' + m.n">
      <line v-bind="slideArrow(m)" stroke="#ffb454" stroke-width="1.8"
            stroke-dasharray="5 3" marker-end="url(#tri-slide)" />
      <text :x="arrowTip(m)[0] + 9" :y="arrowTip(m)[1] + 4" fill="#ffb454" :font-size="FS.small">
        {{ m.slide.label }}
      </text>
    </g>

    <!-- 模型坐在邊的中點 -->
    <g v-for="m in MODELS" :key="m.n">
      <circle :cx="mid(...m.edge)[0]" :cy="mid(...m.edge)[1]" r="7" :fill="m.c" />
      <text :x="label(m)[0]" :y="label(m)[1]" :fill="m.c" :font-size="FS.title"
            text-anchor="middle" font-weight="600">{{ m.n }}</text>
      <text :x="label(m)[0]" :y="label(m)[1] + 15" fill="#8fa0bc" :font-size="FS.small"
            text-anchor="middle">放棄{{ m.drop }}</text>
    </g>

    <!-- 頂點 -->
    <g v-for="v in V" :key="v.k">
      <circle :cx="P[v.k][0]" :cy="P[v.k][1]" r="5" :fill="v.c" />
      <text :x="pt(v.a, R + 28)[0]" :y="pt(v.a, R + 28)[1] + 4"
            :fill="v.c" :font-size="FS.title" text-anchor="middle">{{ v.t }}</text>
    </g>
  </svg>
</template>

<style scoped>
.tri {
  display: block;
  width: var(--tri-w, 100%);
  max-width: var(--tri-max, 700px);
  height: auto;
  margin-inline: var(--tri-mx, auto);
}
</style>
