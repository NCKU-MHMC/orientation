<script setup>
// 圖 B-5:生成學習三難(Xiao, Kreis & Vahdat, 2022)。六個家族一律落在三角形的邊上:
// 一條邊由兩個可兼得的目標張成,對邊的頂點即該邊放棄的目標;沿邊的位置表示兩者之間的偏重。
import { fs, palette } from './chart-style.js'

const props = defineProps({
  focus: { type: String, default: '' }, // 'AR' | 'NF' | 'VAE' | 'EBM' | 'DPM' | 'GAN'
  compact: { type: Boolean, default: false },
})

// 頂點:上=品質,左下=速度,右下=覆蓋
const V = {
  quality: { x: 330, y: 52 },
  speed: { x: 86, y: 286 },
  coverage: { x: 574, y: 286 },
}

const edges = [
  {
    from: 'quality', to: 'speed', color: palette.bad,
    give: '犧牲 mode coverage', gx: 194, gy: 172, anchor: 'end',
    side: 'right', // 家族名寫在三角形內側
    members: [{ key: 'GAN', t: 0.45 }],
  },
  {
    from: 'quality', to: 'coverage', color: palette.p,
    give: '犧牲 sampling speed', gx: 468, gy: 172, anchor: 'start',
    side: 'left',
    members: [{ key: 'DPM / FM', t: 0.30 }, { key: 'AR', t: 0.55 }, { key: 'EBM', t: 0.80 }],
  },
  {
    from: 'speed', to: 'coverage', color: palette.q,
    give: '犧牲 sample quality', gx: 330, gy: 307, anchor: 'middle',
    side: 'above',
    members: [{ key: 'VAE', t: 0.42 }, { key: 'NF', t: 0.68 }],
  },
]

const at = (e, t) => ({
  x: V[e.from].x + (V[e.to].x - V[e.from].x) * t,
  y: V[e.from].y + (V[e.to].y - V[e.from].y) * t,
})
const labelPos = (e, p) => {
  if (e.side === 'above') return { x: p.x, y: p.y - 14, anchor: 'middle' }
  if (e.side === 'left') return { x: p.x - 13, y: p.y + 5, anchor: 'end' }
  return { x: p.x + 13, y: p.y + 5, anchor: 'start' }
}
// focus 允許用短名(DPM 對應 'DPM / FM')
const dim = (key) => props.focus && !key.split(' / ').includes(props.focus)
</script>

<template>
  <div class="tri">
    <svg :width="compact ? 330 : 660" :height="compact ? 170 : 340" viewBox="0 0 660 340">
      <polygon :points="`${V.quality.x},${V.quality.y} ${V.speed.x},${V.speed.y} ${V.coverage.x},${V.coverage.y}`"
        fill="none" :stroke="palette.ink" stroke-width="2" />

      <text :x="V.quality.x" :y="V.quality.y - 16" text-anchor="middle" font-weight="700"
        :fill="palette.ink" :style="{ fontSize: fs('label') }">sample quality</text>
      <text :x="V.speed.x" :y="V.speed.y + 32" text-anchor="middle" font-weight="700"
        :fill="palette.ink" :style="{ fontSize: fs('label') }">sampling speed</text>
      <text :x="V.coverage.x" :y="V.coverage.y + 32" text-anchor="middle" font-weight="700"
        :fill="palette.ink" :style="{ fontSize: fs('label') }">mode coverage</text>

      <g v-for="e in edges" :key="e.give">
        <text v-if="!compact" :x="e.gx" :y="e.gy" :text-anchor="e.anchor" :fill="e.color"
          :style="{ fontSize: fs('tick') }">{{ e.give }}</text>
        <g v-for="m in e.members" :key="m.key" :style="{ opacity: dim(m.key) ? 0.28 : 1 }">
          <circle :cx="at(e, m.t).x" :cy="at(e, m.t).y" r="6.5" :fill="e.color"
            stroke="#fff" stroke-width="1.5" />
          <text :x="labelPos(e, at(e, m.t)).x" :y="labelPos(e, at(e, m.t)).y"
            :text-anchor="labelPos(e, at(e, m.t)).anchor" font-weight="700" :fill="palette.ink"
            :style="{ fontSize: fs('label') }">{{ m.key }}</text>
        </g>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.tri { text-align: center; }
.tri svg { margin: 0 auto; }
</style>
