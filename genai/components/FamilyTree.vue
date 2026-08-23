<script setup>
// 家族樹:依訓練散度與 logprob 形式分層。
import { fs, palette } from './chart-style.js'

const box = { w: 140, h: 46 }
const nodes = [
  { id: 'root', x: 400, y: 24, main: '生成物件', sub: '逼近 p_data', tone: 'ink' },
  { id: 'fwd', x: 226, y: 116, main: 'forward KL 家族', sub: 'logprob 可得', tone: 'p' },
  { id: 'jsd', x: 606, y: 116, main: 'JSD 家族', sub: '無 logprob', tone: 'bad' },
  { id: 'rev', x: 760, y: 116, main: 'reverse KL 家族', sub: '缺 p_data.logprob', tone: 'q' },
  { id: 'exact', x: 16, y: 212, main: 'AR・NF', sub: 'logprob 精確', tone: 'p' },
  { id: 'bound', x: 164, y: 212, main: 'VAE', sub: '下界(ELBO)', tone: 'p' },
  { id: 'multi', x: 312, y: 212, main: 'DPM / FM', sub: '多步分解', tone: 'p' },
  { id: 'ebm', x: 460, y: 212, main: 'EBM', sub: '未正規化(缺 log Z)', tone: 'p' },
  { id: 'gan', x: 606, y: 212, main: 'GAN', sub: '判別器代理', tone: 'bad' },
  { id: 'rlhf', x: 760, y: 212, main: 'RLHF・VI', sub: 'reward / energy 代理', tone: 'q' },
]
const edges = [
  ['root', 'fwd'], ['root', 'jsd'], ['root', 'rev'],
  ['fwd', 'exact'], ['fwd', 'bound'], ['fwd', 'multi'], ['fwd', 'ebm'],
  ['jsd', 'gan'], ['rev', 'rlhf'],
]
const byId = Object.fromEntries(nodes.map((n) => [n.id, n]))
const toneColor = { p: palette.p, q: palette.q, bad: palette.bad, ink: palette.ink }
</script>

<template>
  <div class="ftree">
    <svg width="900" height="286" viewBox="0 0 900 286">
      <line v-for="([a, b], i) in edges" :key="i"
        :x1="byId[a].x + box.w / 2" :y1="byId[a].y + box.h"
        :x2="byId[b].x + box.w / 2" :y2="byId[b].y"
        :stroke="palette.grid" stroke-width="1.8" />
      <g v-for="n in nodes" :key="n.id">
        <rect :x="n.x" :y="n.y" :width="box.w" :height="box.h" rx="8"
          :fill="palette.paper" :stroke="toneColor[n.tone]" stroke-width="1.8" />
        <text :x="n.x + box.w / 2" :y="n.y + 19" text-anchor="middle" font-weight="600"
          :fill="toneColor[n.tone]" :style="{ fontSize: fs('note') }">{{ n.main }}</text>
        <text :x="n.x + box.w / 2" :y="n.y + 37" text-anchor="middle"
          :fill="palette.muted" :style="{ fontSize: fs('tick') }">{{ n.sub }}</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.ftree { text-align: center; }
.ftree svg { margin: 0 auto; }
</style>
