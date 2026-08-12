<script setup>
// ③ 的骨架:一個散度能不能算,決定它變成哪一個演算法家族。
const ROWS = [
  {
    ruler: 'forward KL',
    c: '#5edfff',
    needs: [
      { t: '從 p_data 取樣', ok: true },
      { t: '算 p_θ 密度', ok: true },
    ],
    stand: '不需要代理',
    family: 'MLE → AR / VAE / DPM',
  },
  {
    ruler: 'reverse KL',
    c: '#ff6b9d',
    needs: [
      { t: '從 p_θ 取樣', ok: true },
      { t: '算 p_data 密度', ok: false },
    ],
    stand: 'reward / energy 代理',
    family: 'VI · RLHF',
  },
  {
    ruler: 'JSD',
    c: '#ffb454',
    needs: [
      { t: '算 p_data 密度', ok: false },
      { t: '算 p_θ 密度', ok: false },
    ],
    stand: '判別器代理',
    family: 'GAN',
  },
]
</script>

<template>
  <div class="map">
    <div class="head">
      <span>散度</span><span>它要什麼</span><span>由誰代理</span><span>於是變成</span>
    </div>
    <div v-for="r in ROWS" :key="r.ruler" class="row" :style="{ '--c': r.c }">
      <div class="ruler">{{ r.ruler }}</div>
      <div class="needs">
        <span v-for="n in r.needs" :key="n.t" class="need" :class="{ bad: !n.ok }">
          <b>{{ n.ok ? '✓' : '✗' }}</b>{{ n.t }}
        </span>
      </div>
      <div class="stand" :class="{ none: r.stand === '不需要代理' }">{{ r.stand }}</div>
      <div class="family">{{ r.family }}</div>
    </div>
  </div>
</template>

<style scoped>
/* 字級對齊 chart-style.js 的 14 / 12.5 / 11.5 ladder */
.map { font-size: 0.875rem; }
.head, .row {
  display: grid;
  grid-template-columns: 6.6em 1fr 8.4em 11em;
  gap: 10px;
  align-items: center;
}
.head {
  color: var(--muted);
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  padding: 0 4px 6px;
}
.row {
  border-left: 4px solid var(--c);
  border-radius: 6px;
  background: var(--panel);
  padding: 9px 10px;
  margin-bottom: 8px;
}
.ruler { font-family: var(--mono); color: var(--c); font-weight: 600; }
.needs { display: flex; gap: 8px; flex-wrap: wrap; }
.need {
  border: 1px solid var(--edge);
  border-radius: 999px;
  padding: 2px 9px;
  background: var(--panel-deep);
  white-space: nowrap;
}
.need b { color: #7ee0a0; margin-right: 5px; }
.need.bad { border-color: color-mix(in srgb, #ff6b9d 55%, transparent); }
.need.bad b { color: #ff6b9d; }
.stand { color: var(--amber); font-size: 0.78rem; }
.stand.none { color: var(--muted); opacity: 0.55; }
.family {
  font-family: var(--mono);
  font-size: 0.78rem;
  color: var(--ink);
  border: 1px solid var(--c);
  border-radius: 6px;
  padding: 3px 8px;
  text-align: center;
}
</style>
