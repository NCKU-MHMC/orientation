<script setup>
// 家族的體質對照。刻意用三格量表取代形容詞:
// 「偏糊」「銳利」「極高」這種詞放在同一張表裡沒有可比性,點數有。
//
// 值可以是 0–2 的定值,也可以是 [lo, hi] 區間。用區間是因為有些欄位的高低
// 取決於框架內的設計選擇(路徑、取樣器、步數),不是家族本身的體質。
// 把這種欄位畫成定值等於誤導。
const COLS = ['密度', '樣本品質', '取樣速度', '訓練穩定', 'mode 覆蓋']

const ROWS = [
  { n: 'Autoregressive', c: '#5edfff', v: [2, 2, 0, 2, 2], note: '逐 token,慢在取樣', tag: '第 ④ 段' },
  { n: 'Normalizing Flow', c: '#5edfff', v: [2, 1, 2, 2, 2], note: '可逆變換 + Jacobian', tag: '' },
  { n: 'VAE', c: '#5edfff', v: [1, 0, 2, 2, 2], note: '只有下界', tag: '第 ④ 段', hi: true },
  { n: 'GAN', c: '#ff6b9d', v: [0, 2, 2, 0, 0], note: '寫不出密度', tag: '第 ④ 段', hi: true },
  {
    n: 'Diffusion / Flow Matching', c: '#b48cff', v: [1, 2, [0, 2], 2, 2],
    note: '步數是路徑與取樣器的選擇,不是體質', tag: '下堂課',
  },
]

// 定值 → 實心到該格;區間 → lo 以下實心,lo..hi 之間畫成空心,表示「可移動」
const cell = (v) => {
  const [lo, hi] = Array.isArray(v) ? v : [v, v]
  return [0, 1, 2].map((k) => (k <= lo ? 'on' : k <= hi ? 'range' : ''))
}
</script>

<template>
  <div class="mx">
    <div class="head">
      <span />
      <span v-for="c in COLS" :key="c">{{ c }}</span>
      <span>備註</span>
    </div>
    <div v-for="r in ROWS" :key="r.n" class="row" :class="{ hi: r.hi }" :style="{ '--c': r.c }">
      <span class="name">
        {{ r.n }}<i v-if="r.tag">{{ r.tag }}</i>
      </span>
      <span v-for="(v, i) in r.v" :key="i" class="meter">
        <i v-for="(cls, k) in cell(v)" :key="k" :class="cls" />
      </span>
      <span class="note">{{ r.note }}</span>
    </div>
  </div>
</template>

<style scoped>
.mx { font-size: 0.875rem; }
.head, .row {
  display: grid;
  grid-template-columns: 13em repeat(5, 1fr) 10.5em;
  gap: 8px;
  align-items: center;
}
.head {
  color: var(--muted);
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  padding: 0 4px 5px;
  text-align: center;
}
.head span:first-child, .head span:last-child { text-align: left; }
.row {
  background: var(--panel);
  border-left: 4px solid var(--c);
  border-radius: 6px;
  padding: 6px 8px;
  margin-bottom: 5px;
}
.row.hi { background: color-mix(in srgb, var(--c) 12%, var(--panel)); }
.name { font-family: var(--mono); color: var(--c); font-size: 0.78rem; }
.name i {
  display: block;
  font-style: normal;
  font-size: 0.68rem;
  color: var(--muted);
  opacity: 0.7;
}
.meter { display: flex; gap: 3px; justify-content: center; }
.meter i {
  width: 15px;
  height: 7px;
  border-radius: 2px;
  background: var(--edge);
}
.meter i.on { background: var(--c); }
.meter i.range {
  background: transparent;
  border: 1.5px dashed var(--c);
  opacity: 0.75;
}
.note { font-size: 0.72rem; color: var(--muted); }
</style>
