<script setup>
// 配圖 B-1:單一高斯 q 擬合雙峰 p,三個散度解出三種行為。
import { computed, ref } from 'vue'
import { XS, FITS, bimodal, fitAll, gauss } from './divergence-math.js'
import { typeScale } from './chart-style.js'

const props = defineProps({
  // 左峰權重。給定值 = 靜態圖(既有頁面);不給 = 顯示滑桿,拖到哪即時重解到哪
  w: { type: Number, default: null },
  curves: { type: Array, default: () => ['forward', 'jsd', 'reverse'] },
  annotate: { type: String, default: '' },  // '' | 'forward' | 'reverse'
})

// H 壓到 200:這張圖最擠的一頁(澄清四)還要放 caption + 圖例 + 兩個說明框,
// 471px 的內容高度扣完只剩約 270px 給圖。
const W = 720, H = 200, PAD = 26
const FS = typeScale(W)
const XMIN = -4.2, XMAX = 4.2, YMAX = 0.75
const sx = (x) => PAD + ((x - XMIN) / (XMAX - XMIN)) * (W - 2 * PAD)
const sy = (y) => H - 26 - (y / YMAX) * (H - 52)

const idx = XS.map((_, i) => i).filter((i) => XS[i] >= XMIN && XS[i] <= XMAX)
const line = (vals) =>
  idx.map((i, k) => `${k ? 'L' : 'M'}${sx(XS[i]).toFixed(1)},${sy(vals[i]).toFixed(1)}`).join('')

const META = {
  forward: { c: '#5edfff', label: 'forward KL', tag: 'mode-covering' },
  jsd: { c: '#ffb454', label: 'JSD', tag: '看資料而定' },
  reverse: { c: '#ff6b9d', label: 'reverse KL', tag: 'mode-seeking' },
}

const drag = ref(0.5)
const w = computed(() => props.w ?? drag.value)
// 公布過的兩個權重直接用 FITS(精度高、與講稿數字一致),其餘現算
const table = computed(() => FITS[w.value] ?? fitAll(w.value))

const p = computed(() => bimodal(w.value))
const pPath = computed(() => `${line(p.value)}L${sx(XMAX)},${sy(0)}L${sx(XMIN)},${sy(0)}Z`)
// 解幾乎相同的曲線會整條疊在一起,只看得到最後畫的那條(w≈0.3 時 JSD 與 reverse KL
// 就是同一個解)。同組的線改畫成互相錯開的虛線,每條各佔 1/n 段,疊區三色交替可見。
const DASH = 7
const same = (a, b) => Math.abs(a.mu - b.mu) < 0.12 && Math.abs(a.sigma - b.sigma) < 0.12

const fits = computed(() => {
  const raw = props.curves.map((k) => ({ k, ...META[k], ...table.value[k] }))
  return raw.map((f) => {
    const group = raw.filter((g) => same(f, g))
    return {
      ...f,
      d: line(XS.map((x) => gauss(x, f.mu, f.sigma))),
      dash: group.length > 1 ? `${DASH} ${DASH * (group.length - 1)}` : null,
      off: -DASH * group.indexOf(f),
    }
  })
})
</script>

<template>
  <div>
    <svg :viewBox="`0 0 ${W} ${H}`" class="w-full">
      <!-- p_data -->
      <path :d="pPath" fill="#b48cff" fill-opacity="0.16" stroke="#b48cff" stroke-width="1.6" />
      <text :x="sx(-3.6)" :y="sy(0.66)" fill="#b48cff" :font-size="FS.label">p_data</text>

      <!-- 三條擬合曲線 -->
      <path v-for="f in fits" :key="f.k" :d="f.d" :stroke="f.c" stroke-width="2.4" fill="none"
            :stroke-dasharray="f.dash" :stroke-dashoffset="f.off" />

      <!-- 標註:forward KL 為什麼必須在兩峰之間配置質量 -->
      <g v-if="annotate === 'forward'">
        <line :x1="sx(0)" :y1="sy(0.02)" :x2="sx(0)" :y2="sy(0.56)"
              stroke="#5edfff" stroke-width="1" stroke-dasharray="3 3" />
        <text :x="sx(0)" :y="sy(0.62)" fill="#5edfff" :font-size="FS.small" text-anchor="middle">
          這裡 p≈0,q 仍不能取 0
        </text>
        <text :x="sx(0)" :y="H - 8" fill="#8fa0bc" :font-size="FS.small" text-anchor="middle">
          ← 兩峰之間出現虛假的機率質量 →
        </text>
      </g>

      <!-- 標註:reverse KL 忽略掉的那個峰 -->
      <g v-if="annotate === 'reverse'">
        <line :x1="sx(-1.6)" :y1="sy(0.02)" :x2="sx(-1.6)" :y2="sy(0.56)"
              stroke="#ff6b9d" stroke-width="1" stroke-dasharray="3 3" />
        <text :x="sx(-1.6)" :y="sy(0.60)" fill="#ff6b9d" :font-size="FS.small" text-anchor="middle">
          整個峰未被覆蓋,不受懲罰
        </text>
      </g>

      <line :x1="PAD" :y1="sy(0)" :x2="W - PAD" :y2="sy(0)" stroke="#243350" stroke-width="1" />
    </svg>

    <div class="fit-legend">
      <span v-for="f in fits" :key="f.k" :style="{ color: f.c }">
        <span class="rule" :style="{ borderTopColor: f.c, borderTopStyle: f.dash ? 'dashed' : 'solid' }" />
        {{ f.label }} · μ={{ f.mu }} σ={{ f.sigma }}
        <span style="color: var(--muted)">({{ f.tag }})</span>
      </span>
    </div>

    <!-- 未指定 w 時當成可拖動的展示界面:拖到哪就重解到哪 -->
    <label v-if="props.w === null" class="fit-slider">
      <span>左峰權重 w</span>
      <input type="range" min="0.1" max="0.9" step="0.05" v-model.number="drag" />
      <span class="val">{{ w.toFixed(2) }} : {{ (1 - w).toFixed(2) }}</span>
    </label>
  </div>
</template>

<style scoped>
.fit-legend {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 4px;
  font-family: var(--mono);
  font-size: 0.72rem;
}
.fit-slider {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  justify-content: center;
  margin-top: 6px;
  font-family: var(--mono);
  font-size: 0.72rem;
  color: var(--muted);
}
.fit-slider input {
  width: 16rem;
  accent-color: #b48cff;
}
.fit-slider .val {
  color: #b48cff;
  min-width: 5.4rem;
}
.rule {
  display: inline-block;
  width: 1rem;
  border-top: 2.4px solid;
  vertical-align: middle;
}
</style>
