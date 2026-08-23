<script setup>
// 圖 B-0:mode-covering ↔ mode-seeking 光譜,三列技術逐步補齊。
const props = defineProps({
  rows: { type: Number, default: 3 }, // 顯示前幾列
  mark: { type: String, default: '' }, // 高亮某列:'objective' | 'decoding' | 'weights'
})

const allRows = [
  {
    key: 'objective',
    name: '訓練目標',
    left: 'forward KL・MLE',
    mid: 'JSD',
    right: 'reverse KL・RLHF',
  },
  {
    key: 'decoding',
    name: '解碼設定',
    left: 'temperature T > 1',
    mid: 'T = 1',
    right: '收緊 top-p・調高 CFG 係數',
  },
  {
    key: 'weights',
    name: '權重微調',
    left: 'SFT(仍是 MLE)',
    mid: '',
    right: 'DPO / DDO・小 β',
  },
]
const shown = allRows.slice(0, props.rows)
</script>

<template>
  <div class="spec">
    <div class="spec-axis">
      <div class="pole left">
        <div class="pole-name">mode-covering</div>
        <div class="pole-desc">廣覆蓋、過度平滑</div>
      </div>
      <div class="bar" />
      <div class="pole right">
        <div class="pole-name">mode-seeking</div>
        <div class="pole-desc">銳利、可能丟失眾數</div>
      </div>
    </div>
    <div v-for="r in shown" :key="r.key" class="spec-row" :class="{ hot: mark === r.key }">
      <div class="row-name">{{ r.name }}</div>
      <div class="cell left">{{ r.left }}</div>
      <div class="cell mid">{{ r.mid }}</div>
      <div class="cell right">{{ r.right }}</div>
    </div>
  </div>
</template>

<style scoped>
.spec { width: 100%; max-width: 56rem; margin: 0 auto; }
.spec-axis { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.7rem; }
.bar {
  flex: 1; height: 0.55rem; border-radius: 999px;
  background: linear-gradient(90deg, var(--data), var(--rule-2), var(--model));
}
.pole { text-align: center; min-width: 9rem; }
.pole-name { font-weight: 600; color: var(--ink); font-size: 0.95rem; }
.pole-desc { font-size: 0.78rem; color: var(--muted); }
.spec-row {
  display: grid; grid-template-columns: 6.2rem 1fr 1fr 1fr;
  align-items: center; gap: 0.4rem;
  padding: 0.32rem 0.2rem; border-radius: 8px;
}
.spec-row.hot { background: var(--accent-tint); }
.row-name { font-weight: 600; color: var(--ink-2); font-size: 0.88rem; }
.cell { font-size: 0.85rem; color: var(--ink); text-align: center; }
.cell.left { color: var(--data); }
.cell.right { color: var(--accent-ink); text-align: right; padding-right: 0.4rem; }
.cell.mid { color: var(--muted); }
</style>
