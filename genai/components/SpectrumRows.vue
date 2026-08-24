<script setup>
// 圖 B-0:mode-covering ↔ mode-seeking 的覆蓋程度,三列技術逐步補齊。
const props = defineProps({
  rows: { type: Number, default: 3 }, // 顯示前幾列
  mark: { type: String, default: '' }, // 高亮某列:'objective' | 'decoding' | 'weights'
  ddo: { type: Boolean, default: false }, // 權重微調列下方加一條橫跨兩端的 DDO 帶
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
    right: 'DPO・小 β',
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
    <template v-for="r in shown" :key="r.key">
      <div class="spec-row" :class="{ hot: mark === r.key }">
        <div class="row-name">{{ r.name }}</div>
        <div class="cell left">{{ r.left }}</div>
        <div class="cell mid">{{ r.mid }}</div>
        <div class="cell right">{{ r.right }}</div>
      </div>
      <!-- 其他方法各佔一端,DDO 的抬升項與壓低項分屬兩端,因此畫成一條橫跨的帶 -->
      <div v-if="ddo && r.key === 'weights'" class="spec-span">
        <div class="row-name" />
        <div class="span-bar">
          <span class="tip left">抬升項</span>
          <span class="line" />
          <span class="pill">DDO 兩端同時施力</span>
          <span class="line" />
          <span class="tip right">壓低項</span>
        </div>
      </div>
    </template>
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

/* DDO 橫跨帶:兩端箭頭指向兩極,中間標出方法名 */
.spec-span {
  display: grid; grid-template-columns: 6.2rem 1fr;
  align-items: center; gap: 0.4rem; padding: 0.1rem 0.2rem 0.2rem;
}
.span-bar { display: flex; align-items: center; gap: 0.4rem; padding-right: 0.4rem; }
.span-bar .line { flex: 1; height: 1px; background: var(--rule-2); }
.tip { font-size: 0.78rem; font-weight: 600; position: relative; }
.tip.left { color: var(--data); padding-left: 0.55rem; }
.tip.right { color: var(--accent-ink); padding-right: 0.55rem; }
/* 箭頭以 CSS 三角形畫,避免多一張 SVG */
.tip::before {
  content: ''; position: absolute; top: 50%; margin-top: -0.22rem;
  border-top: 0.22rem solid transparent; border-bottom: 0.22rem solid transparent;
}
.tip.left::before { left: 0; border-right: 0.36rem solid var(--data); }
.tip.right::before { right: 0; border-left: 0.36rem solid var(--accent-ink); }
.pill {
  font-size: 0.78rem; font-weight: 600; color: var(--third);
  border: 1px solid var(--third); border-radius: 999px;
  padding: 0.05rem 0.5rem; white-space: nowrap;
}
</style>
