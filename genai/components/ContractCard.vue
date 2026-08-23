<script setup>
// 圖 B-C:介面契約。分布 = 至多提供兩個介面的物件。
defineProps({
  compact: { type: Boolean, default: false },
})
</script>

<template>
  <div class="contract" :class="{ compact }">
    <div class="contract-head">分布 p 作為一個物件</div>
    <div class="iface">
      <code>p.sample()</code>
      <span class="arrow">→</span>
      <span class="desc">從 p 抽出一個樣本</span>
    </div>
    <div class="iface">
      <code>p.logprob(x)</code>
      <span class="arrow">→</span>
      <span class="desc">回傳 log p(x)</span>
    </div>
  </div>
</template>

<style scoped>
.contract {
  width: fit-content;
  margin: 0 auto;
  border: 1px solid var(--rule-2);
  border-radius: 8px;
  padding: 1.1rem 1.8rem;
  background: var(--paper-2);
}
.contract-head {
  font-family: var(--serif);
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 0.7rem;
  font-size: 1.05rem;
}
.iface {
  display: flex;
  align-items: baseline;
  gap: 0.7rem;
  margin: 0.45rem 0;
  font-size: 1.05rem;
}
.iface code {
  background: var(--paper-3);
  border-radius: 5px;
  padding: 0.15rem 0.55rem;
  color: var(--ink);
  font-size: 1rem;
}
.arrow { color: var(--muted); }
.desc { color: var(--ink-2); }
.compact { padding: 0.6rem 1.1rem; }
.compact .contract-head { font-size: 0.92rem; margin-bottom: 0.35rem; }
.compact .iface { font-size: 0.9rem; margin: 0.25rem 0; }
.compact .iface code { font-size: 0.85rem; }
</style>
