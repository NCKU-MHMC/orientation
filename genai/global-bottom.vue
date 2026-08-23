<script setup>
// 角落常駐的介面契約徽章(對應板書「寫在黑板角落,整堂保留」)與頁碼。
import { useNav } from '@slidev/client'
const nav = useNav()
</script>

<template>
  <div v-if="nav.currentPage.value > 1" class="page-no">
    <SlideCurrentNo /><span class="sep">/</span><SlidesTotal />
  </div>
  <div v-if="nav.currentPage.value > 1" class="contract-badge">
    <code>sample()</code><span class="sep">·</span><code>logprob(x)</code>
  </div>
</template>

<style scoped>
.page-no {
  position: fixed;
  left: 3rem;
  bottom: 0.9rem;
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.04em;
  color: var(--muted);
  z-index: 40;
}
.page-no .sep { color: var(--rule-2); }
.contract-badge {
  position: fixed;
  right: 0.7rem;
  bottom: 0.55rem;
  font-family: var(--mono);
  font-size: 0.58rem;
  color: var(--muted);
  background: rgba(251, 250, 247, 0.92);
  border: 1px solid var(--rule);
  border-radius: 999px;
  padding: 0.12rem 0.55rem;
  z-index: 40;
}
.contract-badge code { color: var(--ink-2); }
.sep { margin: 0 0.3rem; }
</style>
