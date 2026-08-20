<script setup>
// 角落常駐的介面契約徽章(對應板書「寫在黑板角落,整堂保留」)。
import { useNav } from '@slidev/client'
const nav = useNav()
</script>

<template>
  <div v-if="nav.currentPage.value > 1" class="contract-badge">
    <code>sample()</code><span class="sep">·</span><code>logprob(x)</code>
  </div>
</template>

<style scoped>
.contract-badge {
  position: fixed;
  right: 0.7rem;
  bottom: 0.55rem;
  font-size: 0.58rem;
  color: #94a3b8;
  background: #f8fafcE6;
  border: 1px solid #e2e8f0;
  border-radius: 999px;
  padding: 0.12rem 0.55rem;
  z-index: 40;
}
.contract-badge code { color: #64748b; }
.sep { margin: 0 0.3rem; }
</style>
