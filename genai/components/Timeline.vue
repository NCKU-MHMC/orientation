<script setup>
// 改進史時間軸(垂直,吃 props 重用)。
// items: [{ name, year, note, tag, url }],tag 為三難頂點或覆蓋程度方向的短標;
// url 有值時,name 變成指向該文獻的連結(開新分頁)。
defineProps({
  items: { type: Array, required: true },
  dense: { type: Boolean, default: false },
})
</script>

<template>
  <div class="tl" :class="{ dense }">
    <div v-for="(it, i) in items" :key="i" class="tl-item">
      <div class="tl-rail">
        <div class="tl-dot" />
        <div v-if="i < items.length - 1" class="tl-line" />
      </div>
      <div class="tl-body">
        <a v-if="it.url" class="tl-name" :href="it.url" target="_blank" rel="noopener">{{ it.name }}</a>
        <span v-else class="tl-name">{{ it.name }}</span>
        <span v-if="it.year" class="tl-year">{{ it.year }}</span>
        <span v-if="it.tag" class="tl-tag">{{ it.tag }}</span>
        <div v-if="it.note" class="tl-note">{{ it.note }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tl { max-width: 52rem; margin: 0 auto; }
.tl-item { display: flex; gap: 0.8rem; }
.tl-rail { display: flex; flex-direction: column; align-items: center; width: 1rem; }
.tl-dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; background: var(--rule-2); margin-top: 0.4rem; flex: none; }
.tl-line { width: 1px; flex: 1; background: var(--rule); }
.tl-body { padding-bottom: 0.72rem; }
.dense .tl-body { padding-bottom: 0.45rem; }
.tl-name { font-weight: 600; color: var(--ink); font-size: 0.92rem; }
a.tl-name { text-decoration: none; border-bottom: 1px solid var(--rule-2); }
.tl-year { font-family: var(--mono); color: var(--muted); font-size: 0.78rem; margin-left: 0.45rem; }
.tl-tag {
  margin-left: 0.55rem; font-size: 0.72rem; color: var(--accent-ink); background: var(--accent-tint);
  border-radius: 999px; padding: 0.05rem 0.5rem; vertical-align: 0.08rem;
}
.tl-note { color: var(--ink-2); font-size: 0.82rem; margin-top: 0.08rem; }
</style>
