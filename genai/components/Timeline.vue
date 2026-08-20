<script setup>
// 改進史時間軸(垂直,吃 props 重用)。
// items: [{ name, year, note, tag }],tag 為三難頂點或光譜方向的短標。
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
        <span class="tl-name">{{ it.name }}</span>
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
.tl-dot { width: 0.62rem; height: 0.62rem; border-radius: 50%; background: #2563eb; margin-top: 0.32rem; flex: none; }
.tl-line { width: 2px; flex: 1; background: #e2e8f0; }
.tl-body { padding-bottom: 0.72rem; }
.dense .tl-body { padding-bottom: 0.45rem; }
.tl-name { font-weight: 600; color: #0f172a; font-size: 0.92rem; }
.tl-year { color: #64748b; font-size: 0.8rem; margin-left: 0.45rem; }
.tl-tag {
  margin-left: 0.55rem; font-size: 0.72rem; color: #b45309; background: #fef3c7;
  border-radius: 999px; padding: 0.05rem 0.5rem; vertical-align: 0.08rem;
}
.tl-note { color: #475569; font-size: 0.82rem; margin-top: 0.08rem; }
</style>
