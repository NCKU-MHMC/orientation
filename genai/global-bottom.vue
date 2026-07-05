<script setup>
import { computed } from 'vue'
import { useNav } from '@slidev/client'

const nav = useNav()
const page = computed(() => nav.currentPage.value)
const total = computed(() => nav.total.value)
const progress = computed(() =>
  total.value > 1 ? ((page.value - 1) / (total.value - 1)) * 100 : 0,
)
const isCover = computed(() => page.value === 1)
const isDemo = computed(() => {
  try {
    return nav.currentSlideRoute.value?.meta?.slide?.frontmatter?.layout === 'iframe'
  } catch {
    return false
  }
})
const pad = (n) => String(n).padStart(2, '0')
</script>

<template>
  <footer v-if="!isCover" class="hud" aria-hidden="true">
    <div class="hud-bar">
      <div class="hud-fill" :style="{ width: progress + '%' }" />
    </div>
    <div v-if="!isDemo" class="hud-row">
      <span class="hud-label">生成學習入門 · L1</span>
      <span class="hud-page">{{ pad(page) }} / {{ pad(total) }}</span>
    </div>
  </footer>
</template>

<style scoped>
.hud {
  position: fixed;
  inset: auto 0 0 0;
  z-index: 40;
  pointer-events: none;
}
.hud-bar {
  height: 3px;
  background: #1b2740;
}
.hud-fill {
  height: 100%;
  /* 課程敘事弧:青(VAE) → 紫(理論/FM) → 粉(GAN);fixed 讓漸層錨定整個視窗寬 */
  background: linear-gradient(90deg, #5edfff, #b48cff, #ff6b9d) fixed;
  transition: width 0.45s cubic-bezier(0.22, 1, 0.36, 1);
}
.hud-row {
  display: flex;
  justify-content: space-between;
  padding: 4px 14px 6px;
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 10.5px;
  letter-spacing: 0.14em;
  color: #8fa0bc;
  background: linear-gradient(180deg, transparent, rgba(12, 18, 32, 0.55));
}
.hud-page { color: #e8edf6; }
</style>
