<script setup>
// 推論期四層介入:介入點、方法、所需介面。
const props = defineProps({
  focus: { type: Number, default: 0 }, // 0 = 不高亮;1..4 高亮該層
})

const layers = [
  { n: 1, point: '改變條件 c', methods: 'prompt・few-shot・RAG・memory', iface: '僅 sample(條件版)' },
  { n: 2, point: '改變抽樣', methods: 'temperature・top-k / top-p / min-p・beam', iface: 'logprob(逐 token)' },
  { n: 3, point: '改變 logits', methods: 'logit bias・constrained decoding・contrastive decoding・DoLa・CFG', iface: 'logprob(逐 token)' },
  { n: 4, point: '改變樣本聚合', methods: 'self-consistency・best-of-n・MBR・reranking', iface: 'sample(logprob 可選)' },
]
</script>

<template>
  <div class="ls">
    <div class="ls-head">
      <div>層</div><div>介入點</div><div>方法</div><div>所需介面</div>
    </div>
    <div v-for="l in layers" :key="l.n" class="ls-row" :class="{ hot: focus === l.n, sampleOnly: l.n === 1 || l.n === 4 }">
      <div class="ls-n">{{ l.n }}</div>
      <div class="ls-point">{{ l.point }}</div>
      <div class="ls-methods">{{ l.methods }}</div>
      <div class="ls-iface" :class="{ nolp: l.iface.startsWith('僅') }">{{ l.iface }}</div>
    </div>
  </div>
</template>

<style scoped>
.ls { max-width: 56rem; margin: 0 auto; }
.ls-head, .ls-row {
  display: grid; grid-template-columns: 2.4rem 8.5rem 1fr 12rem;
  gap: 0.6rem; align-items: center;
}
.ls-head { font-size: 0.8rem; color: var(--muted); padding: 0 0.6rem 0.3rem; }
.ls-row {
  background: var(--paper-2); border: 1px solid var(--rule); border-radius: 8px;
  padding: 0.5rem 0.6rem; margin-bottom: 0.4rem; font-size: 0.9rem; color: var(--ink);
}
.ls-row.hot { background: var(--accent-tint); border-color: var(--accent); }
.ls-n { font-family: var(--mono); font-weight: 500; color: var(--muted); text-align: center; }
.ls-point { font-weight: 600; }
.ls-methods { color: var(--ink-2); }
.ls-iface { font-size: 0.82rem; color: var(--accent-ink); }
.ls-iface.nolp { color: var(--third); font-weight: 600; }
</style>
