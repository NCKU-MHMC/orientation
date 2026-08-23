<script setup>
// 圖 B-4:家族介面能力矩陣。focus 指定家族時高亮該列(節標頁定位用)。
const props = defineProps({
  focus: { type: String, default: '' }, // 'AR' | 'NF' | 'VAE' | 'DPM' | 'GAN'
  compact: { type: Boolean, default: false },
})

const rows = [
  { key: 'AR', name: 'AR', logprob: '精確(chain rule)', lp: 'ok', sample: '逐 token 序列(慢)', obj: 'forward KL / MLE' },
  { key: 'NF', name: 'Normalizing Flow', logprob: '精確(變數變換,需可逆 + Jacobian)', lp: 'ok', sample: '一步', obj: 'forward KL / MLE' },
  { key: 'VAE', name: 'VAE', logprob: '僅下界(ELBO)', lp: 'bound', sample: '一步', obj: 'forward KL 的下界' },
  { key: 'EBM', name: 'EBM', logprob: '未正規化(差 log Z)', lp: 'bound', sample: 'MCMC 多步(慢)', obj: 'forward KL(MCMC 梯度)/ score matching' },
  { key: 'DPM', name: 'DPM / FM', logprob: '下界;經 probability flow ODE 精確', lp: 'bound', sample: '多步迭代(慢)', obj: 'forward KL 的另一種分解' },
  { key: 'GAN', name: 'GAN', logprob: '無', lp: 'none', sample: '一步', obj: 'JSD(理論)/ non-saturating(實務)' },
]
</script>

<template>
  <div class="fm" :class="{ compact }">
    <div class="fm-head">
      <div>家族</div><div><code>logprob</code></div><div><code>sample</code></div><div>訓練目標</div>
    </div>
    <div v-for="r in rows" :key="r.key" class="fm-row"
      :class="{ hot: focus === r.key, dim: focus && focus !== r.key }">
      <div class="fm-name">{{ r.name }}</div>
      <div class="fm-lp" :class="r.lp">{{ r.logprob }}</div>
      <div>{{ r.sample }}</div>
      <div class="fm-obj">{{ r.obj }}</div>
    </div>
  </div>
</template>

<style scoped>
.fm { max-width: 58rem; margin: 0 auto; }
.fm-head, .fm-row {
  display: grid; grid-template-columns: 9.5rem 1.35fr 1fr 1.1fr;
  gap: 0.55rem; align-items: center;
}
.fm-head { font-size: 0.8rem; color: var(--muted); padding: 0 0.6rem 0.3rem; }
.fm-head code { font-size: 0.8rem; }
.fm-row {
  background: var(--paper-2); border: 1px solid var(--rule); border-radius: 8px;
  padding: 0.45rem 0.6rem; margin-bottom: 0.35rem; font-size: 0.85rem; color: var(--ink);
}
.fm-row.hot { background: var(--accent-tint); border-color: var(--accent); }
.fm-row.dim { opacity: 0.45; }
.fm-name { font-weight: 600; }
.fm-lp.ok { color: var(--third); }
.fm-lp.bound { color: var(--ink-2); }
.fm-lp.none { color: var(--warn); font-weight: 600; }
.fm-obj { color: var(--ink-2); }
.compact .fm-row { padding: 0.28rem 0.5rem; font-size: 0.72rem; margin-bottom: 0.22rem; }
.compact .fm-head { display: none; }
</style>
