<script setup>
// 統一引導式的逐項解剖:base 項、係數、比值項。
</script>

<template>
  <div class="gf">
    <div class="gf-eq">
      <span class="term ghost">log&nbsp;p<sub>guided</sub></span>
      <span class="op">=</span>
      <span class="term base">log&nbsp;p<sub>base</sub></span>
      <span class="op">+</span>
      <span class="term coef">w</span>
      <span class="term ratio">( log&nbsp;p<sub>A</sub> − log&nbsp;p<sub>B</sub> )</span>
    </div>
    <div class="gf-notes">
      <div class="note base-n">
        <div class="tag">base 項</div>
        <div>出發的分布。prompt 改動的就是這一項</div>
      </div>
      <div class="note coef-n">
        <div class="tag">係數</div>
        <div>光譜上的座標:越大越往 mode-seeking 端</div>
      </div>
      <div class="note ratio-n">
        <div class="tag">比值項</div>
        <div>移動方向:抬高 p<sub>A</sub> 相對 p<sub>B</sub> 偏好的區域</div>
      </div>
    </div>
    <div class="gf-tail">計算完成後再正規化;整條式子只呼叫 logprob 介面</div>
  </div>
</template>

<style scoped>
.gf { text-align: center; }
.gf svg { margin: 0 auto; }
.gf-eq {
  font-size: 1.7rem; font-family: 'KaTeX_Math', 'Times New Roman', serif;
  color: #0f172a; display: flex; justify-content: center; align-items: baseline; gap: 0.55rem;
}
.term { padding: 0.1rem 0.35rem; border-radius: 6px; }
.term.base { background: #dbeafe; }
.term.coef { background: #fef3c7; }
.term.ratio { background: #d1fae5; }
.op { color: #64748b; }
.gf-notes { display: flex; justify-content: center; gap: 1.1rem; margin-top: 1.1rem; }
.note { max-width: 14rem; font-size: 0.85rem; color: #334155; }
.note .tag {
  font-weight: 700; font-size: 0.8rem; width: fit-content; margin: 0 auto 0.25rem;
  padding: 0.06rem 0.5rem; border-radius: 999px;
}
.base-n .tag { background: #dbeafe; color: #1d4ed8; }
.coef-n .tag { background: #fef3c7; color: #b45309; }
.ratio-n .tag { background: #d1fae5; color: #047857; }
.gf-tail { margin-top: 0.9rem; font-size: 0.85rem; color: #64748b; }
</style>
