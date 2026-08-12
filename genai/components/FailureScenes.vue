<script setup>
// ① 的兩個失敗現場,⑤ 回收時再放一次。
defineProps({ verdict: { type: Boolean, default: false } })
</script>

<template>
  <div class="scenes">
    <div class="scene" style="--c: #5edfff">
      <div class="tag">現場 A · base model</div>
      <div class="ask">「這題的標準答案是什麼?」</div>
      <div class="say"><span>「這取決於很多因素……不同情境下可能有不同看法……」</span></div>
      <div class="diag">試圖覆蓋全部 → <b>空泛 hedging</b></div>
    </div>

    <div class="mid" v-if="verdict">同一條軸<br />的兩端</div>
    <div class="mid" v-else>?</div>

    <div class="scene" style="--c: #ff6b9d">
      <div class="tag">現場 B · 對齊後模型</div>
      <div class="ask">同一個 prompt,取樣 10 次</div>
      <div class="say">
        <span v-for="i in 4" :key="i">「當然!很高興為您……」</span>
      </div>
      <div class="diag">十個回答幾乎一樣 → <b>多樣性塌陷</b></div>
    </div>
  </div>
</template>

<style scoped>
.scenes { display: grid; grid-template-columns: 1fr 5.2em 1fr; gap: 12px; align-items: center; }
.scene {
  border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);
  border-radius: 10px;
  background: var(--panel);
  padding: 12px 14px;
}
.tag {
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  color: var(--c);
  margin-bottom: 8px;
}
.ask {
  font-size: 0.78rem;
  color: var(--muted);
  border-left: 2px solid var(--edge);
  padding-left: 8px;
  margin-bottom: 8px;
}
.say { display: flex; flex-direction: column; gap: 4px; font-size: 0.875rem; }
.say > span, .say {
  color: var(--ink);
}
.say > span {
  background: var(--panel-deep);
  border: 1px solid var(--edge);
  border-radius: 8px 8px 8px 2px;
  padding: 3px 8px;
  opacity: 0.9;
}
.diag { margin-top: 10px; font-size: 0.78rem; color: var(--muted); }
.diag b { color: var(--c); }
.mid {
  text-align: center;
  font-size: 0.875rem;
  line-height: 1.35;
  color: var(--amber);
  font-family: var(--mono);
}
</style>
