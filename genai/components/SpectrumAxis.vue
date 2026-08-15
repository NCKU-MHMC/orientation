<script setup>
// 板書 B-0:貫穿全課的那條軸。第一堂只畫第一列,下面兩列留白給第二堂。
const props = defineProps({
  rows: { type: Number, default: 1 }, // 顯示到第幾列
})

const LANES = [
  {
    name: '訓練目標',
    items: [
      { at: 18, text: 'forward KL · MLE', c: '#5edfff' },
      // JSD 不是軸上的一個點:兩峰權重不對稱時它會倒向大峰,對稱時才落在中間。
      // 釘死 50% 會與第一堂「澄清四」那頁的結論矛盾,所以畫成區間。
      { at: 50, span: 30, text: 'JSD · GAN', c: '#ffb454' },
      { at: 82, text: 'reverse KL · RLHF', c: '#ff6b9d' },
    ],
  },
  {
    name: '解碼設定',
    items: [
      { at: 18, text: '溫度 T > 1', c: '#5edfff' },
      { at: 50, text: 'T = 1', c: '#ffb454' },
      { at: 82, text: 'top-p 收緊 · CFG↑', c: '#ff6b9d' },
    ],
  },
  {
    name: '權重微調',
    items: [
      { at: 20, text: 'SFT(仍是 MLE)', c: '#5edfff' },
      { at: 80, text: 'DPO / DDO · β 小', c: '#ff6b9d' },
    ],
  },
]
</script>

<template>
  <div class="axis">
    <div class="bar" />
    <div class="ends">
      <div>
        <b style="color: #5edfff">mode-covering</b>
        <span>涵蓋全部,品質低</span>
      </div>
      <div class="text-right">
        <b style="color: #ff6b9d">mode-seeking</b>
        <span>品質高,覆蓋不足</span>
      </div>
    </div>

    <div v-for="(lane, i) in LANES" :key="lane.name" class="lane">
      <div class="lane-name">{{ lane.name }}</div>
      <div class="lane-track">
        <template v-if="i < props.rows">
          <span v-for="it in lane.items.filter((x) => x.span)" :key="it.text + '-span'" class="span"
                :style="{ left: it.at - it.span / 2 + '%', width: it.span + '%', background: it.c }" />
          <span v-for="it in lane.items" :key="it.text" class="chip"
                :style="{ left: it.at + '%', color: it.c, borderColor: it.c }">{{ it.text }}</span>
        </template>
        <span v-else class="blank">{{ i === 0 ? '今天結束前填上這一列' : '下週補完' }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 字級對齊 chart-style.js 的 14 / 12.5 / 11.5 ladder */
.axis { font-size: 0.875rem; }
.bar {
  height: 6px;
  border-radius: 3px;
  background: linear-gradient(90deg, #5edfff, #ffb454 50%, #ff6b9d);
}
.ends {
  display: flex;
  justify-content: space-between;
  margin: 4px 0 10px;
  color: var(--muted);
}
.ends b { display: block; font-family: var(--mono); letter-spacing: 0.06em; }
.lane { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
.lane-name {
  width: 4.6em;
  flex: none;
  text-align: right;
  color: var(--muted);
  font-family: var(--mono);
  font-size: 0.72rem;
}
.lane-track {
  position: relative;
  flex: 1;
  height: 26px;
  border-radius: 6px;
  background: rgba(21, 29, 46, 0.6);
  border: 1px dashed var(--edge);
}
.chip {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  white-space: nowrap;
  padding: 2px 8px;
  border: 1px solid;
  border-radius: 999px;
  background: var(--panel-deep);
  font-size: 0.78rem;
}
/* 有 span 的項目不是一個點,而是一段可能的落點範圍 */
.span {
  position: absolute;
  top: 50%;
  height: 22px;
  transform: translateY(-50%);
  border-radius: 11px;
  opacity: 0.16;
}
.blank {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: var(--muted);
  opacity: 0.5;
  font-size: 0.78rem;
}
</style>
