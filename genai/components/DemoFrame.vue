<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'

const props = defineProps({
  src: { type: String, required: true }, // public/demos 下的檔名
  title: { type: String, default: '' },
  designWidth: { type: Number, default: 1280 }, // demo 版面的設計寬度
  maxH: { type: Number, default: 500 }, // 投影片內可用高度(px,980 寬座標系)
})

const url = `${import.meta.env.BASE_URL}demos/${props.src}`
const frame = ref(null)
const scale = ref(0.5)
const boxH = ref(props.maxH)

const AVAIL_W = 980 - 16 // 980 寬的版面扣掉 .demo-frame 左右內距

// Slidev 會預先掛載相鄰頁,onload 時機不可靠,故常駐輪詢量測。
let timer = null
function fit() {
  const el = frame.value
  if (!el) return
  let doc
  try {
    doc = el.contentDocument
  } catch {
    return // 非同源時放棄縮放,維持預設
  }
  if (!doc || !doc.body) return
  doc.body.classList.add('embed') // 觸發 demo 內建的緊湊模式
  // 只量 body 自身的版面高度:documentElement.scrollHeight 會被 iframe 的視窗高度撐住,
  // 而 iframe 高度又由這裡算出去,兩者互相回饋會把盒子鎖在初次量到的值。
  const ch = Math.ceil(doc.body.getBoundingClientRect().height)
  if (ch < 50) return
  const s = Math.min(AVAIL_W / props.designWidth, props.maxH / ch)
  scale.value = s
  boxH.value = Math.min(props.maxH, ch * s)
}
onMounted(() => {
  fit()
  timer = setInterval(fit, 600)
})
onBeforeUnmount(() => clearInterval(timer))
</script>

<template>
  <div class="demo-frame">
    <a v-if="title" class="demo-title" :href="url" target="_blank" rel="noopener">
      {{ title }}<span class="demo-open" aria-hidden="true">↗</span>
    </a>
    <div class="demo-box" :style="{ height: boxH + 'px', width: designWidth * scale + 'px' }">
      <iframe
        ref="frame"
        :src="url"
        :style="{
          width: designWidth + 'px',
          height: boxH / scale + 'px',
          transform: `scale(${scale})`,
          transformOrigin: 'top left',
          border: 'none',
        }"
      />
    </div>
  </div>
</template>

<style scoped>
.demo-frame {
  padding: 0.6rem 0.5rem 0;
}
.demo-title {
  display: block;
  text-align: center;
  font-family: var(--sans);
  font-size: 1rem;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 0.4rem;
  text-decoration: none;
  border-bottom: none;
}
.demo-title:hover { color: var(--accent-ink); }
/* 提示這行標題可以點開獨立的 demo 頁 */
.demo-open {
  font-size: 0.78em;
  color: var(--muted);
  margin-left: 0.35em;
  vertical-align: 0.12em;
}
.demo-title:hover .demo-open { color: var(--accent-ink); }
.demo-box {
  overflow: hidden;
  border: 1px solid var(--rule);
  border-radius: 8px;
  background: var(--paper-2);
  margin: 0 auto;
}
</style>
