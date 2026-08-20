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
  const ch = Math.max(doc.documentElement.scrollHeight, doc.body.scrollHeight)
  if (ch < 50) return
  const s = Math.min(980 / props.designWidth, props.maxH / ch)
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
    <div v-if="title" class="demo-title">{{ title }}</div>
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
  padding: 0.75rem 1.25rem 0;
}
.demo-title {
  font-size: 1.05rem;
  font-weight: 600;
  color: #1e293b;
  margin-bottom: 0.4rem;
}
.demo-box {
  overflow: hidden;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background: #fff;
  margin: 0 auto;
}
.demo-title {
  text-align: center;
}
</style>
