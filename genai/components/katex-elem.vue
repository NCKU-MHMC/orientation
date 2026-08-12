<!--
行內數學。

用途:markdown-it 會把「以區塊級 HTML 標籤開頭、直到空行」的區段當成 raw HTML,
不解析裡面的 $...$。所以寫在 <div>、<b>、表格 cell 等地方的行內數學不會被渲染,
會原樣印出 $H(p)$。這個元件直接呼叫 katex,不受 markdown 解析範圍影響。

  <katex-elem expr="p(y \mid x)" />
  <katex-elem expr="\log p_\theta" opt='{"displayMode":true}' />

一般段落裡的 $...$ 仍然照常用 markdown 寫,不需要這個元件。
-->

<script setup lang="ts">
import katex from 'katex'
import { onMounted, ref, watch } from 'vue'

const props = defineProps<{
  expr: string
  opt?: string | null
}>()

const root = ref<HTMLElement | null>(null)
const render = () => {
  if (!root.value) return
  // throwOnError:false → 寫錯的式子顯示成紅字,不會讓整頁掛掉
  katex.render(props.expr, root.value, {
    throwOnError: false,
    ...JSON.parse(props.opt ?? '{}'),
  })
}

onMounted(render)
watch(() => [props.expr, props.opt], render)
</script>

<template>
  <span ref="root" class="katex-elem" />
</template>

<style scoped>
.katex-elem { color: var(--ink); }
</style>
