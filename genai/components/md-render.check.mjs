// markdown-it 把「以區塊級 HTML 標籤開頭、直到空行」的區段當成 raw HTML,
// 不解析裡面的 $...$ 與 **粗體**。這類錯誤不會讓 build 失敗,只會在投影片上
// 原樣印出 $H(p)$ 或 **粗體**,所以必須靠檢查抓。
//
// 修法:數學改用 <katex-elem expr="..." />,粗體改用 <b>...</b>。
import { readFileSync } from 'node:fs'
import assert from 'node:assert/strict'

const BLOCK = /^\s*<\/?(div|span|b|a|table|p|section|svg|template)\b/i
const FILES = process.argv.slice(2)
assert.ok(FILES.length, '用法:node md-render.check.mjs <deck.md> ...')

const bad = []
for (const file of FILES) {
  const lines = readFileSync(file, 'utf8').split('\n')
  let inHtml = false, inFence = false
  lines.forEach((l, i) => {
    if (/^```/.test(l)) inFence = !inFence
    if (inFence) return
    if (l.trim() === '') { inHtml = false; return }
    if (BLOCK.test(l)) inHtml = true
    if (!inHtml) return
    for (const m of l.match(/\$[^$\n]+\$/g) ?? []) bad.push(`${file}:${i + 1} 行內數學 ${m} → 改用 <katex-elem expr="…" />`)
    for (const m of l.match(/\*\*[^*\n]+\*\*/g) ?? []) bad.push(`${file}:${i + 1} 粗體 ${m} → 改用 <b>…</b>`)
  })
}

assert.equal(bad.length, 0, `raw HTML 區塊裡有不會被渲染的 markdown:\n  ${bad.join('\n  ')}`)
console.log(`md-render ok(檢查 ${FILES.length} 份投影片)`)
