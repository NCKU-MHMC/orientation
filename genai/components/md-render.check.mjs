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

// CommonMark 的強調規則:** 要能「開啟」粗體必須是 left-flanking,要能「關閉」必須是
// right-flanking。中文的 「『（《 都算 Unicode 標點,所以寫成 到**「詞」**這 時,
// 開頭的 ** 後面接標點、前面接文字 → 不是 left-flanking → 整個粗體失效,
// ** 會原樣印在投影片上。這跟上面的 raw HTML 問題無關,空行救不了它。
//
// 判斷 opener/closer:同一行裡的 ** 依序交替(第 0、2、4… 個是 opener)。
// 不配對就無從判斷角色,所以奇數個 ** 直接略過。
const PUNCT = /[\p{P}\p{S}]/u
const SPACE = /[\s ]/
const flankable = (prev, next, role) => {
  if (next === '' || SPACE.test(next)) { if (role === 'open') return false }
  if (prev === '' || SPACE.test(prev)) { if (role === 'close') return false }
  if (role === 'open') return !PUNCT.test(next) || prev === '' || SPACE.test(prev) || PUNCT.test(prev)
  return !PUNCT.test(prev) || next === '' || SPACE.test(next) || PUNCT.test(next)
}

const bad = []
for (const file of FILES) {
  const lines = readFileSync(file, 'utf8').split('\n')
  let inHtml = false, inFence = false
  lines.forEach((l, i) => {
    if (/^```/.test(l)) inFence = !inFence
    if (inFence) return
    if (l.trim() === '') { inHtml = false; return }
    if (BLOCK.test(l)) inHtml = true

    if (inHtml) {
      for (const m of l.match(/\$[^$\n]+\$/g) ?? []) bad.push(`${file}:${i + 1} 行內數學 ${m} → 改用 <katex-elem expr="…" />`)
      for (const m of l.match(/\*\*[^*\n]+\*\*/g) ?? []) bad.push(`${file}:${i + 1} 粗體 ${m} → 改用 <b>…</b>`)
      return
    }

    // markdown 會被解析的區域:檢查 ** 兩端有沒有踩到中文標點造成的失效
    const at = []
    for (let k = 0; k + 1 < l.length; k++) if (l[k] === '*' && l[k + 1] === '*') { at.push(k); k++ }
    if (at.length % 2) return
    at.forEach((pos, n) => {
      const role = n % 2 ? 'close' : 'open'
      if (flankable(l[pos - 1] ?? '', l[pos + 2] ?? '', role)) return
      const near = l.slice(Math.max(0, pos - 6), pos + 8)
      bad.push(`${file}:${i + 1} 粗體的${role === 'open' ? '起始' : '結束'} ** 緊鄰中文標點,不會渲染:…${near}… → 改用 <b>…</b>`)
    })
  })
}

assert.equal(bad.length, 0, `有不會被渲染的 markdown:\n  ${bad.join('\n  ')}`)
console.log(`md-render ok(檢查 ${FILES.length} 份投影片)`)
