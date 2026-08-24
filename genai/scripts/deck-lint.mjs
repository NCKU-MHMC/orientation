// 簡報文字驗收:禁用詞、句式配額、第二人稱、CJK 粗體陷阱、HTML 行內 markdown。
import { readFileSync } from 'node:fs'

const files = process.argv.slice(2)
if (files.length === 0) {
  console.error('usage: node scripts/deck-lint.mjs <deck.md> [...]')
  process.exit(2)
}

// 比喻性口語術語與後設編排用語,一律不得出現在投影片檔(含講稿)。
const BANNED = [
  '一把尺', '把尺', '那條軸', '這條軸', '骨架', '骨幹', '招牌病', '體質',
  '伏筆', '下一堂', '下堂', '下週會', '留到第二堂', '留到下', '這頁回收',
  // 宣傳語與空洞強調
  '最關鍵', '徹底改變', '令人驚豔', '值得注意的是', '不可或缺',
]

let failed = false
const report = (file, line, msg) => {
  failed = true
  console.error(`${file}:${line}: ${msg}`)
}

for (const file of files) {
  const raw = readFileSync(file, 'utf8')
  const lines = raw.split('\n')

  // 建立「可見文字」遮罩:排除 HTML 註解(講稿)、frontmatter、程式碼區塊
  let inComment = false
  let inFence = false
  let fmCount = 0
  const visible = lines.map((l) => {
    const trimmed = l.trim()
    if (/^```/.test(trimmed)) { inFence = !inFence; return '' }
    if (inFence) return ''
    if (trimmed === '---') { fmCount++; return '' }
    let out = l
    if (inComment) {
      const end = out.indexOf('-->')
      if (end === -1) return ''
      out = out.slice(end + 3)
      inComment = false
    }
    out = out.replace(/<!--.*?-->/g, '')
    const start = out.indexOf('<!--')
    if (start !== -1) {
      out = out.slice(0, start)
      inComment = true
    }
    return out
  })

  lines.forEach((l, i) => {
    for (const w of BANNED) {
      if (l.includes(w)) report(file, i + 1, `禁用詞「${w}」`)
    }
    // CJK 粗體陷阱:依 CommonMark flanking 規則,開頭 ** 後接標點且前接文字、
    // 或結尾 ** 前接標點且後接文字時,不會解析為粗體
    {
      const punct = /[,。、;:?!「」『』()().,;:!?'"\-]/
      const re2 = /\*\*/g
      let mm
      let open = false
      while ((mm = re2.exec(l))) {
        const before = l[mm.index - 1] ?? ' '
        const after = l[mm.index + 2] ?? ' '
        if (!open && punct.test(after) && !/\s/.test(before) && !punct.test(before))
          report(file, i + 1, '開頭 ** 後接標點且前貼文字,粗體會失效')
        if (open && punct.test(before) && !/\s/.test(after) && !punct.test(after))
          report(file, i + 1, '結尾 ** 前接標點且後貼文字,粗體會失效')
        open = !open
      }
    }
    // 行內 HTML 標籤同行夾 markdown 語法,不會被解析
    if (/^\s*<[a-zA-Z][^>]*>.*(\*\*|(^|[^$])\$[^$]+\$)/.test(l))
      report(file, i + 1, 'HTML 標籤同行內含 markdown/數學語法,不會渲染')
  })

  visible.forEach((l, i) => {
    if (l.includes('你')) report(file, i + 1, '可見文字使用第二人稱「你」')
  })

  // 主題把 h1 之後緊接的段落渲染成淺灰註解(.slidev-layout h1 + p),
  // 因此每個 h1 後必須是一句可獨立閱讀的短註解:非清單/表格/公式/HTML,句末不留冒號。
  visible.forEach((l, i) => {
    if (!/^# /.test(l)) return
    let j = i + 1
    while (j < visible.length && visible[j].trim() === '') j++
    const next = (visible[j] ?? '').trim()
    if (next === '' || /^(<|\$\$|\||[-*]\s|\d+\.\s|#)/.test(next)) {
      // 封面(h1 後接 h2)與 center/statement 版面的 h1 後接 div,兩者不需註解
      if (/^(<div|##)/.test(next)) return
      return report(file, i + 1, 'h1 之後應緊接一句淺灰註解(主題以 h1 + p 呈現)')
    }
    if (/[:：]$/.test(next)) report(file, j + 1, '淺灰註解句末不應以冒號銜接下一段')
    // 以全形寬度計:CJK 佔 2、拉丁佔 1,超過 72 就不只一行
    // markdown 連結只算顯示文字,URL 不佔版面
    const shown = next.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    const width = [...shown.replace(/\$[^$]*\$/g, '字')]
      .reduce((w, c) => w + (c.charCodeAt(0) > 0x2e80 ? 2 : 1), 0)
    if (width > 72) report(file, j + 1, `淺灰註解過長(寬度 ${width}),應為一行短句`)
  })

  // 「不是…而是…」each deck ≤ 3
  const visText = visible.join('\n')
  const contrast = visText.match(/不是[^。\n]{0,40}而是/g) ?? []
  if (contrast.length > 3)
    report(file, 0, `「不是…而是…」句式 ${contrast.length} 處,超過上限 3`)
}

if (failed) process.exit(1)
console.log('deck-lint: OK')
