// 量測每一頁是否溢出 980×552 版面。用法:node scripts/measure-slides.mjs lecture-01.md
import { chromium } from 'playwright-chromium'
import { spawn } from 'node:child_process'

const deck = process.argv[2] ?? 'lecture-01.md'
const port = 3099

const server = spawn('npx', ['slidev', deck, '--port', String(port), '--remote'], {
  stdio: 'ignore',
  detached: true, // slidev 會再生子行程,結束時須殺整個 process group
})
const kill = () => { try { process.kill(-server.pid, 'SIGTERM') } catch {} }
process.on('exit', kill)

// 等 dev server 起來
async function waitUp() {
  for (let i = 0; i < 120; i++) {
    try {
      const r = await fetch(`http://localhost:${port}/`)
      if (r.ok) return
    } catch {}
    await new Promise((r) => setTimeout(r, 1000))
  }
  throw new Error('dev server 未啟動')
}

await waitUp()
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })

// 取得總頁數
await page.goto(`http://localhost:${port}/1`, { waitUntil: 'networkidle' })
const total = await page.evaluate(() => __slidev__?.nav?.total ?? 0)
if (!total) throw new Error('讀不到總頁數')

let bad = 0
for (let n = 1; n <= total; n++) {
  await page.goto(`http://localhost:${port}/${n}`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(400)
  const r = await page.evaluate(() => {
    const el = document.querySelector('.slidev-page .slidev-layout')
    if (!el) return null
    // scrollHeight 抓不到 grid/overflow 裁切,另量所有子元素的實際底緣
    const box = el.getBoundingClientRect()
    let maxB = box.bottom, maxR = box.right
    for (const c of el.querySelectorAll('*')) {
      // KaTeX 的伸縮符號用名目寬度極大的 SVG,由父層裁切,屬設計內,不算溢出
      if (c.closest('.katex')) continue
      const b = c.getBoundingClientRect()
      if (b.height === 0 && b.width === 0) continue
      if (b.bottom > maxB) maxB = b.bottom
      if (b.right > maxR) maxR = b.right
    }
    const scale = box.height / el.clientHeight || 1
    return {
      sw: el.scrollWidth, sh: el.scrollHeight,
      cw: el.clientWidth, ch: el.clientHeight,
      spillB: (maxB - box.bottom) / scale, spillR: (maxR - box.right) / scale,
    }
  })
  if (!r) { console.log(`p${n}: 找不到 layout 元素`); continue }
  const oh = Math.max(r.sh - r.ch, r.spillB)
  const ow = Math.max(r.sw - r.cw, r.spillR)
  if (oh > 4 || ow > 4) {
    bad++
    console.log(`p${n}: 溢出 高+${oh}px 寬+${ow}px(內容 ${r.sw}×${r.sh} / 版面 ${r.cw}×${r.ch})`)
  }
}
await browser.close()
kill()
console.log(bad === 0 ? `measure-slides(${deck}): 全部 ${total} 頁無溢出` : `measure-slides(${deck}): ${bad} 頁溢出`)
process.exit(bad === 0 ? 0 : 1)
