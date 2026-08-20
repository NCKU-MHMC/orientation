// 列出每頁 h1 之後緊接的元素(套用主題的 h1 + p 淺灰註解樣式與否)。
// 用法:node scripts/audit-h1.mjs lecture-01.md
import { chromium } from 'playwright-chromium'
import { spawn } from 'node:child_process'

const deck = process.argv[2] ?? 'lecture-01.md'
const port = 3097
const server = spawn('npx', ['slidev', deck, '--port', String(port), '--remote'], {
  stdio: 'ignore', detached: true,
})
const kill = () => { try { process.kill(-server.pid, 'SIGTERM') } catch {} }
process.on('exit', kill)
for (let i = 0; i < 120; i++) {
  try { if ((await fetch(`http://localhost:${port}/`)).ok) break } catch {}
  await new Promise((r) => setTimeout(r, 1000))
}
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
await page.goto(`http://localhost:${port}/1`, { waitUntil: 'networkidle' })
const total = await page.evaluate(() => __slidev__?.nav?.total ?? 0)
for (let n = 1; n <= total; n++) {
  await page.goto(`http://localhost:${port}/${n}`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(250)
  const r = await page.evaluate(() => {
    const el = document.querySelector('.slidev-page .slidev-layout')
    const h1 = el?.querySelector('h1')
    if (!h1) return { title: '(無 h1)', next: '', op: '' }
    const nx = h1.nextElementSibling
    return {
      title: h1.textContent.trim(),
      layout: [...el.classList].join('.'),
      next: nx ? `${nx.tagName.toLowerCase()}${nx.className ? '.' + String(nx.className).split(' ')[0] : ''}` : '(無)',
      op: nx ? getComputedStyle(nx).opacity : '',
      text: nx ? nx.textContent.trim().slice(0, 48) : '',
    }
  })
  console.log(`${String(n).padStart(2)} | ${r.title} | ${r.next} op=${r.op} | ${r.text}`)
}
await browser.close()
kill()
process.exit(0)
