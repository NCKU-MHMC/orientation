// 依 CSS selector 找出含該元素的頁面並截圖。用法:node scripts/shot.mjs lecture-02.md .tri outdir
import { chromium } from 'playwright-chromium'
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'

const deck = process.argv[2] ?? 'lecture-02.md'
const sel = process.argv[3] ?? '.tri'
const out = process.argv[4] ?? '/tmp/shots'
const port = 3098
mkdirSync(out, { recursive: true })

const server = spawn('npx', ['slidev', deck, '--port', String(port), '--remote'], {
  stdio: 'ignore',
  detached: true,
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
// selector 也可寫成 "p:3,7,12" 直接指定頁碼
const pages = sel.startsWith('p:') ? sel.slice(2).split(',').map(Number) : null
for (const n of pages ?? Array.from({ length: total }, (_, i) => i + 1)) {
  await page.goto(`http://localhost:${port}/${n}`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(300)
  if (pages || await page.locator(`.slidev-page ${sel}`).count()) {
    await page.screenshot({ path: `${out}/p${n}.png` })
    console.log(`p${n} 已截圖`)
  }
}
await browser.close()
kill()
process.exit(0)
