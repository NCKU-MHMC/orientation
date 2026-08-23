// demo 頁截圖:node scripts/shot-demo.mjs <檔案>::<輸出png>[::embed] ...
// embed 表示掛上 body.embed(投影片 iframe 的緊湊模式)。會印出頁面的 JS 例外。
import { chromium } from 'playwright-chromium'
const b = await chromium.launch()
for (const spec of process.argv.slice(2)) {
  const [f, out, embed] = spec.split('::')
  const p = await b.newPage({ viewport: { width: 1280, height: embed ? 700 : 1100 } })
  const errs = []
  p.on('pageerror', e => errs.push(String(e)))
  await p.goto('file://' + f)
  if (embed) await p.evaluate(() => document.body.classList.add('embed'))
  await p.waitForTimeout(3500)
  await p.screenshot({ path: out, fullPage: !embed })
  if (errs.length) console.log(f.split('/').pop(), 'ERRORS', errs)
  await p.close()
}
await b.close()
