// 元件驗收:SVG 字級一律取自 chart-style.js,禁 opacity 屬性(UnoCSS attributify
// 會把 opacity="1" 解析成 opacity:0.01)。
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

const dir = 'components'
let failed = false
const report = (f, line, msg) => {
  failed = true
  console.error(`${dir}/${f}:${line}: ${msg}`)
}

for (const f of readdirSync(dir).filter((f) => f.endsWith('.vue'))) {
  const lines = readFileSync(join(dir, f), 'utf8').split('\n')
  const usesSvg = lines.some((l) => l.includes('<svg'))
  lines.forEach((l, i) => {
    // SVG/模板內寫死字級(數字直書)。CSS 的 rem 允許,px 數字不允許。
    if (usesSvg && /font-size\s*[:=]\s*["']?\d/.test(l) && !l.includes('rem'))
      report(f, i + 1, '寫死 font-size,應改用 chart-style.js 的 typeScale/fs()')
    // opacity 屬性(靜態或 v-bind)都會踩 UnoCSS attributify 陷阱
    if (/\s:?opacity\s*=\s*"/.test(l) && !/fill-opacity|stroke-opacity/.test(l))
      report(f, i + 1, 'SVG opacity 屬性會被 UnoCSS 攔截,改用 :style="{ opacity: … }"')
  })
}

if (failed) process.exit(1)
console.log('svg-fontsize: OK')
