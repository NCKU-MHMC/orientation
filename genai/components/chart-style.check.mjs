// 驗證圖表字級沒有退回硬寫的數字,以及最擠的幾頁還放得下 16:9。
// 執行:npm run check
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import assert from 'node:assert/strict'
import { CONTENT, PX, typeScale, budget } from './chart-style.js'

const HERE = dirname(fileURLToPath(import.meta.url))
const vue = readdirSync(HERE).filter((f) => f.endsWith('.vue'))

// 1) 任何 SVG 裡的 font-size 都必須綁 typeScale,不能寫死。
//    寫死的話會被 viewBox 縮放比放大成不同大小,這正是當初對不齊的原因。
for (const f of vue) {
  const src = readFileSync(join(HERE, f), 'utf8')
  const hardcoded = src.match(/font-size="[\d.]+"/g) ?? []
  assert.equal(
    hardcoded.length,
    0,
    `${f}: font-size 寫死了 ${hardcoded.join(', ')},請改成 :font-size="FS.*"`,
  )
  if (src.includes('<svg')) {
    assert.ok(src.includes('typeScale('), `${f}: 有 SVG 卻沒有引入 typeScale`)
  }
}

// 2) 換算後的實際字級要落在宣告的 px 上,誤差 < 0.15px。
const CASES = [
  ['DecompAxes', 700, CONTENT.w],
  ['DivergenceFit', 720, CONTENT.w],
  ['TokenBars', 700, CONTENT.w],
  ['TwoTracks', 700, CONTENT.w],
  ['JsdSaturate(兩張並排)', 340, (CONTENT.w - 12) / 2],
  ['Trilemma(預設 700px)', 560, 700],
  // 第二堂
  ['GuidanceLadder', 700, CONTENT.w],
  ['GuidanceShift', 700, CONTENT.w],
  ['TempTopP', 700, CONTENT.w],
  ['IclBayes', 700, CONTENT.w],
  ['DdoMechanism', 700, CONTENT.w],
  ['SemanticEntropy', 700, CONTENT.w],
  ['Calibration', 700, CONTENT.w],
]
for (const [name, vb, render] of CASES) {
  const FS = typeScale(vb, render)
  for (const key of ['title', 'label', 'small']) {
    const actual = FS[key] * (render / vb)
    assert.ok(
      Math.abs(actual - PX[key]) < 0.15,
      `${name} 的 ${key} 實際渲染成 ${actual.toFixed(2)}px,應為 ${PX[key]}px`,
    )
  }
}

// 3) 放了圖的投影片必須都塞得進 471px 的內容高度。
//    rendered = viewBoxH × (內容寬 / viewBoxW)
const px = (vbW, vbH, render = CONTENT.w) => vbH * (render / vbW)
const TIGHT = [
  // [頁名, 圖高(px), 說明框數, 其他固定元素(caption/圖例/尾句/表格)]
  ['澄清四 · JSD 在中間', px(720, 200), 1, 22 + 22 + 29],
  ['換一個散度', px(720, 200), 1, 22 + 32],
  ['配圖 B-2 · 兩條軌道', px(700, 240), 1, 8],
  ['③ 分類樹', px(700, 250), 1, 8],
  ['③ 生成三難', px(560, 285, 700), 0, 12 + 21],
  ['④ ELBO 間隙', px(700, 225), 1, 8],
  ['④ 重參數化', px(700, 200), 0, 8 + 50],
  ['④ VAE 四缺陷', px(700, 232), 0, 0],
  ['④ 對抗迴路', px(700, 215), 0, 8 + 50],
  ['④ GAN 改進 (1)', px(700, 200), 1, 8],
  ['④ DPM / Flow', px(700, 175), 1, 12 + 21],
  ['④ TokenBars', px(700, 205), 0, 4 + 42],
  ['② JSD 飽和', px(340, 150, (CONTENT.w - 12) / 2), 1, 4 + 22 + 8 + 18],
  // 第二堂
  ['L2 ① 六個介入點', px(700, 232), 1, 12],
  ['L2 ② 係數即座標', px(700, 205), 1, 12],
  ['L2 ② top-k/top-p', px(700, 200), 1, 12],
  ['L2 ③ ICL 後驗', px(700, 190), 1, 12],
  ['L2 ④ 分岔圖回放', px(700, 240), 0, 12 + 29],
  ['L2 ④ 配圖 B-3', px(700, 236), 1, 12],
  ['L2 ⑤ 語意熵', px(700, 218), 1, 12],
  ['L2 ⑤ 校準', px(700, 212), 1, 12],
  ['L2 ⑥ Punchline', px(700, 232), 1, 16 + 16],
]
for (const [name, chart, notes, extra] of TIGHT) {
  const left = budget(notes, extra) - chart
  assert.ok(left >= 0, `${name} 超出 16:9 版面 ${(-left).toFixed(0)}px`)
  console.log(`${name.padEnd(22)} 圖 ${chart.toFixed(0)}px,餘裕 ${left.toFixed(0)}px`)
}

console.log('chart-style ok')
