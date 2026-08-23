// 共用圖表樣式:所有 SVG 元件的字級與配色一律取自此處,禁止在元件內寫死。
// 與 style.css、public/demos/theme.css 同一套語意色;改配色只改這三個檔。
export const typeScale = {
  title: 19, // 圖表標題
  label: 15, // 主要標註(曲線名、節點名)
  tick: 12.5, // 座標刻度
  note: 13, // 補充註記
}

export const palette = {
  p: '#2f5d7c', // 目標分布 p_data:petrol blue
  q: '#c05a32', // 模型分布 q:clay,同時是全套簡報的焦點色
  accent: '#5f7350', // 第三方(混合、引導後):olive
  bad: '#a03a2e', // 失效、警示:brick
  ink: '#191817', // 主要文字
  muted: '#7a736a', // 次要文字、刻度
  grid: '#ddd8ce', // 網格、輔助線
  soft: '#edeae2', // 底色、填色
  paper: '#fbfaf7', // 面板底色
  rule: '#c9c2b5', // 較深的分隔線
  tint: '#f0e2d8', // 焦點色的淡底
}

// 十六進位色加透明度,SVG 的 fill/stroke 直接用
export const alpha = (hex, a) => {
  const n = parseInt(hex.slice(1), 16)
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`
}

export const font = {
  serif: "'Source Serif 4', 'Noto Serif TC', Georgia, serif",
  sans: "'Source Sans 3', 'Noto Sans TC', system-ui, sans-serif",
  mono: "'IBM Plex Mono', ui-monospace, monospace",
}

export const fs = (key) => `${typeScale[key]}px`
