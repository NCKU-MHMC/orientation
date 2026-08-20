// 共用圖表樣式:所有 SVG 元件的字級與配色一律取自此處,禁止在元件內寫死。
export const typeScale = {
  title: 19, // 圖表標題
  label: 15, // 主要標註(曲線名、節點名)
  tick: 12.5, // 座標刻度
  note: 13, // 補充註記
}

export const palette = {
  p: '#2563eb', // 目標分布 p(data)
  q: '#d97706', // 模型分布 q(model)
  accent: '#059669', // 第三方(混合、引導後)
  bad: '#dc2626', // 失效、警示
  ink: '#1e293b', // 主要文字
  muted: '#64748b', // 次要文字
  grid: '#e2e8f0', // 網格、輔助線
  soft: '#f1f5f9', // 底色
}

export const fs = (key) => `${typeScale[key]}px`
