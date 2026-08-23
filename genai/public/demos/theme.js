/* canvas 繪圖用的配色單一來源,與 theme.css 的 CSS 變數同值。
   用法:<script src="theme.js"></script> 後以 DEMO.ink、DEMO.a(DEMO.data, 0.3) 取用。
   禁止在各 demo 的繪圖程式裡寫死色碼。 */
window.DEMO = (() => {
  const T = {
    paper: '#f4f2ed', // 頁面底色
    paper2: '#fbfaf7', // 面板底色
    ground: '#edeae2', // canvas 底色
    ink: '#191817', // 主要文字、主線條
    ink2: '#4a453d', // 次要文字
    muted: '#7a736a', // 刻度、註記
    rule: '#ddd8ce', // 細分隔線
    rule2: '#c9c2b5', // 較深的分隔線
    grid: '#dfdacf', // 網格線

    accent: '#c05a32', // 焦點色(clay),同 model
    accentInk: '#9c4620', // 焦點色的文字版
    accentTint: '#f0e2d8',

    data: '#2f5d7c', // p_data / 目標分布 / 真實樣本
    model: '#c05a32', // p_theta / 模型分布 / 生成樣本
    third: '#5f7350', // 混合分布、引導後、第三方
    warn: '#a03a2e', // 失效、警示、擾動
  }
  // 十六進位色加上透明度,回傳 rgba()
  T.a = (hex, alpha) => {
    const h = hex.replace('#', '')
    const n = parseInt(h.length === 3 ? h.replace(/./g, (c) => c + c) : h, 16)
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`
  }
  // 資料密度熱圖用的色階:0 = 紙面,1 = 焦點色。t 落在 [0,1]。
  T.heat = (t) => {
    const s = [
      [0.0, 237, 234, 226],
      [0.35, 214, 209, 195],
      [0.62, 176, 168, 148],
      [0.84, 192, 90, 50],
      [1.0, 122, 42, 18],
    ]
    const v = Math.min(1, Math.max(0, t))
    for (let i = 1; i < s.length; i++) {
      if (v <= s[i][0]) {
        const a = s[i - 1], b = s[i], k = (v - a[0]) / (b[0] - a[0])
        return [a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k, a[3] + (b[3] - a[3]) * k]
      }
    }
    return [122, 42, 18]
  }
  // 類別色序:同一張圖裡多條線/多個群的預設順序
  T.series = [T.data, T.model, T.third, T.ink2, T.warn]
  T.font = {
    mono: "'IBM Plex Mono', ui-monospace, monospace",
    sans: "'Source Sans 3', 'Noto Sans TC', system-ui, sans-serif",
  }
  return T
})()
