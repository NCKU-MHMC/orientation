// 圖表字級的單一來源。
//
// 問題:每個 SVG 的 viewBox 寬度不同,卻都被拉伸到相同的版面寬度,
// 所以同樣寫 font-size="11",在 viewBox 300 的圖裡是 15.7px,
// 在 viewBox 340 的圖裡只有 13.8px。字級因此永遠對不齊。
//
// 解法:字級一律以「投影片上的實際 px」宣告,再依各自的縮放比換算回
// user unit。改 viewBox 尺寸時字級不會跟著跑掉。

// Slidev 預設畫布 980×551.25(16:9);.slidev-layout 的 px-14 py-10 = 56 / 40。
export const CANVAS = { w: 980, h: 551.25 }
export const CONTENT = { w: 868, h: 471 }

// 與投影片 HTML 的字級同一把尺:
//   title 14px = text-sm(圖內標題、節點文字)
//   label 12.5px        (座標軸、資料標籤)
//   small 11.5px ≈ text-xs(註腳、圖例)
export const PX = { title: 14, label: 12.5, small: 11.5 }

/**
 * 建立某個 SVG 的字級表。
 * @param {number} viewBoxW 該元件 viewBox 的寬度(user unit)
 * @param {number} [renderW] 該 SVG 在投影片上實際佔的寬度(px);預設為整個內容區寬
 * @returns {{title:number,label:number,small:number,u:(px:number)=>number}}
 */
export function typeScale(viewBoxW, renderW = CONTENT.w) {
  const u = (px) => +(px * (viewBoxW / renderW)).toFixed(2)
  return { title: u(PX.title), label: u(PX.label), small: u(PX.small), u }
}

// 內容區高度扣掉標題與說明框之後,還剩多少給圖。
// h1(1.7rem + mb-4)約 51px;一個 border-l-4 說明框(p-3 + 兩行 text-sm)約 78px。
export const H1 = 51
export const NOTE = 78

/**
 * 圖在某張投影片上可用的高度(px),用來反推 viewBox 高度上限。
 * @param {number} [notes] 這張投影片有幾個說明框
 * @param {number} [extra] 其他固定元素(圖例、caption)佔掉的 px
 */
export function budget(notes = 1, extra = 0) {
  return CONTENT.h - H1 - notes * NOTE - extra
}
