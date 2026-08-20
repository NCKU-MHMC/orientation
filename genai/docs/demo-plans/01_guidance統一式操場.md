# Demo 01:Guidance 統一式操場

**課程位置** 第一堂②(引導生成的統一形式)
**展示的單一概念** $\log p_{guided} = \log p_{base} + w(\log p_A - \log p_B)$,係數 $w$ 是軸上的座標。

## 設計目標
讓「所有旋鈕是同一個旋鈕」從表格變成可拉動的桿子。學生拉 $w$ 時,同一個畫面依序重現
temperature、CFG、contrastive decoding 的行為。

## 畫面與互動
- 主畫面:20 個 token 的類別分布長條圖(仿 next-token distribution),三組可切換的預設
  logits:base、$p_A$、$p_B$
- 控制:$w$ 滑桿(−3 到 +3)、預設情境按鈕(temperature / CFG / contrastive / **prompt
  engineering**)、「對數空間檢視」開關
- 讀數:熵 $H$、top-1 機率、被截斷質量
- prompt engineering 預設的特殊行為:按下後只更換 base 的 logits,$w$ 滑桿變灰不可動,
  畫面標註「此手法不在係數的位置上」

## 實作要點(成本 0,Tier 0)
- 純 JS + canvas,logits 陣列寫死;guided = softmax(base + w*(A−B));約 150–200 行
- 對數空間檢視:y 軸切換為 logit 值,三條折線顯示線性組合真的是線性

## 課堂展示腳本(90 秒)
1. temperature 預設,拉 $w$:分布變尖再變平(30s)
2. CFG 預設,同一根桿子:條件強化(20s)
3. prompt engineering 預設:桿子變灰。提問「為什麼這個手法動不了係數?」(40s)

## 三方討論紀錄
- E:連續高斯分布畫起來比較平滑,建議用兩個 Gaussian。
- P:反對。類別長條直接對應 token 分布的心智圖像,且 renormalize 前後的質量搬移在長條圖
  上看得見;高斯太抽象。**決議:類別分布。**
- D:只拉桿子看分布變形,學生記不住「線性」這件事,要求加對數空間開關,讓折線的
  平移與縮放肉眼可見。**決議:採納。**
- P:追加 prompt engineering 灰桿設計,對應課程結論 3(prompt 不在係數位置上),
  這是第一堂最需要體感的一句話。**決議:採納,列為驗收必要項。**

## 驗收標準
- 四個預設情境切換順暢;灰桿狀態有明確視覺與文字說明
- 對數空間檢視中,$w$ 變動時折線呈嚴格線性位移
