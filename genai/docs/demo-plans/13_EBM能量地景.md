# Demo 13:Energy-Based Model 能量地景

**檔名** `public/demos/ebm-2d-interactive.html`(+ 同名 `.check.mjs`)
**課程位置** 第二堂⑤ EBM 段
**展示的單一概念** $p(x)=e^{-E(x)}/Z$:能量函數給的是**差一個常數 $\log Z$ 的 logprob**;
sample 沒有直接手段,要靠 Langevin dynamics 多步迭代,且模態相距遠時混合慢。

## 畫面與互動
- 主 canvas:能量地景熱圖 + 等高線;其上疊多條 Langevin 鏈的動畫
  ($x_{t+1}=x_t-\eta\nabla E(x_t)+\sqrt{2\eta}\,\xi_t$)。
- 能量函數(寫死,可切換):(a) 近距雙峰(兩個 Gaussian 井);(b) 遠距雙峰;
  (c) 環形谷。無訓練迴圈(訓練屬講稿內容)。
- 互動:
  1. 「啟動 / 暫停」:20 條鏈從均勻初始化開始走
  2. 步長 $\eta$ 滑桿、溫度(雜訊倍率)滑桿、「單步」按鈕
  3. 點 canvas 查 $E(x)$:讀數顯示 $-E(x)$ 並標註「$\log p(x) = -E(x) - \log Z$,$Z$ 未知」
  4. 能量函數切換;切到遠距雙峰後,計數器顯示「最近 500 步內跨峰的鏈數」
- 讀數列:步數、各峰的鏈佔比、跨峰次數。

## 實作要點(Tier 0)
- 純 JS + canvas;約 300 行;`CORE` 物件包 `energy(x, kind)`、`gradE(x, kind)`、
  `langevinStep(x, eta, temp, kind, rng)`,置於 `===CORE:BEGIN/END===` 區塊
- 梯度用解析式(Gaussian 混合的 ∇E 可解析);等高線用 marching squares 或密集取樣
- `body.embed` 緊湊模式同既有 demo

## 課堂展示腳本(90 秒)
1. 近距雙峰:鏈落谷、兩峰都有人口(25s)
2. 點查能量:能比較兩點高低,但絕對 logprob 差一個未知的 $\log Z$(25s)
3. 切遠距雙峰:同樣的 Langevin,幾乎沒有鏈跨峰;sample 介面的成本可見(40s)

## 驗收(check.mjs)
- `gradE` 與數值梯度一致(數點,±1e-4)
- 近距雙峰:2000 步 × 20 鏈後,兩峰人口比例都 > 20%
- 遠距雙峰:同設定下單鏈 2000 步跨峰次數 ≤ 近距情形的十分之一
- $e^{-E}$ 網格積分為有限正值(歸一化常數存在)
