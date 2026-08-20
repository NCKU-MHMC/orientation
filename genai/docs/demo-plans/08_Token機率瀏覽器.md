# Demo 08:Token 機率瀏覽器(旗艦)

**課程位置** 第一堂②③(係數是軸上座標;第 2 層改取樣)
**展示的單一概念** temperature / top-p 直接對 logits 動手,而且 logprob 介面真實存在、
就在瀏覽器裡。

## 畫面與互動
- 輸入框:任意 prompt;逐步生成,每個位置顯示 top-20 機率長條
- temperature 與 top-p 滑桿:對「已快取的 logits」即時重新塑形,零延遲
- 讀數:每步熵、被截斷質量;整句累計 logprob
- 「重取樣此步」按鈕:同一位置依當前設定重抽,展示分岔

## 實作要點(成本 0,Tier 1;Tier 3 備援)
- **主方案:Transformers.js(瀏覽器內推論)**
  - 模型:SmolLM2-135M-Instruct(約 135MB,WASM 可跑)或 Qwen2.5-0.5B(WebGPU 較佳)
  - 拿得到完整 logits,這是雲端免費層常拿不到的
  - 模型檔經瀏覽器快取,課前在講課機器上預載一次,課堂完全離線
- 滑桿零延遲的關鍵:推論時把每步原始 logits 存進記憶體,滑桿只做後處理
  (除以 T、截斷、renormalize),不重新推論
- **備援(Tier 3,現場臨時換 prompt 且本機跑不動時)**:
  - Hugging Face Inference(免費層,text-generation 回傳 token 細節)
  - OpenRouter :free 模型(logprobs 視底層供應商;備課時實測一次)
  - Groq / Google AI Studio 免費層(速度快/額度穩,但 logprobs 支援需確認,
    只當「生成文字」備援,分布視覺化退化為僅顯示已選 token)

## 課堂展示腳本(2 分鐘)
1. 輸入實驗室相關 prompt,逐步生成,看每步分布(40s)
2. 拉 temperature:同一步的長條被壓平/銳化,熵讀數同步變(30s)
3. 拉 top-p:尾巴整塊消失,「被截斷質量」讀數上升,連回 DDO 段
   「top-p 是人為降溫」的實驗觀察(30s)
4. 收尾:「這就是 logprob 介面,它一直都在,你們只是沒呼叫過」(20s)

## 三方討論紀錄
- E:WebLLM(WebGPU)能跑更大模型,但要求較新 GPU;Transformers.js 有 WASM 退路,
  低階筆電也能跑 135M。**決議:Transformers.js 為主,偵測到 WebGPU 時自動換
  Qwen2.5-0.5B。**
- P:135M 模型的分布品質夠教學嗎?E:教 temperature/top-p 的機制不需要模型聰明,
  只需要分布真實。P 接受,但要求 demo 頁面註明模型大小,避免學生誤讀生成品質。
  **決議:採納。**
- D:最大風險是教室網路。**決議:課前預載 + Service Worker 快取列為部署必要程序,
  寫入驗收;現場斷網時整個 demo 仍可跑。**
- P:是否加 greedy/sampled 雙路徑對比?三方認定超出單一概念。**決議:不加,
  「重取樣此步」按鈕已足夠展示隨機性。**

## 驗收標準
- 斷網狀態下(模型已快取)完整可用
- 滑桿操作零延遲(不觸發推論);熵與截斷質量讀數正確
