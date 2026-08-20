# Demo 07:BPB / Tokenizer 不變性

**課程位置** 第二堂⑤ AR 段(跨 tokenizer 比較)
**展示的單一概念** 逐 token PPL 不可跨 tokenizer 比較;BPB 可以。
$\mathrm{BPB}=\frac{T}{N_{bytes}}\log_2\mathrm{PPL}_{token}$

## 畫面與互動
- 上半:同一段文字以兩個模型各自的 tokenizer 上色切分(彩色 span 顯示 token 邊界),
  兩行並排,邊界明顯不同
- 下半:兩模型的表格:token 數、$\sum\log p$、token-level PPL、BPB
- 五段預設文字(中文、英文、程式碼、混合、重複字串)切換
- 附加檢視:同一模型內,把 token 逐一點開看 chain rule 的累加,總和不因顯示粒度而變

## 實作要點(成本 0,Tier 2:課前預計算)
- 課前在實驗室機器以 Ollama(免費、本地)跑兩個小模型(不同 tokenizer,
  例:Qwen2.5-0.5B 與 SmolLM2-360M 的 GGUF 版),對五段文字輸出逐 token logprob,
  存成靜態 JSON(<100KB)
- 前端純渲染,約 200 行;token 上色用預計算的邊界資訊
- 備選:tokenizer 部分可用 Transformers.js 的 tokenizer-only 模組現場切分(輕量),
  但 logprob 仍用預計算,避免課堂載入整個模型

## 課堂展示腳本(60 秒)
1. 中文段:兩模型 token 數差一截,PPL 數字完全不能比(20s)
2. 指 BPB 欄:分母換成 bytes 後才在同一尺度上(20s)
3. 點開累加檢視:同一字串的總 logprob 與切法無關,chain rule telescoping(20s)

## 三方討論紀錄
- E:原案兩個模型都在瀏覽器跑,每個 100–500MB,教室網路與低階筆電都撐不住。
  **決議:降為 Tier 2 預計算,前端零依賴。**
- P:糾正原始構想的一個概念錯誤:「同字串 logprob 與 tokenization 無關」只在
  同一個底層分布下成立;跨模型比較的重點是 BPB 歸一化,兩件事必須分開兩個檢視呈現,
  否則會教出錯誤結論。**決議:拆成「跨模型 BPB」與「同模型 telescoping」兩個分頁。**
- D:五段預設文字中,「重複字串」段落是驚喜點(PPL 极低),保留作為收尾彩蛋。
  **決議:採納。**

## 驗收標準
- 兩分頁概念隔離明確;同模型分頁中任意顯示粒度下總 logprob 恆等
- JSON 由腳本一鍵重新生成(附 Ollama 預計算腳本於 repo)
