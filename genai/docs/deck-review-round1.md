# 審查回應(Round 1:藍圖組織)

Panel:R1 方法論、R2 領域、R3 教學、DA。原始報告在 `docs/deck-review-panel-r1/`。
本文件逐條裁決;「藍圖修訂」欄指向修訂後 `docs/deck-blueprint.md` 的落點。

## 駁回與部分駁回

**R3-1 / DA-3 / R1-11(demo「未建置、憑空發明」)— 前提錯誤,駁回主體。**
被點名的五個 demo(guidance-playground、asr-noisy-channel、mle-vs-ddo-gradient、
interface-contract、exposure-bias-track)全部已建置於 `public/demos/`,各附
`.check.mjs` 驗收與 `docs/demo-plans/01–06` 規格書(含展示腳本與三方決議)。
大綱的 demo 表只列三個 2D demo 是因為它成文於 demo 建置之前;demo-plans README
即為「recorded mapping」。接受的部分:
(a) 藍圖補一張「大綱建議 demo ↔ 實際 demo」對應表(見藍圖 Demo 對應節);
(b) token probability browser(建議第一優先)未建置,其兩個論證目標由
guidance-playground 覆蓋:20-token 類別分布長條 + w 滑桿即「係數是光譜座標」,
逐 token 機率條與熵讀數即「logprob 介面存在」的體感(以寫死 logits 呈現;
真模型版屬 Tier 1,另案)。calibration scatter 與 semantic entropy 隨量測模組
一併移出本課(見 DA-2 裁決)。

**R3-1 後半(砍 asr-noisy-channel、interface-contract)— 駁回。**
兩者各服務一個不可省的論點(第 4 層的 log 線性組合;介面矩陣的「無」格),
且已建置、離線可跑,砍除不省任何成本。

**DA-8(a)(ContractCard 全頁常駐)— 部分接受。**
以 `global-bottom.vue` 在每頁角落放縮小契約徽章,兌現「黑板角落」原意;
但保持極小字級,不與正文爭奪注意力。

## 接受(結構性修訂)

| # | 發現 | 藍圖修訂 |
|---|---|---|
| R1-1 | DPO 被引用三次但從未教 | L1 §④ 於 RLHF 後插入 DPO 推導頁(閉式解代入偏好損失 → 隱式 reward;Rafailov et al., 2023) |
| R1-2 / R3-4 / DA-12 | demo 分鐘未入帳 | Demo 對應表加分鐘欄;①的 B-1 靜態頁與 divergence-2d 合併(R3-5)、②動機頁併入統一式頁(R1-13)騰出時間;各節標可捨頁 |
| R1-3 / R2-2 | 「無密度換得一步生成」被 NF 反例推翻 | 改述為架構自由:generator 不受可逆性或序列分解約束,一步仍能維持品質;NF 頁明寫對照 |
| R1-4 / R3-3 / DA-6 | L2 GAN p34 一頁四工 | 拆頁:DDO 原型與單向轉移論證併入結構極限頁之後獨立成頁;年表+應用另成一頁 |
| R1-5 / R2-6 / DA-7 | 光譜三列的漸進呈現斷裂、編號錯 | 列 1(訓練目標)在①小結頁畫出;列 2(解碼設定)在②結論頁與③第 2 層頁;列 3(權重微調)在④;⑤總結呈現全圖 |
| R1-6 / R2-4 | ②表的 RLHF/DDO 列與各方法列缺引文 | 表中補作者年份(Sanchez 2023、Li 2023、Chuang 2024、Karras 2024、Ouyang 2022、Zheng 2025);引文即支撐,不需前向指涉 |
| R1-7 | β 先於其動機出現;α 無下文 | β 的引入與數值動機(log p 達 10³ 量級)移到 DDO 構造頁;圖 B-3 與構造頁同式;不引入 α |
| R1-8 | σ(log p/q)=p/(p+q) 恆等式無頁可依 | ①判別器讀法頁以此恆等式作結 |
| R1-9 / R2-5 | 總結頁漏掉假設與代理開放問題 | 總結頁明列:本堂假設 π_ref 兩介面皆備;兩個開放問題(各方法用什麼代理、無 logprob 如何訓練),以開放問題語氣陳述 |
| R1-10 / R3-9 | 實驗數字放錯堂 | L1 p47 只留兩則定性觀察;具體家族與數字移至 L2 DPM 段 DDO 對照頁(查證 Zheng et al., 2025 原文數字後填入) |
| R1-12 / R3-7 / R3-12 | p29/p40/L2 p23 過載 | ICL 頁只留公式與 memory 一句,lost-in-the-middle 移至 RAG/微調頁;p40 拆為 β 後果頁與逐點計分極限頁;AR 改進史拆兩頁 |
| R1-14 | √JSD 度量性、false premise、PF-ODE 缺引文 | 補 Endres & Schindelin (2003)、Kalai et al. (2025)、Kim et al. (QA)²、Song et al. (2021) 於首次使用處 |
| R1-15 / DA-2 | Appendix D 懸空指標 | 第六題定位改為自足表述「logprob 介面讀數的可信度(校準)」;量測模組移出本課,決策僅記於編輯註 |
| R1-16 / R2-12 | 主張未落在最後實質頁;頁數不一致 | L2 收尾順序:定位表 → 分層結語 → 主張回收(最後實質頁)→ 文獻 → 結尾;頁數更正 |
| R2-1 | reverse KL 分解敘述數學錯誤 | exposure bias 頁改寫:前綴分布與每步 KL 引數同時對調,給完整式 |
| R2-3 / R3-14 | interface-contract 擺位偏離規格 | 記錄偏離:S1 ①不放(demo 含四張模型卡,違反第一堂不指涉具體模型的紀律);demo 頁移到介面矩陣頁之後 |
| R2-7 | 文獻策略未定 | 採「逐頁作者年份 + 末頁完整文獻(兩頁)」 |
| R2-8 | DPM 時間軸年序讀感 | 順序改為 DDPM → DDIM → Score-SDE |
| R2-9 | VAE demo 漏第一觀察項 | 補「mode covering 造成的過度平滑」為第一項 |
| R2-10 | mode collapse 其餘三層無蹤 | 講稿留一行指向 Metz et al. (2017)、Salimans et al. (2016) |
| R2-11 | B-4/B-5 未逐段回接 | 家族節標頁以 FamilyMatrix/Trilemma 的 focus prop 呈現定位小圖 |
| R2-13 | zero-shot 編輯涵蓋面過寬 | 改述為「其中一類可寫成統一式」,以 InstructPix2Pix(Brooks et al., 2023)為嚴格實例 |
| R3-2 / DA-4 | ①節 25 分鐘無現象錨點 | 開場加一頁 base vs aligned 實際輸出對照(標示為示意樣本);forward/reverse KL 頁各加一行現象帶(泛泛安慰語 ↔ 回答同質化) |
| R3-6 | I(X;Z) 超出前置知識 | 移入講稿;頁面留分類器白話讀法 |
| R3-8 | 光譜位置論證與 DPO 表混頁 | 拆開:光譜定位頁(SpectrumRows)獨立;DPO/DDO 對照表與 β 參數化同頁 |
| R3-10 | 回接頁易寫成後設敘述 | 寫作規格加一條:回接頁一律直接陳述事實,不敘述「回顧/回到」這個動作 |
| R3-11 | GAN 損失頁塞推導 | 頁面留對照表與逐項一句讀法;分解推導與引文入講稿 |
| R3-13 / DA-1 | 開場對照表第三欄不可讀 | L1 p3 只列題目與機率問題兩欄;三座標完整版留在 L2 定位頁 |
| R3-15 | L2 開場平淡 | L2 首頁內容即以「無 logprob 的模型如何訓練」的問題開場,代理表作答 |
| DA-8(b)(c) | DDO 低谷無鷹架 | DDO 前加前備頁(兩行:d* 恆等式、②表末列,直接陳述);「三個無需」移到機制圖頁 |
| DA-9 | 無休息點 | L1 於②後、L2 於④後各排休息頁,回接句不跨休息點 |
| DA-10 | 課前自檢無載體 | 產出 `docs/課前自檢.md` handout,L1 封面講稿附發放提示 |
| DA-11 | 頁數自述不一致 | 修正 |

## 維持原判(經覆核)

- DDO 佔④篇幅(DA 覆核為 non-defect):對應大綱明定 32 分鐘,維持。
- demo 間距(DA 覆核為 non-defect):維持。
- divergence-2d 不在大綱 demo 表(R2 覆核):置於 B-1 位置概念正確,維持(與靜態頁合併)。
