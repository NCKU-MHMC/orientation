# DA — Devil's Advocate(Round 2:成品攻擊)

對象:`lecture-01.md`、`lecture-02.md`(最終 Slidev 成品)。
基準:`docs/Generative_Models_Intro_Two_Session_Outline_EN.md`、`docs/deck-review-round1.md`。
Round 1 已裁決事項不重審,只驗收兌現;引文查證含 NVIDIA DDO 專案頁(數字已比對)。

---

## 最強反論(Strongest Counter-Argument)

全課主張是「base model 的含糊與 aligned model 的千篇一律,位於同一條 mode-covering–mode-seeking 光譜的兩端」。但兩端量的不是同一對分布:左端是 $\mathrm{KL}(p_{\text{data}}\|p_\theta)$,右端是 $\mathrm{KL}(\pi\|\pi^*)$,而 $\pi^*\propto\pi_{\text{ref}}\exp(r/\beta)$ 是偏好加權後的傾斜分布,不是 $p_{\text{data}}$。第一堂「reward model 的介面身分」頁把這個換軌藏進一句改寫:介面盤點頁明確定義 $p_{\text{data}}$ 為「資料」,該頁卻說 reward model「充當『人類覺得好的分布』的 logprob 代理」——同一個符號 $p_{\text{data}}$ 在課程中途換了指涉對象,而光譜圖從未標註這件事。一個夠尖銳的學生會問:「reverse KL 的 mode-seeking 是相對 $\pi^*$ 的眾數,不是 $p_{\text{data}}$ 的;若兩端的散度連引數都不同,『同一條光譜』是數學陳述還是修辭?」現行講稿沒有給講者任何一句可以接住這個問題的話。辯護是存在的($\pi^*$ 是 $\pi_{\text{ref}}$ 的銳化,故 mode-seeking 行為特徵仍成立;光譜刻畫的是散度的「方向」而非固定分布對),但這個辯護必須寫在頁面或講稿上,而不是留給講者臨場發明。與此同時,課程的收尾論證(結語頁的成本結構)被投影片自己的四層表直接反駁(見攻擊 1),等於開場主張與收尾建議兩頭都站在有裂縫的地基上。

---

## 攻擊清單

### A1【MUST】第 4 層要不要 logprob?兩堂課自相矛盾,而 L2 結語正建立在錯的那一邊

- L1「四層總覽」頁:「**第 1 層是唯一不需要 logprob 的層:任何黑箱 API 都能做**。」
- 同一份 deck 的 `LayerStack.vue` 第 4 列:「`sample(logprob 可選)`」,且程式碼把第 1、4 兩列同標 `sampleOnly` 樣式;L1 第 4 層頁講稿也寫「第 1 層 + 第 4 層都不需要 logprob」。大綱 ③ 表同:「layer 4 … sample (logprob optional)」。
- L2 結語頁:「實驗室現行方法多在**第 1 層**,只呼叫 `sample()`…**第 2 到 4 層額外需要的只有 `logprob`**;而這三層的方法**不需要任何額外訓練資源**。」

第 4 層(self-consistency、best-of-n、MBR)在黑箱 API 上今天就能跑;實驗室若已在用 self-consistency,結語的「往下三層走、增量只是 logprob」立刻被自己人的日常經驗戳破。**修法**:總覽頁改為「第 1、4 層僅需 sample」;結語的成本結構改寫為「黑箱 API 可達第 1+4 層;第 2、3 層才是 logprob 帶來的增量」,並相應修正「多在第 1 層」的敘述。

### A2【MUST】光譜兩端的散度引數不同,reward model 頁換掉了 $p_{\text{data}}$ 的指涉而未聲明

- L1 介面盤點頁:「$p_{\text{data}}$(資料)」。
- L1 reward model 頁表格:「reverse KL | 缺 $p_{\text{data}}$.logprob | **reward model**」,正文:「充當『人類覺得好的分布』的 logprob 代理」。
- L1 RLHF 頁:「$\min_\pi \mathrm{KL}(\pi\|\pi^*)$…這正是 reverse KL:對齊訓練住在 mode-seeking 端。」

RLHF 的 reverse KL 目標是 $\pi^*$,不是資料分布;「缺 $p_{\text{data}}$.logprob、由 reward model 補上」把兩個不同的目標分布焊在同一格。**修法**:reward model 頁加一行明示換軌——「對齊的目標分布不再是 $p_{\text{data}}$ 而是偏好分布 $p_{\text{good}}$(或 $\pi^*$);reward model 代理的是後者的 logprob」——並在講稿給出光譜為何仍成立的一句話(mode-seeking 是散度方向的性質,對任何目標分布都導致收斂到少數眾數)。

### A3【MUST】RLHF 閉式解的引文張冠李戴

L1 RLHF 頁:「這個目標有閉式最優解(**Ouyang et al., 2022 附錄**;變分法一步)」;②表 RLHF 列同樣掛 Ouyang et al., 2022。InstructGPT 論文定義了該目標,但其附錄並沒有 $\pi^*\propto\pi_{\text{ref}}\exp(r/\beta)$ 的變分推導;這個推導的標準出處是 Rafailov et al. (2023) Appendix A.1(本 deck 已引)或更早的 Peters & Schaal (2007)。任何學生翻開 arXiv 2203.02155 附錄五分鐘就能證偽這個指涉,且違反大綱 production rule 3(每個主張帶正確支撐)。**修法**:改掛 Rafailov et al. (2023);Ouyang et al. 保留在「目標」而非「閉式解」上。

### A4【MUST】開場「示意樣本」展示的不是它聲稱的現象

- L1 slide 2 左欄標「**base model,抽樣三次**」,三則樣本:「壓力大的時候睡眠通常會受影響…」「**這樣的情況持續多久了?**…」「睡不好有時和咖啡因有關…」。
- 講稿:「左側是 pretraining 後的**續寫**行為」;forward KL 頁:「開場看到的含糊、發散的**續寫**」。

三則樣本全是格式良好、對題的**回覆**,第二則還是高品質的追問——沒有一則是續寫。真實 base model 對這句輸入的典型行為(接著替使用者把話說完、跑成論壇串、重複)一個都沒出現;左欄實際展示的「三個不同的完整回答」恰恰是高溫 aligned model 也會給的東西,無法把左右欄的差異歸因到 pretraining vs alignment。標示方面,「(以下為示意樣本)」只出現在題句行內一次,兩個欄位標題「base model,抽樣三次」的具體措辭反而在暗示真實抽樣紀錄。**修法**:左欄至少一則改為可辨識的續寫失敗型態(接寫使用者的話、格式漂移),或把「續寫」的措辭全面改為「發散的回覆」;「示意樣本」標註移入(或複製到)兩個樣本框內。統計性質上右欄(同模板)成立,左欄需要重寫才配得上它被賦予的論證負載。

### A5【SHOULD】FID 表支撐不了它上方那句話

L2「另一條提速路線」頁:「DDO 微調後的模型**免 guidance 即達到原先開 CFG 的品質**,推論成本直接減半(不再需要每步兩次前向…)」,但表頭是「FID(前 → 後,**無 guidance**)」——兩欄都是無 guidance 數字,頁面上不存在任何「開 CFG 的基線」數字可供「達到」二字對照。數字本身與 NVIDIA 專案頁一致(1.79→1.30、1.58→0.97、1.96→1.26、4.74→1.79,均 guidance-free;「cutting the inference cost by half」也是原文主張),查證義務(R1-10)已兌現;缺的是表內的 CFG 基線欄。另外 EDM 在 CIFAR-10 的 1.79 本來就是無 guidance 的結果,對這一列「減半」是空話。**修法**:補一欄「基線 + CFG」FID(或footnote),「減半」限定於原本以 CFG 兩次前向推論的模型;講稿注明 CIFAR-10 列不適用減半敘述。附帶:大綱 ⑤ 寫的是「inference cost is unchanged」,與 deck 的「減半」兩者相對基準不同,應統一措辭以免講者連自己的大綱都對不上。

### A6【SHOULD】Kirk et al. (2024) 撐不起「與 KL 預算的鬆緊同步」

L1「β 是唯一的煞車」頁:「實測:RLHF 後輸出多樣性系統性下降,**且與 KL 預算的鬆緊同步**(Kirk et al., 2024)」。該文比較 SFT/BoN/RLHF 的多樣性與泛化,並未系統性掃 β/KL 預算來展示「同步」。前半句有支撐,後半句沒有。**修法**:刪去「且與 KL 預算的鬆緊同步」,或降格為講稿內的理論推論(由目標函數形式預期,而非該文實測)。

### A7【SHOULD】Kalai et al. (2025) 被綁在它不主張的句子上

L2 false premise 頁:「拒答作為輸出模式,**結構上沒有位置**,除非 post-training 明文補上(Kalai et al., 2025 從訓練目標與評測誘因分析幻覺的必然性)」。該文的論證是:預訓練的統計性質使錯誤不可避免(即使語料乾淨),而**二元計分的評測懲罰棄答**、獎勵猜測——它的處方是改評測,不是 post-training;它也不主張拒答在結構上不存在(模型能輸出 IDK,是誘因壓掉它)。括號內的描述本身是準確的,但它所依附的主句(語料問答配對 → 拒答無位置 → post-training 補上)是 deck 自己的論證,掛在引文後會被讀成該文結論。**修法**:引文只綁「幻覺必然性」半句;「拒答結構上沒有位置」改為本課自己的 forward KL 推論並如此標明,或補一句「Kalai et al. 的處方是改評測誘因」以免講者被讀過該文的學生問倒。

### A8【SHOULD】R1-5 承諾的光譜第二列「③第 2 層頁」沒有兌現

Round-1 裁決:「列 2(解碼設定)在**②結論頁與③第 2 層頁**」。L1「第 2 層.改變抽樣」頁只有 `<TempTopP />` 與正文,無 `SpectrumRows`;講稿寫「第二列光譜(解碼設定)**在上一節已畫出**」——把承諾降級成一句旁白。**修法**:該頁補 `<SpectrumRows :rows="2" mark="decoding" />`,或在 round-2 紀錄中明記偏離與理由。

### A9【SHOULD】「回到主張」「回到開場的主張」兩個標題違反 R3-10

Round-1 裁決 R3-10:「回接頁一律直接陳述事實,**不敘述『回顧/回到』這個動作**」。L1 尾段頁標題「**回到**開場的主張」、L2 倒數第三實質頁標題「**回到**主張」,正是被禁止的後設敘述,且與使用者既有的簡報文字風格要求(禁後設編排敘述)同向牴觸。**修法**:改為陳述句標題,如「兩種輸出,同一條光譜」/「主張,驗畢的形式」。

### A10【SHOULD】三難的「目前沒有任何家族做到」拿 2022 年引文撐 2026 年的現在式

L2:「三個目標同時滿足,**目前**沒有任何家族做到(Xiao, Kreis & Vahdat, 2022)」。同一份 deck 隨後自己列出蒸餾目標、Consistency Models、DDO 免 guidance——學生當場就能問:「一步蒸餾的 diffusion 不就是三個都有?」講稿沒有備答。**修法**:加一行(頁面或講稿):蒸餾模型繼承教師的覆蓋與品質上限、且需先訓練慢的教師,故是三難內的搬運而非突破;或把主張時間定格為引文年份。

### A11【SHOULD】第二堂作業沒有考第二堂

L2 結尾頁:「作業:把自己的題目放進定位表;**介面清單那一欄,親手填一次**。」L1 作業第 3 題已是「用到的每個方法,各呼叫了 `sample()` 還是 `logprob(x)`?」且「下次課前交一頁」——同一欄在第二堂開課前已經交過。第二堂 60 分鐘的家族巡禮(構造、代理、特徵性失效、三難)沒有任何作業或提問承載;學生可以完全沒聽 ⑤ 而交出滿分作業。**修法**:第二堂作業改考本堂內容,例:「為你的題目選一個非 AR 家族,寫出它兩個介面的實作形式與特徵性失效,並指出第一堂哪些方法因此適用/失效。」

### A12【SHOULD】光譜第一列首次亮相時,格子裡是三個未定義詞

`SpectrumRows` 列 1 內容:「forward KL・**MLE**」「JSD・**GAN**」「reverse KL・**RLHF**」。此列首次顯示於 L1 ①小結頁(`:rows="1"`),當下 MLE 未在任何投影片上定義(講稿明言 CE=MLE 推導留給第二堂)、RLHF 要到 ④ 才出現,GAN 更直接違反 L1 開場講稿的紀律宣告:「本堂**全程不涉及任何具體模型結構**」。**修法**:元件加一個 prop 在 L1 首次顯示時隱藏家族名(只留三個散度),或 ①小結頁的講稿逐一給三個名字各一句佔位定義;GAN 一詞至少應延到第二堂。

### A13【CONSIDER】「VI」與「energy 代理」全課零定義

L2「三個散度,三種補法」表:「代理:reward / **energy**;形成的家族:RLHF、**VI**」;`FamilyTree` 節點同。L1 收尾把「reverse KL 一族還有什麼代理」立為開放問題(「本堂只見過 reward model 一種」),L2 的回答是兩個沒有任何一頁、任何一句講稿解釋的名詞。學生問「VI 是什麼」即斷線。**修法**:講稿各給一句(VI:以可算的 $q$ 最小化 reverse KL,ELBO 即其實例;energy:以未正規化能量函數充當 log 密度差),或從表格與樹中刪去。

### A14【CONSIDER】interface-contract demo 只有四張卡,缺 DPM

demo 原始碼中的模型卡為 AR、Flow、VAE、GAN;它被安排在五家族矩陣與家族樹**之後**播放,唯獨缺矩陣中 logprob 欄最微妙的 DPM(「下界;經 PF-ODE 精確」)。**修法**:講稿補一句「DPM 卡的行為與 VAE 同型(下界),差在可經 PF-ODE 精確化」,或補卡。

### A15【CONSIDER】InstructPix2Pix「嚴格符合此式」——雙係數不是單係數式的實例

L2 CFG 頁先寫單係數統一式,再稱 InstructPix2Pix 的「雙 guidance scale(影像一個係數、指令一個係數)是**嚴格符合此式**的實例」。兩個係數對應兩個比值項的線性組合,是統一式的推廣而非該式本身;「嚴格」一詞主動邀請這個反駁。**修法**:改為「符合其雙比值項推廣」,或寫出兩項形式。

### A16【CONSIDER】死引文與 L1 的文獻真空

Chen et al. (2024) SPIN 與 Holtzman et al. (2020) 只存在於 L2 文獻頁,正文與講稿零次引用;L1 自身無文獻頁,而 L1 作業「下次課前交」——學生寫作業的那一週手上沒有任何完整書目。**修法**:SPIN/Holtzman 或刪或在對應頁補一次實引(top-p 頁掛 Holtzman 最自然);L1 作業頁加一行「完整文獻見第二堂末頁/課程 repo」。

### A17【CONSIDER】第 2 層與第 3 層的分界標準沒有陳述

L1 第 2 層頁:「temperature **把 logits 除以 $T$**」;第 3 層卻叫「改變 logits」。除以 T 為何不算改 logits,deck 未給判準,學生必問。**修法**:總覽頁講稿加一句判準(第 2 層:與內容無關的全域重塑與截斷;第 3 層:逐 token、內容相依的修改)。

### A18【CONSIDER】AR 節標頁掛著一張沒有 AR 的三難圖

`Trilemma.vue` 只支援 GAN/VAE/DPM 三點;VAE/GAN/DPM 節標頁各自 `focus` 高亮,AR 節標頁則放了一張無 focus、也永遠不可能出現 AR 的三難圖。全課從未把 AR 放上三角形,但 L2 定位表又要求為題目填「三難取捨」。**修法**:AR 節標頁刪去 Trilemma,或講稿一句話給 AR 定位(品質-覆蓋邊、序列抽樣付速度,與 DPM 同側)。

### A19【CONSIDER】「所有這些控制參數是同一個參數的不同名字」對 contrastive decoding 不成立

L1 三個結論頁:「$w$ 或 $1/\beta$ 越大,越往 mode-seeking 端;**所有這些控制參數是同一個參數的不同名字**。」contrastive decoding 的 $\lambda$ 放大的是 $\log p_{\text{strong}}-\log p_{\text{weak}}$,其方向由比值項決定,並不保證逐 token 熵下降,更不保證朝 $p_{\text{data}}$ 的眾數移動;「同一個參數」是修辭強度超過數學內容的句子。**修法**:改為「同一個**位置**的不同名字:係數放大比值項的作用,移動方向由比值項的選取決定」。

---

## 已驗收、不再攻擊的 Round-1 承諾(核對紀錄)

R1-1(DPO 推導頁)、R1-3(架構自由改述 + NF 對照)、R1-4(GAN 拆頁)、R1-6(②表引文)、R1-7(β 動機在 DDO 構造頁、無 α)、R1-8(σ 恆等式作結)、R1-9(假設 + 兩開放問題頁)、R1-10(L1 兩則定性、數字入 L2 且與原文相符)、R1-12(ICL 頁瘦身、p40 拆頁、AR 史拆頁)、R1-14(四處引文補齊)、R1-15(第六題自足表述)、R2-1(reverse KL 分解修正)、R2-3(demo 移位)、R2-8(DDPM→DDIM→Score-SDE)、R2-9(demo 第一觀察項)、R2-10(Metz/Salimans 講稿行)、R2-11(節標定位小圖)、R2-13(InstructPix2Pix 限縮——但見 A15)、R3-2(開場對照頁存在——但見 A4)、R3-5、R3-6、R3-11、R3-13、R3-15、DA-8(a)(`global-bottom.vue` 契約徽章存在)、DA-9(兩堂休息頁位置正確)、DA-10(`docs/課前自檢.md` 存在且 L1 講稿附發放提示)。五個 demo 檔案與 `.check.mjs` 均在 `public/demos/`,guidance-playground 含熵讀數與灰桿情境,與 round-1 抗辯相符。

## 統計

| 等級 | 數量 |
|---|---|
| MUST | 4(A1–A4) |
| SHOULD | 8(A5–A12) |
| CONSIDER | 7(A13–A19) |
