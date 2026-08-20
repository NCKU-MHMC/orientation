# R2 — Domain Review(生成式建模領域審查)

Reviewer: R2, domain expert(divergences, guidance, RLHF/DPO/DDO, AR/VAE/GAN/DPM-FM)
Object under review: `docs/deck-blueprint.md`(against `docs/Generative_Models_Intro_Two_Session_Outline_EN.md` and `docs/demo-plans/`)
Date: 2026-08-16

Page numbers cite the blueprint's L1/L2 page tables.

---

## Findings

1. **[MUST] L2 p20 — 「下標改為 $q$ 得 reverse KL 分解」在數學上是錯的。**
   只把期望的下標從 $x_{<t}\sim p$ 換成 $x_{<t}\sim q$ 得到的是
   $\sum_t\mathbb{E}_{x_{<t}\sim q}[\mathrm{KL}(p_t\|q_t)]$,它既非 forward KL 也非 reverse KL。
   Reverse KL 的鏈式分解是 $\mathrm{KL}(q\|p)=\sum_t\mathbb{E}_{x_{<t}\sim q}[\mathrm{KL}(q(\cdot\mid x_{<t})\|p(\cdot\mid x_{<t}))]$:
   前綴分布**與**每步 KL 的引數順序要同時換。大綱原文(S2 ⑤ AR 第 3 點)同樣寫得過鬆,
   藍圖把它放上「公式解剖」風格的頁面,學生一驗算就會抓到。
   **修法**:p20 改寫為「reverse KL 的分解中,前綴由 $q$ 演化且每步比較的是
   $\mathrm{KL}(q_t\|p_t)$;訓練目標從未涉足這條軌道」,並給出完整式子。

2. **[SHOULD] L2 p9 — 「無正規化密度換得一步生成」的因果敘述誤導。**
   同一張矩陣(p8)裡 Normalizing Flow 就是反例:exact logprob 且 one-step sample,
   可見「快」不是用「沒有密度」買來的。GAN 放棄密度換到的是**架構自由**
   (generator 不受可逆性與 Jacobian 約束),因而能在一步內達到高品質;
   速度本身來自 one-step 映射,與有無密度無關。
   **修法**:p9 第三件事改寫為「放棄正規化密度,換得不受可逆性限制的 generator 架構,
   使一步生成也能維持品質(對照 NF:一步但受可逆性約束)」。

3. **[SHOULD] L2 p7 — interface-contract demo 的擺位偏離其設計規格,且未記錄偏離。**
   `demo-plans/03_介面契約檢查器.md` 規定:主用點在**第一堂①立契約時**(60 秒),
   第二堂③介面矩陣時 30 秒回放(展示腳本第 3 步:guide 滑桿只有 GAN 不動,
   對應矩陣的「無」格)。藍圖只放一次,且放在 L2 第②節末(p7),位於矩陣頁(p8–11)之前,
   demo 的核心教學時刻(灰色 guide 鈕 ↔ 矩陣空格)因此與其對應頁面脫節。
   S1 ① 不放可以辯護(demo 出現 AR/Flow/VAE/GAN 四張模型卡,違反第一堂
   「不指涉具體模型」的紀律),但這是與 demo 規格的衝突,應明文記錄裁決;
   且即使只留在第二堂,也應在③補 30 秒回放或直接把 p7 移到 p8–9 之後。
   **修法**:(a) 在藍圖註記「S1 ① 擺位因第一堂紀律取消」;(b) p8 或 p9 加 demo 回放列。

4. **[SHOULD] L1 p22–23、L2 p22 — 結果與方法頁未列支撐論文,違反大綱 production rule 3。**
   大綱參考文獻節明列且指定用途:Sanchez et al. (2023)(CFG 的 LLM 版,「used in Session 1」)、
   Li et al. (2023) contrastive decoding、Chuang et al. (2024) DoLa、
   Karras et al. (2024) Autoguidance——p22–23 方法表一個都沒名列。
   L2 p22(false premise 背景)未引 Kalai et al. (2025) 與 (QA)²,而大綱把它們列在
   「AR and comparability」節。相比之下 p29(Xie、Liu)、p40(Kirk)、p41/47(Zheng)、
   p31(Arjovsky & Bottou)都有引,標準不一致。
   **修法**:p22–23 各列補作者年份欄;L2 p22 補 Kalai et al. (2025)、Kim et al. (QA)²。

5. **[SHOULD] L1 p49 — 總結頁默默吞掉大綱⑤的兩個必列項。**
   大綱要求明說:(1) 本堂刻意假設 $\pi_{\text{ref}}$ 兩介面皆備;(2) 兩個開放問題——
   「各方法用什麼代理補 reverse KL 缺的 logprob」與「不提供 logprob 的模型如何訓練」。
   藍圖只留了第二個開放問題。陳述假設本身不是被寫作規格禁止的「預告下一堂」,
   而第一個開放問題正是 L2 p3 代理表的設問,砍掉它會讓 L2 p3 失去回應對象。
   **修法**:p49 補「本堂假設 $\pi_{\text{ref}}$ 提供兩介面」與代理開放問題。

6. **[SHOULD] L1 p25/p37/p48 — SpectrumRows 的列編號與大綱 B-0 三列對不上。**
   大綱光譜三列為:第 1 列訓練目標(forward KL/JSD/reverse KL)、第 2 列解碼設定、
   第 3 列權重微調。藍圖在②(解碼/guidance 節)的 p25 標「第一列」,p37 標「第三列起點」,
   而第 1 列(訓練目標)在①建立後從未被任何 SpectrumRows 呼叫畫出,第 2 列從未被點名。
   **修法**:p18(小結,三散度對照)畫第 1 列;p25 改標第 2 列;p37 維持第 3 列。

7. **[SHOULD] L2 p45 — 參考文獻頁只剩四個自學資源,大綱的論文清單無處落地。**
   大綱 References 分七組列出約三十篇支撐論文。若採「逐頁引用、末頁只放自學資源」策略,
   需先補齊 finding 4 的逐頁引用;否則 p45 應擴為完整文獻頁(可拆兩頁)。
   **修法**:二擇一並在藍圖寫明採哪一種。

8. **[CONSIDER] L2 p38 — Timeline 元件上 DDPM(2020)→Score-SDE(2021)→DDIM(2020) 的年份會被讀成錯誤。**
   DDIM(arXiv 2020-10)與 Score-SDE(arXiv 2020-11, ICLR 2021)幾乎同時;
   在標年份的時間軸上 2020→2021→2020 看起來像排序 bug。
   **修法**:改為 DDPM→DDIM→Score-SDE,或在頁面註明此序為概念序(離散→連續)非年代序。

9. **[CONSIDER] L2 p27 — VAE demo 敘述漏掉大綱 demo 表的第一個目的。**
   大綱:「showing the over-smoothing caused by mode covering, the failure to cover a ring
   topology, and the two ways things break when β…」。藍圖只寫後兩者。過度平滑正是 p26
   特徵性失效的可視化,是這個 demo 與該家族論證的主連結。
   **修法**:p27 補「mode covering 造成的過度平滑」為第一觀察項。

10. **[CONSIDER] L2 p32 — mode collapse 四層成因只呈現第一層,其餘三層無任何蹤跡。**
    大綱允許「at least the first」,但另外三層(如 Metz et al. 2017 Unrolled GANs、
    Salimans et al. 2016 minibatch discrimination 所對應的層次,兩篇都在大綱文獻清單)
    被無聲刪除。**修法**:講稿(speaker notes)留一行指向其餘層次與兩篇文獻即可,不加頁。

11. **[CONSIDER] L2 ⑤ 各家族頁 — B-4 矩陣與 B-5 三角「returned to in each segment」未落實。**
    Appendix B 規定兩圖在⑤每段回接;藍圖各家族段落無任何矩陣/三角的回放標記,
    只有 p13 一句總綱。**修法**:各家族節標頁(p14/24/29/35)加小型定位圖
    (FamilyMatrix/Trilemma 以 highlight prop 重用)。

12. **[CONSIDER] L2 p43–44 — 課程主張未落在最後一張實質投影片。**
    大綱:thesis「returned to on the last slide of Session 2」。藍圖 p43 回主張、
    p44 收尾(層次與介面),之後是文獻與結尾頁。**修法**:對調 p43/p44,
    或把主張併入 p44 作結語末句。

13. **[CONSIDER] L2 p39 — 把所有 zero-shot 編輯等同於 CFG 比值形式,涵蓋面過寬。**
    「$p_A$ 設為以原圖為條件、$p_B$ 為無條件、$w$ 控制改動幅度」確有實例
    (InstructPix2Pix 的雙 guidance scale 正是此形),但標準 inpainting
    (replacement/RePaint)與 SDEdit 不是 guidance 權重法。此頁亦無引文。
    **修法**:措辭改為「其中一類 zero-shot 編輯可寫成統一式」,並點名一個
    嚴格符合此形式的方法作為支撐。

---

## Coverage 核對摘要(無另立 finding 者)

- 大綱兩堂所有節、六題對照表(L1 p3、L2 p42)、光譜三列、介面表左右半
  (L1 p14、L2 p3)、三難(L2 p12–13)、四家族皆四段結構、DPM 段 zero-shot 編輯
  (L2 p39)、收尾語(L2 p44)均有對應頁。
- 六張板書圖 B-0/B-C/B-1/B-2/B-3/B-4/B-5 全數有元件對應。
- 九個 demo 擺位除 finding 3 外均符合大綱 demo 表與 demo-plans
  (guidance-playground→L1②、mle-vs-ddo-gradient→L1④、asr-noisy-channel→L1③第4層、
  exposure-bias-track→L2⑤AR、vae/gan/flow-matching→L2⑤各段)。
  divergence-2d-interactive 不在大綱 demo 表內,置於 L1 p13(B-1 之後)概念上正確。
- Appendix D(量測模組)為大綱明訂的 optional/獨立工作坊,藍圖未排入屬合法裁決;
  大綱「建議新 demo」前三項(token browser、校準散點、語意熵)均未實作成 HTML,
  藍圖未排屬實情,不列缺失。
- 核心數學抽查無誤:$2\,\mathrm{JSD}-2\log2$(p17)、$\sqrt{\mathrm{JSD}}$ 度量性(p15)、
  RLHF 閉式解與 reverse KL 等價(p38)、DDO 判別器與 $p_\theta^*\propto p_{\text{ref}}^{1-1/\beta}p_{\text{data}}^{1/\beta}$
  即②表末列(p41/46)、non-saturating $=\mathrm{KL}(p_g\|p_{\text{data}})-2\,\mathrm{JSD}$
  歸於 Arjovsky & Bottou 2017(p31)、BPB 式(p18)、trilemma 歸於
  Xiao, Kreis & Vahdat 2022(p12)、各 DPM 里程碑年份(finding 8 除外)均正確。
