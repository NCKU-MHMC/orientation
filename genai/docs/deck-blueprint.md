# 簡報藍圖:生成模型導論兩堂課(介面版,2026-08-16 重建;Round 1 修訂)

依據 `docs/Generative_Models_Intro_Two_Session_Outline_EN.md`。本文件為編輯用藍圖,
不對學員公開。審查記錄:`docs/deck-review-panel-r1/`(panel 原文)、
`docs/deck-review-round1.md`(裁決與修訂對照)。

## 產出物

- `lecture-01.md`:第一堂(120 min)「分布的度量與調控」
- `lecture-02.md`:第二堂(120 min)「建構滿足介面的分布」
- `components/`:示意圖元件 + `DemoFrame.vue`;`global-bottom.vue` 角落契約徽章
- `scripts/deck-lint.mjs`、`scripts/svg-fontsize.check.mjs`:驗收腳本
- `docs/課前自檢.md`:課前自檢 handout(大綱 Appendix A 六題)

## 最高優先寫作規格(高於大綱原文的措辭)

1. 投影片與講稿只載學科內容。嚴禁解釋編排理由、嚴禁預告「之後會補」;引用前文結論時
   直接陳述結論本身。回接頁一律陳述事實,不敘述「回顧/回到」這個動作。
2. 禁比喻性口語術語:「尺」「軸」「骨架」「骨幹」「體質」「招牌病」。以正式名詞替代並
   自然變換:散度、距離度量、mode-covering–mode-seeking 光譜、連續體、兩端。
3. 「不是 A,而是 B」句式每份 deck 至多 3 處。
4. 可見文字不用第二人稱「你」。
5. 學術散文規範:無宣傳語、無空洞強調、破折號連用改寫、段落長度自然變化。
6. 每個主張帶支撐:數學性質給推導或指向;實驗結果給論文;量化形容詞對應具體數字。
7. 技術名詞保留英文,不強譯。
8. 16:9 版面不塞滿:一頁裝不下就拆頁。圖以 js/css/svg 自繪,力求自明。
9. 散度性質頁就地連結 LLM 現象(現象帶),第一堂例子一律 LLM-native。

## 語言

正文繁體中文,技術名詞英文。

## 文獻策略

逐頁作者年份;末頁完整文獻兩頁。補充引文:Endres & Schindelin (2003)(√JSD 度量性)、
Brooks et al. (2023) InstructPix2Pix(統一式的 zero-shot 編輯嚴格實例)。

---

# lecture-01.md 「分布的度量與調控」(120 min,54 頁)

## ① 開場與散度的選擇(36 min,p1–19;含 demo 3 min)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 1 | 封面(講稿:發放課前自檢 handout) | |
| 2 | 現象對照(標示為示意樣本):同一問題,base model 多次抽樣的分歧與含糊 vs aligned model 的同質回答 | 對照盒圖 |
| 3 | 課程主張:兩種現象位於同一條 mode-covering–mode-seeking 光譜的兩端;本堂給出論證 | statement |
| 4 | 實驗室題目對照表(兩欄:題目/背後的機率問題) | HTML 表 |
| 5 | 判別式與生成式:$p(y\mid x)$ 輸出空間小而封閉;生成式學可抽樣的 $p_\theta(x)$,目標逼近未知 $p_{\text{data}}$ | 對照盒圖 |
| 6 | 高維 $p(x)$ 無法列表(二值影像 $2^{256\times256}$) | SVG 示意 |
| 7 | 介面契約:`sample()` / `logprob(x)`(此後每頁角落常駐徽章) | ContractCard |
| 8 | 介面盤點:$p_{\text{data}}$ 只有 sample(資料集是樣本集合,沒有密度);$p_\theta$ 本堂假設兩者皆備 | 盤點表 |
| 9 | 節標:逼近需要選擇散度 | section |
| 10 | KL 定義與不對稱:積分由 $p$ 加權 | 公式解剖 |
| 11 | Forward KL:懲罰無界→覆蓋整個支撐集;zero-avoiding / mode-covering。現象帶:訓練目標為 forward KL 的 base model 傾向廣覆蓋的含糊回覆 | KlZeros(forward) |
| 12 | Reverse KL:權重換成 $q$,丟眾數無懲罰;zero-forcing / mode-seeking。現象帶:對齊後回答同質化 | KlZeros(reverse) |
| 13 | Demo:divergence-2d-interactive(3 min;B-1 論述於此頁進行:單峰 $q$ 擬合雙峰 $p$ 的三種解) | DemoFrame |
| 14 | 散度的介面需求表(forward/reverse/JSD × 需要的介面 × 可得性) | 表 |
| 15 | JSD 定義;Jeffreys 對照;有界 $\le\log2$;$\sqrt{\mathrm{JSD}}$ 度量性(Endres & Schindelin, 2003) | 公式+小表 |
| 16 | JSD 飽和與梯度消失 | JsdSaturate |
| 17 | 判別器讀法:$D^*=p/(p+q)$;最優判別器下目標值 $2\,\mathrm{JSD}-2\log2$;白話:JSD 是最優分類器分辨樣本來源的能力(互資訊形式入講稿);頁末恆等式 $\sigma(\log p/q)=p/(p+q)$ | 推導 |
| 18 | 小結:三散度對照表(行為/對稱/上界/失效型態)+ 光譜列 1(訓練目標) | 表+SpectrumRows(1) |
| 19 | 課後練習:單高斯擬合 1D 雙峰,三散度各解 $\mu,\sigma$ | DivergenceFit |

## ② 引導生成的統一形式(22 min,p20–26;含 demo 3 min、討論 3 min)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 20 | 節標:分布固定之後(動機一句併入) | section |
| 21 | 統一式與逐項解剖(base 項/係數/比值項;再正規化;全式只呼叫 logprob) | GuidanceForm |
| 22 | 方法表(上):temperature、CFG for LLM(Sanchez et al., 2023)、contrastive decoding(Li et al., 2023)/ DoLa(Chuang et al., 2024),含所需介面欄 | 表 |
| 23 | 方法表(下):Autoguidance(Karras et al., 2024)、RLHF 最優解(Ouyang et al., 2022)、DDO 最優解(Zheng et al., 2025);top-k/top-p 為硬截斷版 | 表 |
| 24 | Demo:guidance-playground(3 min;同一滑桿重現 temperature/CFG/contrastive;prompt 情境的灰桿) | DemoFrame |
| 25 | 結論:係數是光譜座標;推論期與訓練期只差時機;適用範圍取決於 logprob 介面 | SpectrumRows(2, mark decoding) |
| 26 | 討論(3 min):prompt 只置換 base 項、不觸及係數;多樣性不足或過度銳化無法靠改 prompt 解決 | |
| 27 | 休息(第 58 分) | |

## ③ 推論期的四層介入(20 min,p28–36;含 demo 2 min)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 28 | 節標:四層介入 | section |
| 29 | 四層總表;第 1 層是唯一不需 logprob 的層 | LayerStack |
| 30 | 第 1 層.條件:ICL 是隱式貝氏推論(Xie et al., 2022);memory 是選擇進入後驗的證據 | 公式 |
| 31 | RAG 與 fine-tuning 的機率意義與失效型態;無關脈絡攤平 $p(\text{task}\mid\text{prompt})$(Liu et al., 2024) | 對照表 |
| 32 | 第 2 層.抽樣:temperature 調熵、top-p 截斷再正規化;情感支持系統的取樣設定是設計決策 | TempTopP+SpectrumRows(2) |
| 33 | 第 3 層.logits:constrained decoding 於合法子集再正規化(結構化輸出正解);contrastive/DoLa/CFG 操作細節 | |
| 34 | 第 4 層.聚合:best-of-n / MBR / self-consistency / reranking | |
| 35 | LLM-ASR:噪聲通道即 log 空間線性組合;n-best rescoring 屬第 4 層、speech encoder 屬第 1 層;LM weight 是手調版係數 $w$ | 公式 |
| 36 | Demo:asr-noisy-channel(2 min) | DemoFrame |

## ④ 權重層的介入(32 min,p37–51;含 demo 3 min)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 37 | 節標:改動權重 | section |
| 38 | SFT:新資料上重做 MLE,仍在 mode-covering 端 | SpectrumRows(3) |
| 39 | RLHF 目標、閉式解 $\pi^*\propto\pi_{\text{ref}}\exp(r/\beta)$;整體等價最小化 $\mathrm{KL}(\pi\|\pi^*)$,即 reverse KL | 推導 |
| 40 | DPO:閉式解代入偏好損失,得隱式 reward $=\beta\log(\pi_\theta/\pi_{\text{ref}})$(Rafailov et al., 2023);免 reward model | 推導 |
| 41 | reward model 的介面身分:$p_{\text{data}}$.logprob 的代理,補上需求表中唯一不可得的一格 | 表 |
| 42 | β 的後果:安全與多樣受同一 $\beta$ 支配(Kirk et al., 2024);情感支持系統錨點 | |
| 43 | 逐點計分器的結構極限:無法表達「分布太窄」;抑制塌縮的只有 β 項,其約束對象是與參考模型的距離;LLM-as-judge 同此限 | 示意 |
| 44 | DDO 前備(直接陳述兩件已立事實):$\sigma(\log p_{\text{data}}/p_{\text{ref}})=p_{\text{data}}/(p_{\text{data}}+p_{\text{ref}})$ 是最優判別器;統一式表末列為 $\log p_{\text{ref}}+(1/\beta)(\log p_{\text{data}}-\log p_{\text{ref}})$ | |
| 45 | DDO 構造:$d_\theta=\sigma(\beta\log(p_\theta/p_{\text{ref}}))$;β 的動機($\log p_\theta$ 可達 $10^3$ 量級,直入 sigmoid 梯度消失);BCE 最優 $p_\theta=p_{\text{data}}$(Zheng et al., 2025) | 推導 |
| 46 | 圖 B-3:DDO 機制 + 三個「無需」(無判別器網路/無交替訓練/不對抽樣反傳) | DdoMechanism |
| 47 | 梯度形式:$(1-d_\theta)(p_\theta-p_{\text{data}})\nabla\log p_\theta$;抬升與壓低;MLE 無移除多餘質量的機制 | 公式對照 |
| 48 | Demo:mle-vs-ddo-gradient(3 min) | DemoFrame |
| 49 | 光譜定位:DDO 同時具 forward KL 的抬升與 reverse KL 的壓低,作用於兩端 | SpectrumRows(3) |
| 50 | DPO/DDO 對照表;最優解 $p_\theta^*\propto p_{\text{ref}}^{1-1/\beta}p_{\text{data}}^{1/\beta}$ 即表末列;guidance 於推論期銳化、DDO 內化入權重 | 表+公式 |
| 51 | 兩則定性觀察(Zheng et al., 2025):MLE 續訓無改善反而劣化(forward KL 目標已達上限);top-k/top-p 實為降低有效溫度、掩蓋分布缺陷(具體數字在第二堂 DPM 段) | |

## ⑤ 總結與作業(10 min,p52–54)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 52 | 光譜三列完整;主張回收(含糊迴避出自 forward KL 訓練,同質化出自 reverse KL 對齊;三列各是一個可移動的位置) | SpectrumRows(全) |
| 53 | 本堂的假設與開放問題:假設 $\pi_{\text{ref}}$ 兩介面皆備;開放問題(1)各方法以什麼代理補 reverse KL 缺的 logprob(2)完全不提供 logprob 的模型如何訓練 | |
| 54 | 作業:研究題目的機率形式($x,y,c$)、光譜位置、所呼叫的介面 | |

可捨頁(超時丟棄順序):p19 → p6 → p34。

---

# lecture-02.md 「建構滿足介面的分布」(120 min,65 頁)

2026-08-16 依六份 NotebookLM 家族筆記擴充:新增 NF 與 EBM 段落與三個新 demo。

## ①② 開場問題與代理(23 min,p1–5;含 demo 於③)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 1 | 封面 | |
| 2 | 開場問題:一個不提供 logprob 的模型要如何訓練?設定:以神經網路構造提供 `sample()`/`logprob(x)` 的物件;每種構造方式即一個家族,介面的代價劃分家族 | ContractCard(compact) |
| 3 | 代理表:三散度 × 所需介面 × 缺什麼 × 代理 × 形成的家族;JSD 兩個 logprob 皆不可得,判別器是必然 | 表 |
| 4 | reward 代理與 discriminator 代理同源:都代替 $p_{\text{data}}$.logprob;皆逐點,介面不載分布層級資訊 | 對照 |
| 5 | DDO 的位置:屬 forward KL 列(logprob 齊備)卻借 JSD 列的判別器構造,因此能同時作用於光譜兩端 | |

## ③ 介面能力矩陣(15 min,p6–11;含 demo 2 min)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 6 | 節標:分類即介面能力 | section |
| 7 | 家族介面能力矩陣:AR / NF / VAE / DPM-FM / GAN × logprob × sample × 訓練散度 | FamilyMatrix |
| 8 | GAN 空格解釋三件事:只能經判別器代理訓練;logprob 為前提的方法(guidance、DPO、DDO)不適用;放棄正規化密度換得不受可逆性或序列分解約束的 generator 架構,一步生成仍能維持品質 | |
| 9 | AR 與 NF 同為 exact logprob,差在 sample 形式(逐維序列 vs 一步可逆;NF 證明密度與一步可共存,代價是可逆性約束);DPM 把一步生成拆成多個迴歸子問題,代價付在步數 | |
| 10 | 家族樹(exact / bound / multi-step;JSD;reverse KL) | FamilyTree |
| 11 | Demo:interface-contract(2 min;guide 滑桿只有 GAN 不動,對應矩陣「無」格;VAE 卡顯示 ELBO 與真值兩數) | DemoFrame |

## ④ 生成學習三難(12 min,p12–13)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 12 | 節標:三難 | section |
| 13 | 三難三角(Xiao, Kreis & Vahdat, 2022):GAN/VAE/DPM 定位;邊注:品質–多樣邊即光譜、速度頂點即矩陣 sample 欄、各家族改進史多為逼近第三頂點的嘗試 | Trilemma |
| 14 | 休息(第 50 分) | |

## ⑤ 逐家族(72 min;③④壓縮與休息吸收後的配置:AR 15 / NF 10 / VAE 8 /
GAN 11 / EBM 10 / DPM 16,節標頁附矩陣定位小圖)

家族順序:AR → NF(與 AR 同為 exact logprob,MAF/IAF 銜接)→ VAE → GAN →
EBM(negative phase 銜接判別器與 DDO;score = −∇E 橋接下一段)→ DPM/FM。
NF 段:介面實作(變數變換、三角 Jacobian)/ demo / MAF vs IAF 成本對調 /
連續化(Neural ODE、det→trace、FFJORD)/ 失效(OOD 悖論、拓撲與維度限制)與應用。
EBM 段:介面實作(未正規化 logprob、log Z 不可得)/ demo / 訓練(正負相位、
CD、score matching、NCE)/ 失效(抽樣成本、mode mixing、發散、評估)/ 改進史
(Hopfield→JEM)與今日角色。內容依據:`docs/notebooklm-briefs/`(ebm、nf、
family-crosscheck;交叉查核修正 VQ-VAE 動機、DDIM 條目等)。

### AR / LLM(20 min,p15–25;含 demo 3 min)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 15 | 節標 AR + 定位小圖 | FamilyMatrix/Trilemma(focus) |
| 16 | 介面實作:chain rule;logprob 精確且便宜,sample 序列化 | 公式 |
| 17 | CE 與 KL:$H(p,q)=H(p)+\mathrm{KL}(p\|q)$;one-hot 時等價;label smoothing / distillation 使兩者分離 | 推導 |
| 18 | $H(p)$ 不可估;四條實務路線(差值/參考模型正規化/已知熵合成資料/繞開 likelihood:MAUVE、MMD、下游指標) | 表 |
| 19 | BPB:$\frac{T}{N_{\text{bytes}}}\log_2\mathrm{PPL}_{\text{token}}$ | 公式+例 |
| 20 | Forward KL 鏈式分解,下標 $x_{<t}\sim p$:teacher forcing 是此分解的直接實作 | 公式解剖 |
| 21 | Exposure bias:推論期 prefix 來自 $q$,訓練目標從未度量之;reverse KL 的分解須同時對調前綴分布與每步引數:$\mathrm{KL}(q\|p)=\sum_t\mathbb{E}_{x_{<t}\sim q}[\mathrm{KL}(q_t\|p_t)]$,訓練從未涉足這條軌道;DDO 的壓低項補此缺口;memory agent 長對話漂移機制之一 | ExposureBias(B-2) |
| 22 | Demo:exposure-bias-track(3 min) | DemoFrame |
| 23 | False premise 背景:forward KL 訓練無拒答的結構選項(Kalai et al., 2025);$p(y\mid x)$ 良定義與 $x$ 值得回答是兩回事(Kim et al., (QA)²) | |
| 24 | 改進史:n-gram→RNN/LSTM→attention→Transformer(平行訓練啟動 scaling)→scaling law→instruction tuning/RLHF(往 mode-seeking 移動);每步標三難頂點 | Timeline |
| 25 | 提速路線(speculative decoding、multi-token prediction)與應用(對話、程式、agent、LLM-ASR 的 $p(\text{text})$ 項) | |

### VAE(10 min,p26–30;含 demo 3 min)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 26 | 節標 VAE + 定位小圖 | focus 小圖 |
| 27 | 介面實作:latent $z$;logprob 只有下界(ELBO);sample 一次前傳 | 結構圖 |
| 28 | 特徵性失效:過度平滑。Gaussian likelihood(等價 MSE)× mode-covering;latent 邊際化記號與 ICL 隱式貝氏同語言 | |
| 29 | Demo:vae-2d-interactive(3 min;觀察序:mode covering 的過度平滑→環狀拓撲→β 兩種壞法) | DemoFrame |
| 30 | 改進史+應用:β-VAE(Higgins 2017)→VQ-VAE(van den Oord 2017)→VQ-GAN(Esser 2021)→diffusion 壓縮器(Stable Diffusion);表徵學習、異常偵測、latent 基礎設施 | Timeline |

### GAN(13 min,p31–37;含 demo 5 min)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 31 | 節標 GAN + 定位小圖 | focus 小圖 |
| 32 | 介面實作:僅 sample,一步映射 $G(z)$;訓練經判別器代理;logprob 為前提的方法皆不適用 | 結構圖 |
| 33 | 損失對照表:minimax(JSD;最優判別器處梯度消失)vs non-saturating($\mathrm{KL}(p_g\|p_{\text{data}})-2\,\mathrm{JSD}$,Arjovsky & Bottou, 2017;reverse KL 項帶來 mode seeking、負 JSD 項帶來不穩定);逐項一句讀法,分解推導入講稿 | 表 |
| 34 | Mode collapse 第一層成因:generator 損失無資料項、$D$ 逐點;覆蓋率是分布層級性質,判別器介面沒有承載它的欄位;與 reward model 同構(其餘層次與 Metz 2017、Salimans 2016 入講稿) | |
| 35 | Demo:gan-2d-interactive(5 min;被漏眾數在判別器地景上值高,generator 收不到) | DemoFrame |
| 36 | 判別器想法的單向轉移:$d^*=p/(p+q)$ 是「以自身 logprob 造判別器」的原型;DDO 把該角色放進 likelihood ratio,前提是 logprob 存在,故不適用於 GAN;轉移方向單一 | |
| 37 | 改進史+應用:DCGAN→cGAN→WGAN(Earth Mover 距離對付 JSD 飽和)→StyleGAN→BigGAN→蒸餾目標(多步壓一步);低延遲/即時、超解析、風格與語音轉換、diffusion 加速 | Timeline |

### DPM / Flow Matching(17 min,p38–45;含 demo 3 min)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 38 | 節標 DPM/FM + 定位小圖 | focus 小圖 |
| 39 | 介面實作:forward KL 沿噪聲尺度分解(對照 AR 沿序列分解);每步簡單迴歸;logprob 有下界、經 probability flow ODE 精確(Song et al., 2021);sample 多步 | 對照圖 |
| 40 | 特徵性失效:抽樣慢;品質與多樣的優勢以步數支付 | |
| 41 | 改進史(上):DDPM(Ho et al., 2020)→DDIM(Song et al., 2020,確定性 ODE 減步)→Score SDE(Song et al., 2021,連續時間統一) | Timeline |
| 42 | CFG(Ho & Salimans, 2022):統一式在本家族的原始形式;其中一類 zero-shot 編輯可寫成統一式(InstructPix2Pix 的雙 guidance scale,Brooks et al., 2023):$p_A$ 條件於原圖、$p_B$ 無條件、$w$ 控制幅度 | 公式+圖 |
| 43 | 改進史(下):Latent Diffusion(Rombach et al., 2022)→Flow Matching / Rectified Flow(Lipman et al., 2023,simulation-free、源分布不必 Gaussian)→Consistency / 蒸餾(Song et al., 2023,常借對抗形式) | Timeline |
| 44 | DDO 實驗對照(Zheng et al., 2025;具體家族與數字,查證原文後填入):免 guidance 提升品質、推論成本不變;改權重而非減步數的另一條提速路線 | 數據表 |
| 45 | Demo:flow-matching-2d-interactive(3 min);應用:影像/影音/音訊、分子設計、動作生成 | DemoFrame |

## ⑥ 定位與收尾(10 min,p46–51)

| 頁 | 內容 | 圖/demo |
|---|---|---|
| 46 | 實驗室題目定位表(完整三座標:光譜/介面/三難;第六題以自足語言定位:logprob 介面讀數的可信度,校準) | 表 |
| 47 | 結語:現行方法多在第 1 層、只用 sample 介面;第 2–4 層只需 logprob,實驗室所用模型皆提供,不需額外訓練資源 | LayerStack |
| 48 | 主張(最後實質頁):含糊迴避出自 forward KL 訓練,同質化出自 reverse KL 對齊;同一光譜,三個可移動的位置,六個家族各有座標 | SpectrumRows(全) |
| 49 | 文獻(1/2) | |
| 50 | 文獻(2/2)+ 自學資源 | |
| 51 | 結尾頁 | |

可捨頁:p25 的提速段 → p19 → p10。

---

# 元件清單

SVG(typeScale;禁 opacity 屬性):DivergenceFit、KlZeros、JsdSaturate、TempTopP、
Trilemma、FamilyTree、DdoMechanism、ExposureBias
HTML+CSS(rem):ContractCard、SpectrumRows(rows/mark props)、LayerStack、
FamilyMatrix(focus prop)、Timeline(props 重用)、GuidanceForm
工具:DemoFrame.vue;global-bottom.vue(角落契約徽章)

# Demo 對應(含分鐘與大綱建議清單的映射)

| demo | 頁 | min | 對應大綱/demo-plans |
|---|---|---|---|
| divergence-2d-interactive | L1 p13 | 3 | B-1 圖的互動形式 |
| guidance-playground | L1 p24 | 3 | plans/01;涵蓋大綱建議 demo #1(token browser)的兩個論證目標(寫死 logits;真模型版屬 Tier 1,另案) |
| asr-noisy-channel | L1 p36 | 2 | plans/06 |
| mle-vs-ddo-gradient | L1 p48 | 3 | plans/02 |
| interface-contract | L2 p11 | 2 | plans/03;S1 ① 擺位取消(demo 含四張模型卡,違反第一堂不指涉具體模型的紀律),僅留矩陣頁之後回放 |
| exposure-bias-track | L2 AR 段 | 3 | plans/05 |
| ar-2d-interactive | L2 AR 段 | 3 | plans/11(新) |
| normalizing-flow-2d-interactive | L2 NF 段 | 3 | plans/12(新) |
| vae-2d-interactive | L2 VAE 段 | 3 | 大綱 demo 表 |
| gan-2d-interactive | L2 GAN 段 | 5 | 大綱 demo 表 |
| ebm-2d-interactive | L2 EBM 段 | 3 | plans/13(新) |
| flow-matching-2d-interactive | L2 DPM 段 | 3 | 大綱 demo 表 |

大綱建議 demo #2(calibration)、#3(semantic entropy)隨量測模組移出本課(編輯註:
若實驗室把信心與正確率線列為優先,另辦一小時量測工作坊)。

# 驗收

`npm run check` 與 `npm run build` 全綠;`scripts/measure-slides.mjs` 兩份 deck 無溢出
(量測含 grid 裁切偵測,KaTeX 伸縮符號除外);grep 驗收:「一把尺」「那條軸」「骨架」
歸零、「不是…而是…」每份 ≤3、可見文字無「你」。
Round 2 審查裁決:`docs/deck-review-round2.md`。
