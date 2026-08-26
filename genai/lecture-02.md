---
theme: default
title: 生成模型導論（二）建構滿足介面的分布
titleTemplate: '%s'
transition: fade
lineNumbers: false
drawings:
  persist: false
download: true
exportFilename: genai-lecture-02
fonts:
  sans: 'Source Sans 3,Noto Sans TC'
  serif: 'Source Serif 4,Noto Serif TC'
  mono: 'IBM Plex Mono'
  weights: '400,500,600,700'
class: text-left
---

<div class="rule-accent" />

# 生成模型導論（二）

## 建構滿足介面的分布

<!--
聽眾同第一堂：剛進實驗室的碩一新生，修過深度學習導論，還沒有自己的研究題目。

第一堂把分布當成抽象物件，只假設它提供兩個介面，
在這個假設上談了散度的選擇、統一引導式與權重層的介入。
今天換對象：這兩個介面到底怎麼被造出來。
每一個模型家族就是一種造法；介面以什麼形式提供、用什麼代價換得，
是家族之間全部的差異。
-->

---

# 從操作分布到建構分布

上一堂的每一個結論，都以分布已經在手上為前提

| | 上一堂 | 這一堂 |
|---|---|---|
| 問題 | 兩個分布之間的距離怎麼定義 | 對模型與資料施加什麼約束 |
| | 手上已有的分布怎麼操作 | 分布怎麼解構成可訓練的部件 |
| 分布的地位 | 給定的抽象物件 | 要建構出來的 $p_\theta$ |

<div class="mt-8">
<ContractCard compact />
</div>

<div class="mt-8">
常見的生成模型有 AR(Autoregressive Model)、Flow(Normalizing Flow, NF)、VAE(Variational Autoencoder)、EBM(Energy-Based Model)、DPM(Diffusion Probabilistic Model)、GAN(Generative Adversarial Network) 這六種，它們分別以不同代價構造分布。
</div>

<!--
[約 3 分鐘] 上一堂做了兩件事。
一是定義兩個分布之間的距離，也就是散度怎麼選、選了之後承擔哪一種錯誤。
二是分布固定之後怎麼操作它，統一引導式、推論期的四層介入、DPO 與 DDO 都屬於這一類。
兩件事共用同一個前提：分布已經在手上，兩個介面齊備。

這一堂換題目：對模型與資料施加什麼約束、把分布解構成什麼部件，
才能擬合真實分布，建構出 p_θ。
驗收標準不變，造出來的物件要能提供 sample 與 logprob。
AR、Flow、VAE、EBM、DPM、GAN 各是一種構造方式，
差別在兩個介面各以什麼形式提供、代價是什麼。
-->

---

# 三個散度，三種補法

每一個缺口，對應一種工程回應

| 散度 | 需要的介面 | 代理 | 形成的家族 |
|---|---|---|---|
| forward KL<br><span class="fine">Kullback–Leibler</span> | $p_{\text{data}}$.sample + $p_\theta$.logprob | 不需要 | MLE(Maximum Likelihood Estimation)：AR / Flow / VAE / DPM / EBM |
| reverse KL | $p_\theta$.sample + $p_\theta$.logprob + $p_{\text{data}}$.logprob | reward / energy | RLHF(Reinforcement Learning from Human Feedback)、VI(Variational Inference) |
| JSD<br><span class="fine">Jensen–Shannon divergence</span> | 兩側 logprob | 訓練一個分類器 | GAN |

<div class="mt-5">

<!--GAN 用判別器並非設計偏好：資料側的 logprob 本來就缺，而 GAN 的 generator 又不維護自身密度，JSD 需要的兩個 logprob 就一個都不剩；分類器是唯一能把 JSD 變成可計算目標的代理（第一堂的判別器讀法：最優分類器的損失值即 JSD 的仿射函數）。-->

</div>

<!--
[約 4 分鐘] 這張表是第一堂介面需求表的右半邊：
每一個「缺」都對應一種工程回應，每種回應長成一個家族。

forward KL 不缺任何介面，不需要代理，MLE 一族全在這一列：AR、Flow、VAE、DPM、EBM。
第一列雖然不缺，仍然分成好幾個家族，因為 logprob 還有「以什麼形式提供」的次級差異，
介面矩陣就是按這個差異展開的。

reverse KL 缺資料側的 logprob，代理是 reward 或 energy，對應 RLHF 與變分推論。
VI 一句話：用一個可計算的 q 去最小化 reverse KL，ELBO 就是它的實例，VAE 的 encoder 正是。
energy 代理的意思是用未正規化的能量函數充當 log 密度，差一個常數 log Z。

JSD 兩側的 logprob 都要。資料側本來就缺，而 GAN 的 generator 又不維護自己的密度，
兩個 logprob 一個都不剩，只能訓練一個分類器來代理。
所以 GAN 用判別器不是設計偏好，是唯一能把 JSD 變成可計算目標的辦法；
理論根據是第一堂的判別器讀法：最優分類器的損失值是 JSD 的仿射函數。
-->

---

# 兩種代理，同一個結構

reward model 與判別器並列

| | reward model | GAN 判別器 |
|---|---|---|
| 代理對象 | $p_{\text{data}}$.logprob（偏好版） | $\log(p_{\text{data}}/p_g)$ |
| 輸入輸出 | 單一樣本 → 純量 | 單一樣本 → 純量 |
| 極限 | 逐點評分，無法表達「分布太窄」 | 同左 |

<div class="mt-5">

兩者都代替拿不到的 $p_{\text{data}}$.logprob，也繼承同一個結構極限：介面是逐點的，分布層級的資訊（覆蓋率、多樣性）沒有欄位可以通過。第一堂裡「β 是唯一抑制塌縮的項」那個論證，對判別器一族同樣成立。

</div>

<!--
把兩個代理並排。
reward model 代理的是偏好版的 p_data.logprob；GAN 判別器代理的是 log(p_data/p_g)。
兩者的輸入輸出形式完全一樣：單一樣本進，一個純量出。

所以它們繼承同一個結構極限：介面是逐點的，
分布層級的資訊，覆蓋率、多樣性，沒有欄位可以通過。
第一堂講 reward model 時的論證，對判別器一族原封不動成立，
GAN 的 mode collapse 由同一個極限驅動。
-->

---

# DDO 借用了哪一格

兩個 logprob 都在手上，卻仍造了一個判別器

<div class="mt-2">

DDO(Direct Discriminative Optimization) 屬於 forward KL 列，本來不需要任何代理。

它做的事是**把 JSD 列的判別器構造搬進來**：不另訓分類器，直接宣告 $\sigma(\beta\log(p_\theta/p_{\text{ref}}))$ 是判別器。

</div>

<div class="mt-4">

| | 抬升資料區 | 壓低過剩區 |
|---|---|---|
| 來源 | forward KL 列的 MLE 項 | JSD 列的判別式負例項 |

</div>

<div class="mt-4">

同時作用於覆蓋程度兩端的能力，正來自這次跨列的借用：介面齊備的家族，可以動用其他列的構造。

</div>

<!--
DDO 屬於 forward KL 那一列，兩個 logprob 都在手上，本來不需要任何代理，
但它還是造了一個判別器。

它做的是跨列借用：不另訓分類器，直接宣告 σ(β log(p_θ/p_ref)) 是判別器。
抬升資料區那一項來自第一列的 MLE，壓低過剩區那一項來自 JSD 列的判別式負例。
兩端同時施力的能力就是這樣來的。

反過來不成立。GAN 借不到 logprob 家族的方法，
因為要借的東西本身就是它缺的那個介面。
-->

---
layout: section
---

# 分類即介面能力

六個家族，兩個介面，一張矩陣

<!--
這一節把六個家族排進一張矩陣。
分類的依據全部是可檢驗的介面性質：兩個介面各以什麼形式提供、代價多少。
-->

---

# 家族介面能力矩陣

同一列讀水平取捨，同一欄讀家族差異

<FamilyMatrix />

<!--
[約 3 分鐘] 這張矩陣是今天的主幹，各家族的節標頁都會出現它的高亮版。

橫著讀一列，看的是同一個介面在不同家族的形式與代價；
直著讀一欄，看的是一個家族兩個介面的搭配。

logprob 欄有四種形式：精確、下界、未正規化、沒有。
sample 欄有三種：逐維序列、一步、多步。
兩欄的組合就是家族的定義，不需要任何風格標籤。
-->

---

# GAN 的空格解釋三件事

一個空格，三個後果

<FamilyMatrix focus="GAN" compact />

<div class="mt-4">

1. **訓練只能經過判別器代理**：任何散度都需要密度資訊，GAN 一側拿不出來。
2. **上一堂以 logprob 為前提的方法全數不適用**：統一引導式、DPO(Direct Preference Optimization)、DDO，對 GAN 一條都寫不出來。
3. **放棄正規化密度，換得架構自由**：generator 不受可逆性、序列分解或任何密度簿記的約束，任意一步映射都合法，品質因此能在單步內達成。

</div>

<!--
GAN 在 logprob 欄是空的，這一格解釋三件事。

第一，訓練只能經過判別器代理。任何散度都需要密度資訊，而 GAN 一側拿不出來。
第二，第一堂以 logprob 為前提的方法全數不適用：
統一引導式、DPO、DDO，對 GAN 一條都寫不出來。
第三，放棄正規化密度換來架構自由：
generator 不受可逆性、序列分解或任何密度簿記的約束，任意一步映射都合法。

第三點要說準確：「快」來自 one-step 映射本身；
「一步之內品質高」來自架構不受密度約束。
兩件事要分開，因為 NF 就是反例：密度與一步可以共存，只是表達力要付代價。
-->

---

# 同為 exact，差在 sample；同為分解，差在方向

三種 sample 形式的對照

<div class="mt-2">

| | AR | NF | DPM |
|---|---|---|---|
| logprob | chain rule 逐項相加 | 變數變換公式 | 變分下界；經 probability flow ODE 精確 |
| sample | 逐維序列（慢、表達力強） | 一步（快，但受可逆性約束） | 多步迭代（步數即成本） |
| 分解方式 | 沿維度 | 沿可逆層 | 沿雜訊尺度 |

</div>

<div class="mt-4">

AR 與 NF 的 logprob 同樣精確，形式不同；NF 證明了密度與一步生成可以共存，代價是每一層都必須可逆且 Jacobian 可算，表達力被此束縛。DPM 把一步生成拆成許多個子問題，每個子問題是一次簡單迴歸，品質與密度都保住，代價全數記在抽樣步數上。

</div>

<!--
兩組對照。

AR 與 NF 的 logprob 同樣精確，形式不同：AR 用 chain rule 逐項相加，
NF 用變數變換公式。sample 就分開了：AR 逐維序列，慢但表達力強；
NF 一步，快但每一層都要可逆、Jacobian 要算得動，表達力被這兩個條件束縛。
NF 證明了精確密度與一步生成可以共存。

DPM 是第三欄：把一步生成拆成許多個子問題，每個子問題是一次簡單迴歸，
logprob 拿到變分下界，經 probability flow ODE 還能精確算，品質也保住，
代價全數記在抽樣步數上。

最後一列是三者的分解方式：AR 切在維度之間，NF 切在可逆層之間，
DPM 切在訊噪比之間，同一個 forward KL 的三種切法。

所以「快」與「有密度」是兩個獨立的維度，
真正的取捨在表達力、架構約束與步數之間。
-->

---

# 家族樹

依訓練散度與 logprob 形式分層

<FamilyTree />

<div class="mt-2 text-sm tone-muted text-center">

分支的每一層都是介面問題：訓練散度可不可計算、logprob 以什麼形式提供、sample 要幾步。

</div>

<!--
同一批家族換一種畫法，依訓練用的散度與 logprob 的形式分層。

分支的每一層都是介面問題：訓練散度可不可計算、
logprob 以什麼形式提供、sample 要幾步。

與傳統「explicit density / implicit density」分類的差別在於：
這棵樹的每個節點都是一個可以執行的測試。
拿到一個沒見過的模型，查兩個介面，就能定位它在哪一支，
以及第一堂的哪些方法對它適用。
-->

---
layout: none
---

<DemoFrame src="interface-contract.html" title="介面契約：逐家族呼叫兩個介面" :maxH="500" />

<!--
[2 分鐘] 展示順序：

1. 四張模型卡各按一次 sample()，都正常出點。
2. 按 GAN 的 logprob(x)：跳出 NotImplementedError。矩陣裡的那個空格就是這個錯誤。
3. 拉 guide(w) 滑桿：有 logprob 的三張卡即時銳化，GAN 的滑桿是灰的，
   對應第一堂統一引導式的適用範圍。
4. VAE 卡顯示兩個數字：ELBO 與真實 log p，前者永遠不大於後者，這就是下界的意思。
5. demo 沒有 DPM 卡，口頭補一句：DPM 的行為與 VAE 卡同型，也是下界，
   差別在於可以經 probability flow ODE 精確化。
-->

---
layout: section
---

# 生成學習三難

sample quality・mode coverage・sampling speed

<!--
這一節換一組目標來看：品質、覆蓋、速度三者，沒有家族兼得。
每個家族落在哪一條邊上，決定它的長處與代價。
-->

---

# 沒有家族三者兼得

六個家族，三條邊

<Trilemma/>

<div class="mt-2">

三個目標同時滿足，目前沒有任何家族做到([Xiao, Kreis & Vahdat, 2022](https://arxiv.org/abs/2112.07804))。每個家族都落在某一條邊上：兩端是它拿到的目標，對面的頂點是它付出的代價，沿邊的位置是兩個目標之間的偏重。

</div>

<!--
[約 6 分鐘] 三個目標同時滿足，目前沒有任何家族做到(Xiao, Kreis & Vahdat, 2022)。
每個家族都落在某一條邊上：兩端是它拿到的目標，對面的頂點是它付出的代價，
沿邊的位置是兩個目標之間的偏重。

品質與覆蓋那條邊，就是第一堂覆蓋程度的另一種畫法；速度頂點對應矩陣的 sample 欄。

逐邊點名，各配一個提問：
GAN 快又銳利，為什麼到不了覆蓋角？損失裡沒有資料項，漏掉眾數不受懲罰。
VAE 與 NF 快、覆蓋也好，為什麼到不了品質角？
VAE 是條件均值造成的模糊，NF 是可逆性限制了表達力。
AR、EBM、DPM 品質與覆蓋兼得，為什麼慢？
序列分解、MCMC、多步迭代各自就是它們的本體，不是實作不佳。

同一條邊上還可以問次序：AR 與 EBM 誰更靠覆蓋角？
兩者都以 MLE 訓練，但 EBM 的樣本品質另外受抽樣不穩拖累。

如果有人用一步蒸餾反駁「三者不能兼得」：
蒸餾的上限來自教師，總成本要含教師的訓練與多步推理，並沒有跳出三難。

改進史大半是在不放掉已有兩角的前提下，盡量逼近第三角。
-->

---
layout: center
class: text-center
---

# 中場 Q & A

<!--
休息十分鐘。

時間配置：這十分鐘由第三、四節吸收（第三節 15 分鐘壓到 12，第四節 12 分鐘壓到 8，
節餘轉入第五節），總長維持兩小時。
-->

---
layout: section
---

# AR

逐維分解，精確 logprob

<div class="mt-4">
<FamilyMatrix focus="AR" compact />
</div>

<div class="mt-3">
<Trilemma focus="AR" compact />
</div>

<!--
AR 的定位：logprob 精確而且便宜，sample 是逐維序列。
在三難上落在品質與覆蓋那一側，速度用序列抽樣支付，與 DPM、EBM 同一條邊。
-->

---

# AR 的介面實作

一條恆等式，分解出每一項條件機率

$$\log p(x)=\sum_{t}\log p(x_t\mid x_{<t})$$

<div class="mt-4">

chain rule 把聯合分布拆成逐 token 條件機率的連乘；每一項都是一次 softmax 輸出，可以直接讀取。

| 介面 | 形式 | 代價 |
|---|---|---|
| `logprob(x)` | 精確，一次前向即得全部 | 幾乎免費 |
| `sample()` | 逐 token，天生序列 | 長度即延遲 |

</div>

<!--
chain rule 把聯合分布拆成逐 token 的條件機率連乘。
這條式子是恆等式，不是近似，所以 AR 的 logprob 精確。

兩個介面的代價完全不對稱。
logprob：一次前向就拿到全部項，幾乎免費，這是所有家族裡最便宜的。
sample：逐 token 產生，第 t 個 token 的分布依賴前 t−1 個的取值，長度就是延遲。

兩件事其實是同一個分解的兩面：讓 logprob 便宜的那個結構，同時讓抽樣變成序列。
-->

---
layout: none
---

<DemoFrame src="ar-2d-interactive.html" title="AR：逐維生成，精確 logprob" :maxH="500" />

<!--
[3 分鐘] 展示腳本：

1. 按「生成一點（分步）」兩次：先抽第一維，邊際直方圖高亮；停頓後抽第二維，
   條件曲線會隨第一維的取值而不同。序列性肉眼可見。
2. 點畫面上兩個位置查 logprob：兩項相加就是總 log 密度，
   密度區讀數高、空白區讀數低。
3. 切換維度順序：同一點的兩個分項變了，總和不變。
   分解方式不唯一，密度唯一。
-->

---

# Cross-entropy(CE) 與 KL 的關係

兩者只差一項資料本身的熵

$$H(p,q)=H(p)+\mathrm{KL}(p\,\|\,q)$$

<div class="mt-3 text-sm">

| 項 | 定義 | 讀法 |
|---|---|---|
| $H(p,q)$ | $-\mathbb{E}_{x\sim p}\big[\log q(x)\big]$ | 拿 $p$ 的樣本餵給模型 $q$，取 $-\log q(x)$ 的平均，就是訓練時最小化的 loss |
| $H(p)$ | $-\mathbb{E}_{x\sim p}\big[\log p(x)\big]$ | 資料本身的熵，只由 $p$ 決定，換哪個模型都不變 |

</div>

<div class="mt-4 text-center">

語言建模最小化 CE，等價於最小化 forward KL：AR 家族落在 mode-covering 端的原因

</div>

<!--
兩個量先分清楚。
H(p,q) 是交叉熵：從 p 抽樣本，讀模型 q 給的 −log q(x)，取平均。
這就是訓練時真正在最小化的那個 loss，模型換了它就會變。
H(p) 是資料自身的熵：同樣的平均，但讀的是 p 自己的 −log p(x)。
它只由資料分布決定，與模型無關，訓練再久也壓不下去。

這條恆等式一行就能證：展開 CE 的定義，加減 E_p[log p]。
差值就是 forward KL，所以 CE 的下限是 H(p)，不是零。

三個推論。
分類任務的目標 p 是 one-hot，H(p) 等於零，所以 CE 與 forward KL 是同一個數。
目標一旦變軟，label smoothing 或 distillation，兩者就分離，H(p) 大於零成為 CE 的下限。
語言建模最小化 CE，等價於最小化 forward KL，這是 AR 家族落在 mode-covering 端的原因。

實務上的用處：loss 還很高不一定代表模型差，
H(p) 那一項是資料本身的熵，誰也壓不掉。
-->

---

# $H(p_{\text{data}})$ 估不出來

資料的熵沒有無偏估計途徑

資料只有樣本、沒有密度。CE 的絕對值因此**無法**回答「離最優還有多遠」。

<div class="mt-3">

四條實務路線：

| 路線 | 作法 |
|---|---|
| 同資料比差值 | $H(p)$ 是共同常數，模型間的 CE 差就是 KL 差 |
| 參考模型正規化 | 以另一個模型的 logprob 為基準（DDO 的 log ratio 即此形） |
| 已知熵的合成資料 | 人造分布，$H(p)$ 可解析計算 |
| 繞開 likelihood | MAUVE、MMD(Maximum Mean Discrepancy)、下游任務指標 |

</div>

<!--
資料只有樣本、沒有密度，這件事在第一堂出現在 reverse KL 與 reward model，
在這裡出現在熵：H(p_data) 沒有無偏的估計途徑，
所以 CE 的絕對值無法回答「離最優還有多遠」。

四條實務路線。
同一份資料比差值：H(p) 是共同常數，模型之間的 CE 差就是 KL 差。這條最常用，
leaderboard 上比的從來是相對值。
以參考模型正規化：拿另一個模型的 logprob 當基準，DDO 的 log ratio 就是這個形狀。
用已知熵的合成資料：人造分布，H(p) 可以解析算出來。
繞開 likelihood：MAUVE、MMD、下游任務指標。
-->

---

# 跨 tokenizer 的比較：BPB

換一個與詞表無關的單位

同一份文本，不同 tokenizer 切出的 token 數不同，per-token 的 CE 之間不可比。換算到位元組，以 BPB(Bits Per Byte) 為單位：

$$\mathrm{BPB}=\frac{T}{N_{\text{bytes}}}\cdot\log_2 \mathrm{PPL}_{\text{token}}$$

<div class="mt-4">

$T$ 為 token 數、$N_{\text{bytes}}$ 為位元組數、$\mathrm{PPL}$ 為困惑度(Perplexity)：把「每 token 的困惑度」換算成「每位元組的位元數」，單位與 tokenizer 無關。

</div>

<div class="mt-4 tone-muted">

比較不同詞表的模型、或同模型換 tokenizer 前後，以 BPB 為準。

</div>

<!--
同一份文本，不同 tokenizer 切出的 token 數不同，所以 per-token 的 CE 之間不可比。
換算到位元組就沒有這個問題。

推導很短：總 log loss 以 2 為底是 T 乘 log₂PPL，除以位元組數，
得到每位元組幾個位元。單位與 tokenizer 無關。

資訊論的讀法：這就是模型當壓縮器時的碼長。
比較不同詞表的模型，或同一個模型換 tokenizer 前後，都以 BPB 為準。
-->

---

# 訓練目標的精確形狀

forward KL 沿序列分解（對兩邊同時用 chain rule）

$$\mathrm{KL}(p\,\|\,q)=\sum_t\;\mathbb{E}_{x_{<t}\sim {\color{#2563eb}p}}\Big[\mathrm{KL}\big(p(\cdot\mid x_{<t})\,\big\|\,q(\cdot\mid x_{<t})\big)\Big]$$

<div class="mt-4">

注意期望的下標：**前綴取自 $p$**，也就是資料。

teacher forcing（訓練時餵真實前綴）是這個分解的直接實作，而非工程上的權宜。

</div>

<!--
forward KL 沿序列展開之後長這樣，重點在期望的下標：前綴取自 p，也就是資料。

推導三步，追問時再展開：
一，對 p 和 q 同時用 chain rule，log(p(x)/q(x)) 等於各步 log(p_t/q_t) 的和。
二，KL 是對 x 取期望，把和拿到期望外面。
三，對第 t 項，把第 t 步以後的部分先積掉，剩下的是對前綴取期望，而前綴取自 p，
內層正是逐步的 KL。

所以 teacher forcing，訓練時餵真實前綴，是這個分解的直接實作，不是工程上的權宜。
而這個下標，正是 exposure bias 的根源。
-->

---

# Exposure bias：目標從未度量的那條軌跡

訓練與生成走在不同的前綴上

<ExposureBias />

<div class="mt-2 text-sm">

reverse KL 的分解要把**前綴分布與每步引數同時**對調：
$\mathrm{KL}(q\,\|\,p)=\sum_t\mathbb{E}_{x_{<t}\sim q}\big[\mathrm{KL}(q(\cdot\mid x_{<t})\,\|\,p(\cdot\mid x_{<t}))\big]$。
訓練目標裡沒有任何一項在 $q$ 的前綴上取期望；模型自己生成的軌跡，損失函數從未在那裡取過期望。memory agent 長對話品質漂移的機制之一即在此。DDO 的壓低項恰好在 $p_{\text{ref}}$ 的樣本（即 $q$ 系的軌跡）上施力，補的正是這一塊。

</div>

<!--
訓練與生成走在不同的前綴上。

reverse KL 的分解要把前綴分布與每步的引數同時對調，前綴才會取自 q。
換句話說，訓練目標裡沒有任何一項在模型自己的前綴上取期望；
模型自己生成的軌跡，損失函數從來沒有在那裡量過。

具體後果：長對話裡每一輪回覆都以模型自己過去的輸出為前綴，
誤差沿著 t 累積，而訓練從未在這種前綴上校正過條件分布。
memory agent 這類題目的長對話品質漂移，機制之一就在這裡。
量級上，behavior cloning 的分析給出誤差隨生成長度平方級放大，也就是 O(εH²)。

DDO 的壓低項恰好在 p_ref 的樣本上施力，那正是 q 系的軌跡，補的就是這一塊。
-->

---
layout: none
---

<DemoFrame src="ar-2d-interactive-2.html" title="訓練軌跡與自由生成軌跡的分離" :maxH="500" />

<!--
[3 分鐘] 這一頁的模型是現場訓練的，用 TensorFlow.js：
一個小 MLP 讀最近 k 個點，迴歸下一步的位移；
訓練批次只從真實軌跡上取樣，也就是 teacher forcing。
「訓練」核取方塊預設關閉，示範時先勾起來。

畫面怎麼看：
左圖同時畫兩條軌道，藍色箭頭是每個真實點上的一步預測，前綴取自資料；
橘紅線是模型自己餵自己的 rollout，前綴取自模型。
中間面板是離流形距離，隨 rollout 步數上升，是誤差複利的直接證據。
第三個面板把兩者畫在同一單位下：一步 RMSE 一路下降，rollout 平均偏離卻卡在高處。
兩條線的落差就是 exposure bias，因為訓練損失從來沒有在橘紅那條軌道上取過期望。

示範順序：
① 圓形加 k=1，訓到 step 約 2000，rollout 貼合，先建立正常樣態。
② 切 8 字形加 k=1：交叉點上單看位置有兩個可能方向，條件分布是雙峰，
   而 MSE 只輸出平均，所以藍色箭頭在交叉點直指中間、rollout 在那裡出軌。
   k 拉到 2 等於讓模型知道速度，當場就好。
   這一段是資訊不足，還不是 exposure bias，和換什麼架構無關，要講清楚。
③ 切三圈一岔：岔口的位置與速度都相同，該續圈還是出岔取決於已經繞了第幾圈，
   依賴長度大約 25 到 56 步。這個病不反映在離流形距離上，因為每一步都貼著某條圓，
   要看左上角的圈數計數：k 小於等於 8 蓋不住一圈，圈數失控；
   k=64 蓋過兩圈半，圈數穩定回到 3。加脈絡有用，而且只在依賴長度落進窗內時才生效，
   這就是長對話漂移的 2D 版本。
④ 回到 8 字加 k=2，收斂後取消勾選「訓練」凍結模型，σ 拉到 0.06：
   推論每一步加微擾，誤差複利讓 rollout 明顯散開。
   再勾回訓練、ε 拉到 0.5（σ 不變）重訓 2000 步，重新凍結量測：
   同樣的雜訊下 rollout 穩住了，而一步 RMSE 幾乎不變。
   ε 就是 scheduled sampling，訓練中的自我前綴步也注入同一個 σ，
   訓練分布因此涵蓋推論分布。

收束：修復的條件是「訓練時見過的偏差涵蓋推論時的偏差」，
不在於有沒有用自己的樣本。閃電擾動鈕是同一件事的單次版本，
把 rollout 推離流形一次，看它回不回得來。
補丁能緩解累積速度，結構原因不動，期望的下標仍然在 p；
而且補丁自身有代價：scheduled sampling 假設逐步展開，
與 Transformer 的全前綴平行訓練相衝，實作要 two-pass decoding。

讀數每次重訓、每個起點都會跳動，現場請固定起點，比較同一段的量級，不要唸單次數值。
-->

---

# False premise：結構裡沒有拒答

條件分布不知道自己的條件有多罕見

$p(y\mid x)$ 對**任何** $x$ 都良定義，包括 $p(x)\approx 0$ 的荒謬前提。

<div class="mt-4">

- forward KL 訓練獎勵「在資料上覆蓋」，語料裡的問題幾乎都伴隨回答；拒答作為輸出模式，結構上沒有位置，除非 post-training 明文補上（[Kalai et al., 2025](https://arxiv.org/abs/2509.04664) 從訓練目標與評測誘因分析幻覺的必然性）
- 「$p(y\mid x)$ 算得出來」與「$x$ 值得回答」是兩個獨立命題；false premise 偵測要判定的是後者（基準：(QA)²，Question Answering with Questionable Assumptions，[Kim et al., 2023](https://arxiv.org/abs/2212.10003)）

</div>

<!--
p(y|x) 對任何 x 都良定義，包括 p(x) 幾乎為零的荒謬前提。
模型不會因為條件罕見就拒絕輸出，結構上沒有這個開關。

兩點。
forward KL 訓練獎勵的是「在資料上覆蓋」，而語料裡的問題幾乎都伴隨回答，
所以拒答作為一種輸出模式，在結構上沒有位置，除非 post-training 明文補上。
Kalai et al. (2025) 從訓練目標與評測誘因分析幻覺：
預訓練目標加上只獎勵正確率的評測，共同使「有把握地亂答」成為最優策略。

第二，「p(y|x) 算得出來」與「x 值得回答」是兩個獨立的命題。
false premise 偵測要判定的是後者，等於在條件分布之外另建一個對 x 本身的判斷，
模型主幹不自帶這個判斷。基準可看 (QA)²(Kim et al., 2023)。
-->

---

# AR 的改進史

四步在品質與覆蓋，最後一步移動覆蓋程度

<Timeline :items="[
  { name: 'n-gram', note: '計數即條件機率；脈絡長度受限於統計強度' },
  { name: 'RNN / LSTM', note: 'recurrent neural network 與 long short-term memory：參數化條件分布，脈絡不再截斷', tag: '品質' },
  { name: 'attention → Transformer', year: '2017', note: '訓練可平行，scaling 自此可行', tag: '品質', url: 'https://arxiv.org/abs/1706.03762' },
  { name: 'scaling laws', year: '2020', note: '損失隨規模冪律下降，投資有可預測回報(Kaplan et al.)', tag: '品質・覆蓋', url: 'https://arxiv.org/abs/2001.08361' },
  { name: 'instruction tuning / RLHF', year: '2022', note: '往 mode-seeking 端移動：可用性換多樣性', tag: '覆蓋程度右移', url: 'https://arxiv.org/abs/2203.02155' },
]" />

<!--
五步，用三難的語言讀。

n-gram：計數就是條件機率，脈絡長度受限於統計強度。
RNN 與 LSTM：把條件分布參數化，脈絡不再截斷。
attention 與 Transformer(2017)：訓練可以平行，scaling 自此可行。
scaling laws(2020)：損失隨規模冪律下降，投資有可預測的回報。
instruction tuning 與 RLHF(2022)：往 mode-seeking 端移動，用多樣性換可用性。

前四步都在品質與覆蓋兩角上推進，速度角原地不動；
最後一步不是三難上的突破，是覆蓋程度的移動，也就是第一堂那張圖上的右移。
-->

---

# 速度側與應用

序列抽樣的兩種提速

**序列生成的提速**（sample 介面的補強）：

- speculative decoding：小模型先猜、大模型驗收，可證明不改變輸出分布
- multi-token prediction：一次前向押注多個 token

<div class="mt-5">

**應用**：對話與 agent、程式生成，以及 LLM-ASR(Large Language Model + Automatic Speech Recognition) 裡作為預訓練 backbone 提高辨識能力。

</div>

<!--
速度角是 AR 的短邊，兩種提速方式都在補 sample 介面。

speculative decoding：小模型先猜，大模型驗收。
驗收步驟是精確的 rejection sampling，可以證明輸出分布不變，只是期望步數下降。
multi-token prediction：一次前向押注多個 token。

應用：對話與 agent、程式生成，
以及 LLM-ASR 裡外掛語言模型的角色：n-best rescoring 時在 log 空間
加上 λ·log p_LM，對應第一堂第 4 層的重排序。
-->

---
layout: section
---

# Normalizing Flow

可逆變換，一步抽樣

<div class="mt-4">
<FamilyMatrix focus="NF" compact />
</div>

<div class="mt-3">
<Trilemma focus="NF" compact />
</div>

<!--
NF 的定位：logprob 精確，sample 一步完成，速度角與精確密度同時到手。
代價在表達力：每一層都必須可逆，品質角是它的短邊。
-->

---

# NF 的介面實作

以可逆變換把標準高斯映射到資料分布，變數變換公式給出精確密度

$$\log p_x(x)=\log p_z\big(f^{-1}(x)\big)+\log\big|\det J_{f^{-1}}(x)\big|$$

<div class="mt-3">

- 行列式項是體積修正：空間被拉伸處密度變稀、壓縮處變密，總機率守恆
- `logprob(x)`：逆向映射得 $z$、查 base 密度、累加各層 log-det，精確值三步到手
- `sample()`：$z\sim\mathcal N(0,I)$，前向一步

</div>

<div class="mt-3 aside aside-data text-sm tone-muted">

任意 $D\times D$ 行列式要 $O(D^3)$；NF 以架構設計讓 Jacobian 呈三角形，行列式退化為對角線連乘，降為 $O(D)$。離散 NF 的架構史，大半是三角結構的設計史。

</div>

<!--
用一個可逆變換把標準高斯映射到資料分布，變數變換公式給出精確密度。

行列式那一項是體積修正：空間被拉伸的地方密度變稀、被壓縮的地方變密，總機率守恆。
logprob 三步到手：逆向映射得到 z、查 base 密度、累加各層的 log-det，而且是精確值。
sample 更簡單：從標準高斯抽 z，前向一步。

兩個硬條件：bijection，同維度、可逆、可微；以及 Jacobian 要算得動。
任意 D×D 行列式是 O(D³)，所以 NF 用架構設計讓 Jacobian 呈三角形，
行列式退化成對角線連乘，降到 O(D)。離散 NF 的架構史，大半是三角結構的設計史：
NICE，non-linear independent components estimation(Dinh et al., 2014)，用 additive coupling，det 恆為 1；
RealNVP，real-valued non-volume preserving(Dinh et al., 2016)，用 affine coupling，log|det| 是各 s 之和；
Glow(Kingma & Dhariwal, 2018)加上可學的 invertible 1x1 conv，以 LU 參數化。
coupling 每層只動一半維度，層與層之間用 permutation 混合。
-->

---
layout: none
---

<DemoFrame src="nf-2d-interactive.html" title="Normalizing Flow：可逆變換與精確密度" :maxH="500" />

<!--
[3 分鐘] 這一頁的 flow 也是現場訓練的：一疊 affine coupling，RealNVP 型，
直接最小化精確 NLL，也就是 forward KL 或 MLE，沒有對抗、沒有下界。
損失以 bits/dim 讀出，與 AR 的 BPB 同單位，可以互相比較。按「開始訓練」後即可操作。

展示腳本：
1. 拉「層 k」滑桿，或讓它自動播放：高斯網格被前 k 層逐層折成資料分布，
   全程不撕裂、不重疊，這就是可逆性的視覺形式。k 等於 L 的粒子就是生成樣本，一步完成。
2. 中圖熱圖是逐格做 inverse pass 算出來的精確 log p(x)，
   既不是判別器分數，也不是 ELBO，是全系列 demo 裡唯一的精確密度圖。
   點擊任一位置就是呼叫 logprob 介面，讀出數值，
   並在左圖畫出這個 x 逐層反推回 z 的路徑。
3. 層數從 2 拉到 8：單層 coupling 表達力很弱，表達力是用深度買的，層數 2 蓋不住雙月。
4. 拓撲的代價：選「高斯混合（中心偏移）」，各群之間必然留下細絲；
   選「圓環」，中心會漏出一道無法歸零的密度。
   連續可逆變換是同胚，不能剪開也不能戳洞，這是結構限制，不是訓練不足。
   同一題各家的表現可以對照：VAE 搭橋而糊、GAN 漏群而塌縮、
   FM 多步搬運而慢、NF 留下細絲。
5. 這個 k 滑桿與 FM demo 的 t 滑桿是同一件事的兩種粒度：
   NF 是離散層的堆疊，FM 是連續時間的場，
   機率流 ODE 就是層數趨於無窮的連續極限。
-->

---

# 介面成本的權衡

自迴歸把成本推到一側，coupling 兩側同時便宜

| | MAF<br><span class="fine">masked autoregressive flow</span> | IAF<br><span class="fine">inverse autoregressive flow</span> | Coupling（NICE / RealNVP / Glow） |
|---|---|---|---|
| 依賴結構 | 逐維，條件於 $x_{1:i-1}$ | 逐維，條件於 $u_{1:i-1}$ | 二分區塊，一半條件另一半 |
| logprob | 一次並行評估（快） | 逐維還原 $u$，$O(D)$ 串行（慢） | 一次並行評估（快） |
| sample | 逐維串行 $O(D)$（慢） | 一次抽 $u$，並行生成（快） | 一次並行生成（快） |
| 缺點 | 生成慢，長度即延遲 | 外部樣本 logprob 慢，難以 MLE 訓練 | 每層只動一半維度，靠深度補表達力 |
| 適用 | 密度估計、MLE 訓練 | 即時生成、變分後驗 | 通用密度估計與影像生成 |

<div class="mt-4 text-sm">

<!-- 同一條 chain rule，依賴擺在資料側或雜訊側，兩個介面的成本對調。自迴歸型要兩側都快得靠蒸餾：Parallel WaveNet 的 IAF 型 student 對自家樣本算 logprob 是並行的，因為 $u$ 已知([van den Oord et al., 2018](https://arxiv.org/abs/1711.10433))。 -->

</div>

<!--
同一條 chain rule，依賴擺在資料側或雜訊側，兩個介面的成本表整個對調。

MAF(Papamakarios et al., 2017)：依賴條件於 x_{1:i-1}，
logprob 一次並行算完，快；sample 逐維串行，O(D)，慢。適合密度估計與 MLE 訓練。
IAF(Kingma et al., 2016)：依賴條件於 u_{1:i-1}，
sample 一次抽 u 就並行生成，快；logprob 要逐維還原 u，慢。適合即時生成與變分後驗。

第三欄要一起講，否則會誤以為兩側必須二選一。
NICE、RealNVP、Glow 這類 coupling 流不是逐維自迴歸，而是二分區塊：
前一半原封不動送出去，後一半的 scale 與 shift 只讀前一半算出來。
正向反向都是同一組 s、t，兩個方向都有封閉解，logprob 與 sample 同樣是一次並行前向。
代價在表達力：每層只變換一半維度，得靠深度與 permutation，Glow 的 1x1 conv 就是在做混合。
把 coupling 看成只有兩個 block 的分解，就知道它與 MAF、IAF 是同一條 chain rule 的不同切法，
Jacobian 一樣是三角形。

自迴歸型要兩側都快，得靠蒸餾。Parallel WaveNet：MAF 型的 teacher 高效訓練，
蒸餾給 IAF 型的 student。關鍵是 student 對自己生成的樣本算 logprob 是並行的，
因為 u 已經知道了，於是兩側的快同時到手(van den Oord et al., 2018)。
繞過兩難的原因是：logprob 只需要對自家樣本便宜。
-->

---

# 連續化：行列式退化為跡

層數趨於無窮的極限是一條常微分方程

$$\frac{dz}{dt}=f(z,t),\qquad \frac{d\log p(z(t))}{dt}=-\mathrm{Tr}\!\left(\frac{\partial f}{\partial z}\right)$$

<div class="mt-3">

- Neural ODE(Ordinary Differential Equation)([Chen et al., 2018](https://arxiv.org/abs/1806.07366))：非線性的行列式在連續極限下換成線性的跡：$O(D^3)\to O(D)$，架構約束隨之解除
- 可逆性免費取得：向量場 Lipschitz 連續則軌跡不相交(Picard–Lindelöf)，不再需要 coupling 結構
- FFJORD(Free-form Jacobian of Reversible Dynamics)：Hutchinson 估計跡，免建 Jacobian([Grathwohl et al., 2018](https://arxiv.org/abs/1810.01367))
- 代價：MLE 訓練的每一步都要呼叫數值 ODE solver，慢且不穩

</div>

<!--
層數趨於無窮的極限是一條 ODE(Neural ODE,Chen et al., 2018)。

四點。
非線性的行列式在連續極限下換成線性的跡，O(D³) 降到 O(D)，架構約束隨之解除；
關鍵在於跡是線性算子，對疊加封閉，所以任意網路都可以當向量場。
可逆性免費取得：向量場 Lipschitz 連續則軌跡不相交，這是 Picard–Lindelöf，
不再需要 coupling 結構。
FFJORD(Grathwohl et al., 2018)用 Hutchinson 估計跡，連 Jacobian 都不必建。
代價：MLE 訓練的每一步都要呼叫數值 ODE solver，慢而且不穩；
adjoint 法可以讓顯存與深度無關。

要注意連續化之後，「一步生成」變成「積分一條 ODE」，步數重新成為成本，
與離散 NF 的一步抽樣是不同的取捨。
-->

---

# NF 的特徵性失效與今日角色

分布外樣本的悖論、結構限制，與今日的應用領域

**OOD(Out-Of-Distribution) 悖論**：在複雜影像上以 MLE 訓練的 NF，常給沒見過的更簡單影像更高的 log-likelihood([Nalisnick et al., 2019](https://arxiv.org/abs/1810.09136))：forward KL 只要求資料處機率高，不禁止別處更高；低複雜度影像經可逆映射聚在高斯原點附近的高密度區。精確似然與可靠的 OOD 偵測是兩件事。

<div class="mt-3">

**結構限制**：可逆映射不能改變拓撲；維度必須全程保持，無法下採樣壓縮；單層表達力弱，迫使極深堆疊。

</div>

<div class="mt-3">

**今日角色**：語音 vocoder（WaveGlow 並行生成）、科學計算（Boltzmann Generators 一步抽分子平衡態）、變分後驗(IAF)、模擬推斷與宇宙學。

</div>

<!--
OOD 悖論：在複雜影像上以 MLE 訓練的 NF，常給沒見過的更簡單影像更高的 log-likelihood
(Nalisnick et al., 2019)。原因不神秘：forward KL 只要求資料處機率高，
不禁止別處更高；低複雜度影像經可逆映射之後聚在高斯原點附近的高密度區。
第三個機制是 coupling 的 mask 偏向擬合局部像素相關性，
學到的更像通用壓縮器。結論：精確似然與可靠的 OOD 偵測是兩件事。

結構限制三點：可逆映射不能改變拓撲；維度必須全程保持，無法下採樣壓縮；
單層表達力弱，迫使極深堆疊。
影像上失守的原因也是三個：像素空間的 MLE 把容量花在不可見的高頻；
確定性軌跡在高斯薄殼上沒有糾錯能力；維度不變擋住了 latent 壓縮路線。

今日角色：語音 vocoder（WaveGlow 並行生成）、
科學計算（Boltzmann Generators 一步抽分子平衡態）、
變分後驗(IAF)、模擬推斷與宇宙學。
-->

---
layout: section
---

# VAE

latent 變數，下界訓練

<div class="mt-4">
<FamilyMatrix focus="VAE" compact />
</div>

<div class="mt-3">
<Trilemma focus="VAE" compact />
</div>

<!--
VAE 的定位：引入 latent 變數，logprob 只到下界，sample 是 decoder 一次前向。
在三難上快而且覆蓋好，短邊在品質。
-->

---

# VAE 的介面實作

以下界取代不可解的邊際似然

引入 latent $z$：$p_\theta(x)=\int p_\theta(x\mid z)\,p(z)\,dz$，積分不可解，改而最大化證據下界(Evidence Lower Bound, ELBO)：

$$\log p_\theta(x)\;\ge\;\mathrm{ELBO}=\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]-\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)$$

| 介面 | 形式 |
|---|---|
| `logprob(x)` | 僅下界(ELBO)；與真值的差是 $\mathrm{KL}(q_\phi\,\|\,p_\theta(z\mid x))$ |
| `sample()` | $z\sim p(z)$，decoder 一次前向 |

<!--
引入 latent z 之後，邊際似然是一個對 z 的積分，不可解，所以改成最大化下界。

ELBO 的推導就是 Jensen 不等式套在 log 積分上，課前自檢第 4 題練過。
兩個介面因此長這樣：logprob 只有下界，與真值的差是 KL(q_φ ‖ p_θ(z|x))；
sample 是從先驗抽 z，decoder 一次前向。

那個 gap 可以再拆兩層，追問時用：
approximation gap 是對角高斯變分族本身的表達力上限，最佳化到完美也不會歸零；
amortization gap 是共享的 encoder 無法對每一筆資料輸出各自的最優後驗。

順帶一提，對潛在變數積分這個記號，與第一堂 in-context learning 那條積分是同一種語言。
-->

---

# 特徵性失效：過度平滑

Gaussian 重建項與 mode-covering 的疊加

重建項用 Gaussian likelihood 時，$\log p_\theta(x\mid z)$ 等價於負的均方誤差(Mean Squared Error, MSE)；
MSE 的最優解是**條件均值**：多個可能輸出的平均，而非其中任何一個。

<div class="mt-4">

疊上 mode-covering 的訓練目標，結果是樣本模糊、細節被抹平：兩個機制往同一個方向推。

</div>

<div class="mt-4 tone-muted">

這是失效型態由訓練目標決定的又一例：模糊寫在 Gaussian × forward KL 這個組合裡，調架構調不掉。

</div>

<!--
兩個機制往同一個方向推。

重建項用 Gaussian likelihood 時，log p(x|z) 等價於負 MSE，
而 MSE 的最優解是條件均值：多個可能輸出的平均，不是其中任何一個。
一維類比：資料一半是 +1、一半是 −1，MSE 的最優預測是 0，誰也不像。
影像上同理，紋理的多種可能被平均成糊。

再疊上 mode-covering 的訓練目標，結果就是樣本模糊、細節被抹平。

這是失效型態由訓練目標決定的又一例：
模糊寫在 Gaussian 乘上 forward KL 這個組合裡，調架構調不掉。
-->

---
layout: none
---

<DemoFrame src="vae-2d-interactive.html" title="VAE：過度平滑、拓撲、β 的兩端" :maxH="500" />

<!--
[3 分鐘] 觀察順序：

1. mode covering 造成的過度平滑：樣本雲糊在兩峰之間，對照前一頁的機制。
2. 環狀資料：高斯先驗蓋不乾淨環形拓撲，雲的中心破洞被填掉。
3. β 過大：重建垮掉，樣本全糊；β 過小：latent 空洞化，從先驗抽樣落空。
   兩種壞法對應 ELBO 兩項的失衡。

補充兩個正式名稱。
posterior collapse：encoder 追不上移動中的真後驗，強 decoder 乾脆忽略 z，
之後 encoder 再也拿不到梯度；修法有 cyclical KL annealing 與 free bits。
prior holes：aggregate posterior 與標準高斯不重合，先驗抽樣落進空洞就解碼成糊；
環狀資料那個破洞正是此現象，修法之一是在 latent 空間再訓一個先驗模型。
-->

---

# VAE 的改進史與今日角色

三步對付模糊，最後轉為基礎設施

<Timeline dense :items="[
  { name: 'β-VAE', year: '2017', note: '調 KL 項係數，換 disentanglement(Higgins et al.)', tag: '控制', url: 'https://openreview.net/forum?id=Sy2fzU9gl' },
  { name: 'VQ-VAE', year: '2017', note: 'vector quantized VAE，離散 latent：decoder 無法忽略，繞開 posterior collapse(van den Oord et al.)', tag: '結構', url: 'https://arxiv.org/abs/1711.00937' },
  { name: 'VQ-GAN', year: '2021', note: 'vector quantized GAN：加對抗損失，重建變銳利(Esser et al.)', tag: '品質', url: 'https://arxiv.org/abs/2012.09841' },
  { name: '壓縮器角色', year: '今日', note: 'Stable Diffusion 在 VAE latent 空間跑 diffusion', tag: '基礎設施', url: 'https://arxiv.org/abs/2112.10752' },
]" />

<div class="mt-3">

**應用**：表徵學習、異常偵測（ELBO 作為異常分數），以及作為其他生成模型的 latent 基礎設施。

</div>

<!--
四步，方向一致，都在對付「糊」。

β-VAE(2017)：調 KL 項的係數，換 disentanglement。
VQ-VAE(2017)：把 latent 離散化，decoder 無法忽略，繞開 posterior collapse。
VQ-GAN(2021)：加上對抗損失，重建變銳利。
今日：轉為基礎設施，Stable Diffusion 在 VAE 的 latent 空間跑 diffusion。

最終定位不在正面戰場，而是把「快、覆蓋好、可微下界」三個長處打包，
當別的家族的壓縮層。
應用另有表徵學習與異常偵測，ELBO 直接當異常分數。
-->

---
layout: section
---

# GAN

一步映射，沒有密度

<div class="mt-4">
<FamilyMatrix focus="GAN" compact />
</div>

<div class="mt-3">
<Trilemma focus="GAN" compact />
</div>

<!--
GAN 的定位：一步映射，只有 sample，沒有密度。
在三難上快又銳利，短邊在覆蓋。
-->

---

# GAN 的介面實作

只有 sample，沒有密度簿記

$$x = G(z),\qquad z\sim\mathcal N(0,I)$$

<div class="mt-4">

一步映射，只有 `sample()`。$G$ 不維護任何密度，logprob 那一欄是空的：

- 訓練只能透過判別器代理（本堂開頭的代理表）
- 統一引導式、DPO、DDO 全部以 logprob 為前提，對 GAN 無一適用
- 換得的是不受可逆性與序列分解約束的 generator 架構

</div>

<!--
generator 就是一個從高斯到資料的一步映射，G 不維護任何密度，
所以 logprob 那一欄是空的。

三個後果：訓練只能透過判別器代理；
統一引導式、DPO、DDO 全部以 logprob 為前提，對 GAN 無一適用；
換得的是不受可逆性與序列分解約束的 generator 架構。

訓練形式：判別器 D 與 generator G 交替，D 學分辨真假，G 學騙過 D。
第一堂的判別器讀法給了理論對應：在最優 D 之下，G 的目標值是 2·JSD 減 2log2。
不過實務上用的損失與這個不同。
-->

---

# 理論上的 JSD，實務上的別種東西

換掉損失，也換掉了對應的散度

$2\,\mathrm{JSD}-2\log 2$ 只在最優判別器處成立，而那裡正是梯度消失的位置（JSD 飽和）。實務改用 non-saturating 損失，對應的散度變成([Arjovsky & Bottou, 2017](https://arxiv.org/abs/1701.04862))：

$$\mathrm{KL}\big(p_g\,\|\,p_{\text{data}}\big)-2\,\mathrm{JSD}\big(p_g\,\|\,p_{\text{data}}\big)$$

| | 原始 minimax | non-saturating |
|---|---|---|
| 對應散度 | JSD | reverse KL $-\ 2\cdot$JSD |
| 病理 | 梯度消失 | mode-seeking、易塌縮 |

<div class="mt-3 text-sm tone-muted">

換損失治好了梯度消失，也一併換掉了 mode-covering：每種損失各配一種失效。

</div>

<!--
2·JSD 減 2log2 只在最優判別器處成立，而那裡正是梯度消失的位置，
也就是第一堂那條 JSD 飽和曲線。

所以實務改用 non-saturating 損失，而換掉損失就換掉了對應的散度：
Arjovsky & Bottou (2017) 的 Thm 2.5 給出，
non-saturating 目標在最優 D 下的梯度，等於 KL(p_g‖p_data) 減 2·JSD 的梯度。
注意是梯度相等，不是目標值相等。

讀這個式子：第一項是 reverse KL，mode-seeking 的來源；
第二項的負號與 JSD 反向，越分得開越受獎勵，這是震盪與不穩定的來源。

所以對照表是：原始 minimax 對應 JSD，病理是梯度消失；
non-saturating 對應 reverse KL 減 2·JSD，病理是 mode-seeking、容易塌縮。
換損失治好了梯度消失，也一併換掉了 mode-covering。每種損失各配一種失效。
-->

---

# Mode collapse 的第一層成因

覆蓋率沒有進入 generator 收到的訊號

generator 的損失裡**沒有資料項**：$G$ 只從 $D$ 的評分獲得訊號，而 $D$ 逐點評分。

<div class="mt-4">

覆蓋率是分布層級的性質；逐點介面沒有承載它的欄位。漏掉一整個眾數時，只要現有樣本能騙過 $D$，損失就不會抗議。

</div>

<div class="mt-4 aside aside-model">

與 reward model 的結構極限同構：兩個逐點代理，同一個限制。第一堂裡抑制塌縮的只剩 β；這裡連 β 都沒有。

</div>

<div class="mt-3 text-sm tone-muted">

對症的修補是給介面加欄位：minibatch discrimination 把 batch 內樣本相似度的總和直接接進判別器的輸入，塌縮批次立即被抓([Salimans et al., 2016](https://arxiv.org/abs/1606.03498))。

</div>

<!--
generator 的損失裡沒有資料項：G 只從 D 的評分獲得訊號，而 D 是逐點評分。

覆蓋率是分布層級的性質，逐點介面沒有欄位承載它。
漏掉一整個眾數的時候，只要現有樣本能騙過 D，損失就不會抗議。

這與 reward model 的結構極限同構，兩個逐點代理，同一個限制。
差別在第一堂裡至少還有 β 這一項約束與參考模型的距離，這裡連 β 都沒有。

對症的修補是給介面加欄位：
minibatch discrimination 把 batch 內樣本相似度的總和直接接進判別器的輸入，
塌縮的批次立即被抓(Salimans et al., 2016)。

更深層的成因，交替最佳化的動力學、D 的容量、G 的參數化，各有文獻，課堂只展開第一層。
動力學的一句話圖像：G 過擬合當前的 D、塌到單一安全眾數，
D 追上之後 G 跳到下一個眾數，循環不收斂；
Unrolled GANs(Metz et al., 2017)讓 G 對 k 步之後的 D 最佳化，消掉這個循環。
-->

---
layout: none
---

<DemoFrame src="gan-2d-interactive.html" title="Mode collapse：判別器地景看得見，generator 收不到" :maxH="500" />

<!--
[5 分鐘] 訓練到塌縮，然後疊上判別器地景。

被漏掉的眾數在地景上是高值區，表示 D 明明知道那裡是真資料；
但 G 的梯度只來自它自己樣本所在的位置，那片高值區的資訊傳不過去。

畫面上沒有任何訊號告訴 G「少了一個眾數」。這就是逐點介面的限制，
不是最佳化不夠久，也不是學習率沒調好。
-->

---

# 判別器想法的單向轉移

借用只在一個方向上成立

第一堂 DDO 的原型即是這裡的 $d^*=p/(p+q)$：「用自身 logprob 造判別器」把 GAN 的核心構造搬進了 logprob 家族。

<div class="mt-4">

| 方向 | 可行性 |
|---|---|
| 判別器構造 → logprob 家族(DDO) | 可行：log ratio 直接可算，無需另訓網路 |
| logprob 方法 → GAN | 不可行：$p_\theta$ 寫不出來，式子無從成立 |

</div>

<div class="mt-4">

介面多的一方可以借介面少的一方的構造，反向沒有著力點。

</div>

<!--
第一堂 DDO 的原型就是這裡的 d* = p/(p+q)：
「用自身的 logprob 造判別器」把 GAN 的核心構造搬進了 logprob 家族。

方向是單向的。
判別器構造搬到 logprob 家族可行，因為 log ratio 直接算得出來，不必另訓網路。
反過來，logprob 方法搬到 GAN 不可行，因為 p_θ 寫不出來，式子無從成立。

一句話：介面多的一方可以借介面少的一方的構造，反向沒有著力點，
因為要借的武器本身就是那個缺掉的介面。
-->

---

# GAN 的改進史與今日角色

穩定、品質、規模，最後轉向蒸餾

<Timeline dense :items="[
  { name: 'DCGAN', year: '2015', note: 'deep convolutional GAN：穩定可複製的卷積配方(Radford et al.)', tag: '品質', url: 'https://arxiv.org/abs/1511.06434' },
  { name: 'conditional GAN', note: '條件化生成', tag: '控制', url: 'https://arxiv.org/abs/1411.1784' },
  { name: 'WGAN', year: '2017', note: 'Wasserstein GAN：Earth Mover 距離替代 JSD，支撐集分離時仍有梯度(Arjovsky et al.)', tag: '穩定', url: 'https://arxiv.org/abs/1701.07875' },
  { name: 'StyleGAN', year: '2019', note: '品質與可控性(Karras et al.)', tag: '品質', url: 'https://arxiv.org/abs/1812.04948' },
  { name: 'BigGAN', year: '2019', note: '大規模訓練(Brock et al.)', tag: '品質', url: 'https://arxiv.org/abs/1809.11096' },
  { name: '蒸餾目標', year: '近年', note: '少從頭訓練；把多步模型壓成一步的對抗式蒸餾', tag: '速度', url: 'https://arxiv.org/abs/2311.17042' },
]" />

<div class="mt-2">

**應用**：低延遲與即時場景、超解析、風格與語音轉換、diffusion 的加速蒸餾。

</div>

<!--
六步：穩定、品質、規模，最後轉向蒸餾。

DCGAN(2015)給出穩定可複製的卷積配方；conditional GAN 加上條件化；
WGAN(2017)用 Earth Mover 距離替代 JSD，支撐集分離時仍然有梯度；
StyleGAN(2019)品質與可控性；BigGAN(2019)大規模訓練；
近年少從頭訓練，主力轉為把多步模型壓成一步的對抗式蒸餾。

WGAN 對付的正是第一堂那條 JSD 飽和曲線。具體對照：
兩個平行分布相距 θ 時，KL 是無窮、JSD 飽和成常數，
而 Wasserstein 距離等於 |θ|，呈線性，梯度不消失。

今日角色的轉變可以一句話總結：一步生成的能力被保留，訓練難的部分讓給別的家族。
應用在低延遲與即時場景、超解析、風格與語音轉換，以及 diffusion 的加速蒸餾。
-->

---
layout: section
---

# Energy-Based Model

未正規化的密度，MCMC(Markov Chain Monte Carlo) 抽樣

<div class="mt-4">
<FamilyMatrix focus="EBM" compact />
</div>

<div class="mt-3">
<Trilemma focus="EBM" compact />
</div>

<!--
EBM 的定位：任意純量網路都可以當能量函數，約束最少；
代價是 logprob 只到未正規化為止，sample 要靠 MCMC。
三難上與 DPM 同在慢側，品質另受訓練不穩拖累。
-->

---

# EBM 的介面實作

能量給出相對機率，$\log Z$ 攔住絕對值

$$p(x)=\frac{e^{-E(x)}}{Z},\qquad Z=\int e^{-E(x)}\,dx$$

任意純量網路都可以當 $E$：低能量即高機率，約束最少的一種參數化。

| 介面 | 形式 |
|---|---|
| `logprob(x)` | 只到未正規化為止：$-E(x)-\log Z$，而 $\log Z$ 算不出來 |
| `sample()` | 原生沒有：靠 Langevin dynamics 逐步迭代逼近 |

<div class="mt-3 text-sm">

$Z$ 有多難：$224\times224\times3$ 的二值影像要對 $2^{150528}\approx10^{45000}$ 個狀態加總。相對比較則不受影響：兩點的 $-E$ 之差就是 log 機率比，$\log Z$ 相消。

</div>

<!--
能量越低、機率越高，分母 Z 是對整個空間的積分。
任意純量網路都可以當 E，這是約束最少的一種參數化。

兩個介面：
logprob 只到未正規化為止，是 −E(x) 減 log Z，而 log Z 算不出來。
Z 有多難？224×224×3 的二值影像要對 2 的 150528 次方個狀態加總，大約 10 的 45000 次方。
sample 原生沒有，靠 Langevin dynamics 逐步迭代逼近：
x 每步沿 −∇E 走一小段，再加一點高斯雜訊。注意 ∇ₓ log Z 等於零，
所以抽樣只需要未正規化的梯度。

相對比較完全不受影響：兩點的 −E 之差就是 log 機率比，log Z 相消。
所以矩陣裡這是一種新的 logprob 形式：比 VAE 的下界更弱，絕對值不可得；
但比 GAN 的空格強，相對比較、重排序、OOD 偵測都能做。
-->

---
layout: none
---

<DemoFrame src="ebm-2d-interactive.html" title="EBM：能量地景與 Langevin 抽樣" :maxH="500" />

<!--
[3 分鐘] 展示腳本：

1. 近距雙峰：啟動 20 條鏈，粒子落谷，兩個峰都有人口。
2. 點兩個位置查能量：高低可以比，但絕對的 logprob 差一個未知的 log Z。
3. 切遠距雙峰：同樣的 Langevin，跨峰計數器幾乎不動。
   sample 介面的成本與 mode mixing 的困難，在這裡一眼可見。
-->

---

# 沒有 log Z，怎麼訓練

MLE 梯度拆成兩個相位（$-\nabla_\theta\log Z$ 化為對模型分布的期望）

$$\nabla_\theta\log p(x)=\underbrace{-\nabla_\theta E(x)}_{\text{壓低真實資料的能量}}\;+\;\underbrace{\mathbb E_{x'\sim p_\theta}\big[\nabla_\theta E(x')\big]}_{\text{拉高自生樣本的能量}}$$

| 方法 | 手法 | 對應的差異度量 |
|---|---|---|
| Contrastive Divergence<br><span class="fine">[Hinton, 2002](https://doi.org/10.1162/089976602760128018)</span> | 鏈從資料點起跑只走 $k$ 步；$\log Z$ 梯度對消 | $\mathrm{KL}(p_{\text{data}}\|p_\theta)-\mathrm{KL}(p_k\|p_\theta)$ |
| Score matching<br><span class="fine">[Hyvärinen, 2005](https://jmlr.org/papers/v6/hyvarinen05a.html)</span> | 改對 $x$ 微分：$\nabla_x\log Z=0$，$Z$ 直接消失 | Fisher divergence |
| NCE<br><span class="fine">noise contrastive estimation・[Gutmann & Hyvärinen, 2010](https://proceedings.mlr.press/v9/gutmann10a.html)</span> | 與已知雜訊分布做二元分類，$Z$ 當可學純量 | BCE(Binary Cross Entropy)（漸近 KL） |

<!--
MLE 的梯度拆成兩個相位：壓低真實資料的能量，拉高模型自生樣本的能量。
第二項來自 −∇log Z，它可以化成對模型分布的期望，這是關鍵一步。

問題是第二項需要模型自己的樣本，所以每次參數更新都得跑 MCMC。
實務上用 persistent chain 或 replay buffer 攤銷(Du & Mordatch, 2019)。

三種繞法：
Contrastive Divergence(Hinton, 2002)：鏈從資料點起跑只走 k 步，log Z 的梯度對消，
對應的差異度量是 KL(p_data‖p_θ) 減 KL(p_k‖p_θ)。
Score matching(Hyvärinen, 2005)：改成對 x 微分，∇ₓ log Z 等於零，Z 直接消失，
對應 Fisher divergence。denoising score matching 是它的實用版：
對加噪資料做 score matching，雜訊核的 score 有解析形式，
訓練化為預測所加雜訊的監督任務。
NCE(Gutmann & Hyvärinen, 2010)：與一個已知的雜訊分布做二元分類，Z 當成可學純量，
損失是 BCE，漸近於 KL。

「拉高自生樣本的能量」與第一堂 DDO 的壓低項是同一種力，
DDO 把這個 negative phase 寫進了 likelihood ratio。
-->

---

# EBM 的特徵性失效

抽樣、混合、穩定性、評估

- **抽樣即成本**：一次生成要數百至數千步 Langevin，每步一次前向加反向；對照一次前向即生成的家族，ImageNet 規模的訓練時間由小時級膨脹至年級
- **Mode mixing**：眾數間隔著高能量障壁時，鏈困在單一能量谷，mixing time 隨障壁高度指數成長，各眾數的比例難以還原
- **訓練發散**：能量無上界約束，OOD 區域會長出訓練資料中不存在的假能量井，sampler 掉入後訓練崩壞；與 BatchNorm 尤其不相容
- **評估困難**：exact likelihood 不存在，只能以 Annealed Importance Sampling 等昂貴近似

<!--
四項。

抽樣即成本：一次生成要數百到數千步 Langevin，每步一次前向加一次反向；
對照一次前向就生成的家族，ImageNet 規模的訓練時間由小時級膨脹到年級。
Mode mixing：眾數之間隔著高能量障壁時，鏈困在單一能量谷，
mixing time 隨障壁高度指數成長，各眾數的比例難以還原。demo 的遠距雙峰就是這個。
訓練發散：能量沒有上界約束，OOD 區域會長出訓練資料中不存在的假能量井，
sampler 掉進去之後訓練就崩壞；與 BatchNorm 尤其不相容，
機制是真實資料與高噪 MCMC 樣本的批統計劇烈波動，最佳化跟著震盪。
評估困難：exact likelihood 不存在，只能用 Annealed Importance Sampling 這類昂貴近似。
-->

---

# EBM 的改進史與今日角色

從 Hopfield 到 JEM(Joint Energy-based Model)

<Timeline dense :items="[
  { name: 'Hopfield 網路・Boltzmann Machine', year: '1982–85', note: '能量地景支配網路狀態；隨機隱單元(Hopfield;Ackley, Hinton & Sejnowski)', tag: '概念', url: 'https://doi.org/10.1073/pnas.79.8.2554' },
  { name: 'RBM', year: '1986', note: 'restricted Boltzmann machine：二部圖限制使 Gibbs sampling 可行(Smolensky)', tag: '可訓練', url: 'https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf' },
  { name: 'Contrastive Divergence', year: '2002', note: '短鏈繞過平衡態；2006 年疊層預訓練帶動深度學習復興(Hinton)', tag: '訓練', url: 'https://doi.org/10.1162/089976602760128018' },
  { name: '深度 ConvNet 能量函數', year: '2016', note: '現代網路參數化 E(Xie et al.)', tag: '品質', url: 'https://arxiv.org/abs/1602.03264' },
  { name: 'ImageNet 規模', year: '2019', note: 'replay buffer 穩定 SGLD，stochastic gradient Langevin dynamics(Du & Mordatch)', tag: '規模', url: 'https://arxiv.org/abs/1903.08689' },
  { name: 'JEM', year: '2020', note: '任何 softmax 分類器都暗含 EBM：E(x) = −LogSumExp(logits)(Grathwohl et al.)', tag: '統一', url: 'https://arxiv.org/abs/1912.03263' },
]" />

<div class="mt-2 text-sm">

**今日角色**：score $=-\nabla_x E$，能量地景的梯度場；OOD 與異常偵測(energy score)、機器學習勢能面（分子模擬）、序列級重排序（residual EBM 對整句評分，緩解逐 token 生成的誤差累積）。

</div>

<!--
六步，從 Hopfield 到 JEM。

Hopfield 網路與 Boltzmann Machine(1982–85)：能量地景支配網路狀態，加上隨機隱單元。
RBM(1986)：二部圖限制使 Gibbs sampling 可行。
Contrastive Divergence(2002)：短鏈繞過平衡態；2006 年的疊層預訓練帶動深度學習復興。
深度 ConvNet 能量函數(2016)：用現代網路參數化 E。
ImageNet 規模(2019)：replay buffer 穩定 SGLD。
JEM(2020)：任何 softmax 分類器都暗含一個 EBM，E(x) 等於負的 LogSumExp(logits)。

JEM 的讀法值得多講一句：E(x,y) 等於 −f(x)[y]，對 y 邊際化就得到 E(x)；
既有的分類器免重訓就可以拿來做生成與 OOD 偵測。

今日角色：score 就是 −∇ₓE，能量地景的梯度場；
OOD 與異常偵測用 energy score；機器學習勢能面用在分子模擬；
序列級重排序有 residual EBM，對整句評分，緩解逐 token 生成的誤差累積。
residual EBM 的形式是 P(x) 正比於 P_LM(x) 乘 e^{−E(x)}，以 LM 當雜訊分布用 NCE 訓練，
可以看成第一堂第 4 層重排序的訓練期版本。
-->

---
layout: section
---

# DPM / Flow Matching

多步分解，每步一次迴歸

<div class="mt-4">
<FamilyMatrix focus="DPM" compact />
</div>

<div class="mt-3">
<Trilemma focus="DPM" compact />
</div>

<!--
DPM 的定位：把生成拆成多步，每一步是一次簡單迴歸；
品質與覆蓋都保住，代價全部記在步數上。
-->

---

# DPM 的介面實作

把生成拆成一疊簡單迴歸

AR 沿**序列**分解 forward KL；DPM 沿**雜訊尺度**分解同一個散度：
把資料逐步加入雜訊直到成為純雜訊，模型學每一小步的還原。

<div class="mt-3">

- 每一步是一次簡單迴歸（預測雜訊或速度場），訓練穩定性與 AR 相當
- `logprob(x)`：有變分下界；經 probability flow ODE 可精確計算([Song et al., 2021](https://arxiv.org/abs/2011.13456))
- `sample()`：多步迭代，一步一次前向

</div>

<div class="mt-4 tone-muted">

同一個 forward KL，兩種切法：AR 切在維度之間，DPM 切在訊噪比之間。

</div>

<!--
AR 沿序列分解 forward KL，DPM 沿雜訊尺度分解同一個散度：
把資料逐步加入雜訊直到成為純雜訊，模型學每一小步的還原。
加噪方向是固定的，不學；學的只有還原方向。

三點：
每一步是一次簡單迴歸，預測雜訊或速度場，訓練穩定性與 AR 相當；
logprob 有變分下界，而且經 probability flow ODE 可以精確計算(Song et al., 2021)；
sample 是多步迭代，一步一次前向。

同一個 forward KL，兩種切法：AR 切在維度之間，DPM 切在訊噪比之間。

兩個連結值得講：score 的另一個名字就是 −∇E，
denoising score matching 本來出自 EBM 的訓練工具箱，
diffusion 學的 s(x,t) 就是各雜訊尺度下能量地景的梯度場；
而 PF-ODE 的精確 logprob，用的正是 NF 段那個連續版變數變換，也就是跡。

訓練穩定的來源可以一句話說完：沒有對抗、沒有配分函數、沒有可逆性約束。
-->

---

# 特徵性失效：慢

優勢幾乎全數以抽樣步數支付

一張樣本要跑幾十到上千次前向。

<div class="mt-4">

矩陣裡 DPM 的 sample 欄與三難裡 DPM 的位置說的是同一件事；
於是這個家族的改進史，大半是一部**減步數**的歷史。

</div>

<!--
一張樣本要跑幾十到上千次前向。

矩陣裡 DPM 的 sample 欄，與三難裡 DPM 的位置，說的是同一件事。
而且「慢」不是實作不佳：多步就是這個分解方式的本體。

所以這個家族的改進史，大半是一部減步數的歷史。
-->

---

# 改進史（上）：從千步到少步

三篇論文，步數降兩個量級

<Timeline dense :items="[
  { name: 'DDPM', year: '2020', note: 'denoising diffusion probabilistic models：離散時間逐步去噪，千步量級(Ho et al.)', tag: '品質・覆蓋', url: 'https://arxiv.org/abs/2006.11239' },
  { name: 'DDIM', year: '2020', note: 'denoising diffusion implicit models：非 Markov 過程族，η=0 得確定性抽樣；已訓練的 DDPM 權重免重訓，步數大減(Song et al.)', tag: '速度', url: 'https://arxiv.org/abs/2010.02502' },
  { name: 'Score-based SDE', year: '2021', note: '離散步驟統一成連續時間 SDE(stochastic differential equation)；probability flow ODE 由此而來(Song et al.)', tag: '理論統一', url: 'https://arxiv.org/abs/2011.13456' },
]" />

<!--
三篇論文，步數降兩個量級。

DDPM(2020)：離散時間逐步去噪，千步量級。
DDIM(2020)：非 Markov 的過程族，η 等於 0 時得到確定性抽樣；
已訓練的 DDPM 權重免重訓，步數大減。
Score-based SDE(2021)：把離散步驟統一成連續時間的 SDE，probability flow ODE 由此而來。

DDIM 與 Score-SDE 幾乎同期，都在 2020 年底。概念關係是：
DDIM 的確定性抽樣正是 probability flow ODE 的離散化特例，SDE 框架把它解釋清楚。
-->

---

# CFG(Classifier-Free Guidance) 與 zero-shot 編輯

統一引導式在這個家族的原始形式([Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598))

$$\log p_w = \log p(x\mid c) + w\big(\log p(x\mid c)-\log p(x)\big)$$

<div class="mt-4">

其中一類 zero-shot 編輯直接套用這個形式：$p_A$ 條件於**原圖**、$p_B$ 為無條件分布、$w$ 控制改動幅度。InstructPix2Pix 的雙 guidance scale（影像一個係數、指令一個係數）是此式加到兩個比值項的推廣，每個條件各配一個係數([Brooks et al., 2023](https://arxiv.org/abs/2211.09800))。

</div>

<div class="mt-3 tone-muted text-sm">

不需重訓、不需配對資料：同一個係數，在影像家族裡調的是「聽指令的程度」。

</div>

<!--
CFG 是統一引導式在這個家族的原始形式(Ho & Salimans, 2022)，
第一堂表裡 CFG for LLM 那一列，原產地就在這裡。

實作方式：訓練時隨機丟棄條件，同一個網路同時學到有條件與無條件的分數；
推論時外插，每一步兩次前向，延遲加倍。

其中一類 zero-shot 編輯直接套用這個形式：p_A 條件於原圖，p_B 是無條件分布，
w 控制改動幅度，不需要重訓、也不需要配對資料。
InstructPix2Pix(Brooks et al., 2023)是同一式加到兩個比值項的推廣，
影像一個係數、指令一個係數，每個條件各配一個係數。

一個實務提醒：線性外插套在確定性 flow 上會把軌跡推離資料流形，
出現過飽和、結構崩壞；predictor-corrector 類的修法把外插改成內插。
另外要說清楚，其他編輯法，像 RePaint 的替換式 inpainting 或 SDEdit，
用的是別的機制，不要塞進這條式子。
-->

---

# 改進史（下）：換空間、換目標、換步數

latent 空間、速度場、一步蒸餾

<Timeline dense :items="[
  { name: 'Latent Diffusion', year: '2022', note: '在 VAE latent 空間跑 diffusion，算力降一個量級(Rombach et al.)', tag: '速度', url: 'https://arxiv.org/abs/2112.10752' },
  { name: 'Flow Matching / Rectified Flow', year: '2023', note: 'continuous normalizing flow(CNF) 的免模擬訓練法：在插值路徑上直接迴歸速度場；源分布不必是 Gaussian(Lipman et al.;Liu et al.)', tag: '簡化', url: 'https://arxiv.org/abs/2210.02747' },
  { name: 'Consistency Models / 對抗式蒸餾', year: '2023', note: '多步 ODE 積分蒸餾成一步(Song et al.)；對抗式蒸餾另見 ADD，adversarial diffusion distillation(Sauer et al.)', tag: '速度', url: 'https://arxiv.org/abs/2303.01469' },
]" />

<div class="mt-3 text-sm tone-muted">

Latent Diffusion 的壓縮層即 VAE 段的收尾；一步蒸餾的對抗損失即 GAN 段的今日角色：三個家族在這條時間軸上會合。

</div>

<!--
三步：換空間、換目標、換步數。

Latent Diffusion(2022)：在 VAE 的 latent 空間跑 diffusion，算力降一個量級。
Flow Matching 與 Rectified Flow(2023)：CNF 的免模擬訓練法，
在插值路徑上直接迴歸速度場；抽一對 (x₀, x₁) 就能訓，不需要模擬整條軌跡，
而且源分布不必是 Gaussian，配對與橋接類任務直接受益。
Consistency Models 與對抗式蒸餾(2023)：把多步 ODE 積分蒸餾成一步。

這條時間軸上三個家族會合：
Latent Diffusion 的壓縮層就是 VAE 段的收尾，
一步蒸餾的對抗損失就是 GAN 段的今日角色。
-->

---

# 另一條提速路線：改權重，不減步數

省下的是每步的第二次前向

對原以 CFG 推論的模型，DDO 微調後**免 guidance** 的品質超過原 CFG 基線，每步省下一次前向（[Zheng et al., 2025 專案頁](https://research.nvidia.com/labs/cosmos-lab/ddo/)）；下表為免 guidance 的前後對照：

| 模型 | 資料集 | FID(Fréchet Inception Distance)（前 → 後，無 guidance） |
|---|---|---|
| EDM | CIFAR-10 | 1.79 → 1.30 |
| EDM2 | ImageNet-64 | 1.58 → 0.97 |
| EDM2 | ImageNet 512×512 | 1.96 → 1.26 |
| VAR-d30（AR 家族） | ImageNet 256×256 | 4.74 → 1.79 |

<div class="mt-3 text-sm">

每輪微調成本低於預訓練 epoch 數的 1%，可 self-play 疊代。上一堂兩則定性觀察（MLE 到頂、截斷藏缺陷）的數字版即在此表。

</div>

<!--
前面的提速都在減步數，這一條不動步數，動的是權重。

對原本要用 CFG 推論的模型，DDO 微調之後免 guidance 的品質就超過原來的 CFG 基線，
於是每一步省下第二次前向（Zheng et al., 2025 專案頁）。
表上是免 guidance 的前後對照：EDM 在 CIFAR-10 是 1.79 到 1.30；
EDM2 在 ImageNet-64 是 1.58 到 0.97；ImageNet 512 是 1.96 到 1.26；
VAR-d30 在 ImageNet 256 是 4.74 到 1.79。
每輪微調成本低於預訓練 epoch 數的 1%，而且可以 self-play 疊代。

兩點提醒：CIFAR-10 的 EDM 基線本來就免 guidance，「省一次前向」那句話不適用該列；
表上同時有 diffusion 與 AR 家族，因為這個方法的適用邊界只由 logprob 介面決定，
與家族的其他細節無關。這正是介面語言的預測力。

第一堂那兩則定性觀察，MLE 到頂、截斷藏缺陷，數字版就是這張表。
-->

---
layout: none
---

<DemoFrame src="flow-matching-2d-interactive.html" title="Flow Matching：同一個散度的另一種分解" :maxH="500" />

<div class="px-6 pt-2 text-sm tone-muted">

應用：影像、影片與音訊生成、分子設計、動作生成。

</div>

<!--
[3 分鐘] 展示：向量場把源分布連續搬運成資料分布；
把源分布換成非 Gaussian，訓練照常進行。

軌跡的平直程度對應少步抽樣的可行性，因果鏈值得講清楚：
獨立配對的 (x₀, x₁) 會產生交叉路徑，交叉處模型平均了互相衝突的速度，
軌跡因此彎曲，積分只好用小步；
reflow 用自己上一輪的非交叉配對重訓，軌跡拉直之後一階 Euler 幾步就夠。

應用：影像、影片與音訊生成、分子設計、動作生成。
-->

---
layout: section
---

# 總結

六個家族的優缺點與適用場景

<!--
最後一節把六個家族並排收尾：各自的長處、代價，以及什麼題目該找誰。
-->

---

# 六個家族的優缺點

長處與代價都寫在兩個介面上

| 家族 | 優點 | 缺點 |
|---|---|---|
| AR | logprob 精確且便宜，訓練穩定，離散資料天生適用 | 生成逐維串行，長度即延遲；自由生成會偏離訓練分布 |
| NF | logprob 精確，一步生成，映射可逆 | 每層都要可逆且 Jacobian 可算，表達力受此束縛 |
| VAE | 一步生成，訓練穩定，latent 可供下游使用 | logprob 只有下界；條件均值造成重建模糊 |
| GAN | 一步生成，品質銳利，架構不受密度簿記約束 | 沒有 logprob；訓練不穩，易 mode collapse |
| EBM | 能量函數形式自由，多個能量可直接相加組合 | $\log Z$ 不可算，訓練與抽樣都要 MCMC |
| DPM / FM | 品質與覆蓋兼顧，訓練穩定，可用引導式控制 | 抽樣多步，速度是它的短邊 |

<!--
[約 3 分鐘] 這張表把六個家族的長處與代價並排。

AR：logprob 一次前向就拿到，訓練穩定，離散序列天生適用；代價在生成的串行與誤差累積。
NF：精確密度加一步生成，兩件好事同時到手；代價是可逆與 Jacobian 兩個硬條件壓住表達力。
VAE：一步生成、訓練穩定、latent 可用；代價是 logprob 只到下界，重建偏模糊。
GAN：一步生成、品質銳利、架構自由；代價是整欄 logprob 空白，訓練不穩。
EBM：能量形式自由、可相加組合；代價是 log Z 不可算，訓練與抽樣都得跑 MCMC。
DPM 與 FM：品質與覆蓋兼顧、訓練穩定、可引導；代價全記在步數上。

每一欄的優缺點都不是風格評語，是兩個介面的形式與代價直接推出來的。
-->

---

# 六個家族的適用場景

先看題目要哪個介面，再看能付多少步數

| 家族 | 什麼題目找它 | 代表用途 |
|---|---|---|
| AR | 離散序列，且需要密度讀數 | 語言與程式生成、n-best rescoring、偵測與校準 |
| NF | 要精確密度，又要一步生成 | 變分後驗、語音 vocoder、分子平衡態、模擬推斷 |
| VAE | 要表徵或壓縮，不追求逐點品質 | 異常偵測、擴散模型的 latent 空間基礎設施 |
| GAN | 單步就要高品質，且不需要密度 | 影像轉換與超解析、把多步模型蒸餾成一步 |
| EBM | 對既有分布做修正或整體評分 | 序列級重排序、OOD 偵測、分子勢能面 |
| DPM / FM | 連續訊號的高品質生成與可控編輯 | 影像、影片、音訊、分子設計、動作生成 |

<!--
[約 3 分鐘] 同樣六列，換成「什麼題目該找誰」。

AR：離散序列而且要讀密度，語言與程式生成、rescoring、偵測與校準都在這一列。
NF：精確密度加一步生成同時要求時才輪到它，變分後驗、vocoder、分子平衡態、模擬推斷。
VAE：目標是表徵或壓縮，逐點品質可以讓步，異常偵測與 latent 基礎設施是今日主場。
GAN：單步就要高品質、又不需要密度，影像轉換、超解析、一步蒸餾。
EBM：手上已經有一個分布，要對它修正或整體評分，序列級重排序與 OOD 偵測。
DPM 與 FM：連續訊號的生成與可控編輯，今天大部分影像影片音訊系統在這一列。

選家族的順序：先問要哪個介面、以什麼形式，再問能付多少抽樣步數。
兩個問題答完，候選通常只剩一兩個。
-->

---
layout: statement
---

# 選家族就是選介面

<div class="text-xl leading-relaxed mt-8">

先問題目要 sample 還是 logprob、要哪一種形式；再問能付多少抽樣步數。兩個問題答完，六選一就只剩一兩個候選。

</div>

<div class="mt-8 text-base tone-faint">

沒有一個家族三難全拿；每一次選擇都是拿一個目標換另一個，而換掉的是什麼，寫在它的兩個介面上。

</div>

<!--
收尾一句：選家族就是選介面。

先問題目要 sample 還是 logprob、要哪一種形式，
再問能付多少抽樣步數，六選一通常就只剩一兩個候選。

沒有一個家族三難全拿。每一次選擇都是拿一個目標換另一個，
而換掉的是什麼，就寫在它的兩個介面上。
-->

---

# 參考文獻(1/2)

散度、引導、對齊、條件化

<div class="text-xs leading-relaxed grid grid-cols-2 gap-x-6">
<div>

**散度與機率背景**
- [Stanford CS236](https://deepgenerativemodels.github.io/), Lectures 1–2
- Bishop & Bishop, [*Deep Learning: Foundations and Concepts*](https://www.bishopbook.com/)
- Endres & Schindelin (2003), [*A New Metric for Probability Distributions*](https://doi.org/10.1109/TIT.2003.813506)
- Arjovsky & Bottou (2017), [*Towards Principled Methods for Training GANs*](https://arxiv.org/abs/1701.04862)

**引導與解碼**
- Holtzman et al. (2020), [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751)
- Li et al. (2023), [*Contrastive Decoding*](https://arxiv.org/abs/2210.15097)
- Chuang et al. (2024), [*DoLa*](https://arxiv.org/abs/2309.03883)
- Sanchez et al. (2023), [*Stay on Topic with Classifier-Free Guidance*](https://arxiv.org/abs/2306.17806)
- Ho & Salimans (2022), [*Classifier-Free Diffusion Guidance*](https://arxiv.org/abs/2207.12598)
- Karras et al. (2024), [*Autoguidance*](https://arxiv.org/abs/2406.02507)

</div>
<div>

**對齊與 DDO**
- Ouyang et al. (2022), [InstructGPT](https://arxiv.org/abs/2203.02155)
- Rafailov et al. (2023), [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290)
- Kirk et al. (2024), [*Understanding the Effects of RLHF*](https://arxiv.org/abs/2310.06452)
- Zheng et al. (2025), [*Direct Discriminative Optimization*](https://arxiv.org/abs/2503.01103)(ICML)
- Chen et al. (2024), [*SPIN*](https://arxiv.org/abs/2401.01335)

**條件化與量測**
- Xie et al. (2022), [*In-context Learning as Implicit Bayesian Inference*](https://arxiv.org/abs/2111.02080)
- Liu et al. (2024), [*Lost in the Middle*](https://arxiv.org/abs/2307.03172)
- Wang et al. (2023), [*Self-Consistency*](https://arxiv.org/abs/2203.11171)
- Kalai et al. (2025), [*Why Language Models Hallucinate*](https://arxiv.org/abs/2509.04664)
- Kim et al. (2023), [*(QA)²: Question Answering with Questionable Assumptions*](https://arxiv.org/abs/2212.10003)

</div>
</div>

<!--
文獻頁不唸，留給課後查。

第一頁是兩堂課共用的理論來源：散度與機率背景、引導與解碼、對齊與 DDO、條件化與量測。
想補課的順序建議：先 Stanford CS236 前兩講，再回來看散度那幾篇。
-->

---

# 參考文獻(2/2)

分類、三難，與各家族的演進

<div class="text-xs leading-relaxed grid grid-cols-3 gap-x-5">
<div>

**分類與三難**
- Tomczak, [*Deep Generative Modeling*](https://link.springer.com/book/10.1007/978-3-031-64087-2), 2nd ed. (2024)
- Xiao, Kreis & Vahdat (2022), [*Generative Learning Trilemma*](https://arxiv.org/abs/2112.07804)

**家族演進**
- Kaplan et al. (2020), [*Scaling Laws*](https://arxiv.org/abs/2001.08361)
- Higgins et al. (2017), [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl)；van den Oord et al. (2017), [VQ-VAE](https://arxiv.org/abs/1711.00937)；Esser et al. (2021), [VQ-GAN](https://arxiv.org/abs/2012.09841)
- Radford et al. (2015), [DCGAN](https://arxiv.org/abs/1511.06434)；Arjovsky et al. (2017), [WGAN](https://arxiv.org/abs/1701.07875)；Karras et al. (2019), [StyleGAN](https://arxiv.org/abs/1812.04948)
- [Metz et al. (2017)](https://arxiv.org/abs/1611.02163);[Salimans et al. (2016)](https://arxiv.org/abs/1606.03498)

</div>
<div>

**Normalizing Flow**
- Dinh et al. (2014), [NICE](https://arxiv.org/abs/1410.8516)；Dinh et al. (2016), [RealNVP](https://arxiv.org/abs/1605.08803)；Kingma & Dhariwal (2018), [Glow](https://arxiv.org/abs/1807.03039)
- Kingma et al. (2016), [IAF](https://arxiv.org/abs/1606.04934)；Papamakarios et al. (2017), [MAF](https://arxiv.org/abs/1705.07057)；van den Oord et al. (2018), [Parallel WaveNet](https://arxiv.org/abs/1711.10433)
- Chen et al. (2018), [Neural ODE](https://arxiv.org/abs/1806.07366)；Grathwohl et al. (2018), [FFJORD](https://arxiv.org/abs/1810.01367)
- [Nalisnick et al. (2019)](https://arxiv.org/abs/1810.09136)

**Energy-Based Model**
- [Hopfield (1982)](https://doi.org/10.1073/pnas.79.8.2554);[Ackley, Hinton & Sejnowski (1985)](https://doi.org/10.1207/s15516709cog0901_7)；Smolensky (1986), [RBM](https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf)
- Hinton (2002), [CD](https://doi.org/10.1162/089976602760128018);[Hyvärinen (2005)](https://jmlr.org/papers/v6/hyvarinen05a.html)；Gutmann & Hyvärinen (2010), [NCE](https://proceedings.mlr.press/v9/gutmann10a.html)
- [Xie et al. (2016)](https://arxiv.org/abs/1602.03264);[Du & Mordatch (2019)](https://arxiv.org/abs/1903.08689)；Grathwohl et al. (2020), [JEM](https://arxiv.org/abs/1912.03263)

</div>
<div>

**Diffusion / Flow Matching**
- Ho et al. (2020), [DDPM](https://arxiv.org/abs/2006.11239)；Song et al. (2020), [DDIM](https://arxiv.org/abs/2010.02502)
- Song et al. (2021), [Score-SDE](https://arxiv.org/abs/2011.13456);[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600)
- Rombach et al. (2022), [Latent Diffusion](https://arxiv.org/abs/2112.10752)
- Lipman et al. (2023), [*Flow Matching*](https://arxiv.org/abs/2210.02747)；Liu et al. (2023), [*Rectified Flow*](https://arxiv.org/abs/2209.03003)
- Song et al. (2023), [*Consistency Models*](https://arxiv.org/abs/2303.01469)；Sauer et al. (2023), [ADD](https://arxiv.org/abs/2311.17042)
- Brooks et al. (2023), [*InstructPix2Pix*](https://arxiv.org/abs/2211.09800)

</div>
</div>

<!--
第二頁按家族排：分類與三難、AR、NF、EBM、diffusion 與 flow matching。
每個家族挑一篇代表作讀就夠，不必全掃。
-->

---
layout: end
class: text-center
---
