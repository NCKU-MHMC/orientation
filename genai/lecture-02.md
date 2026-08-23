---
theme: default
title: 生成模型導論(二)建構滿足介面的分布
titleTemplate: '%s'
transition: fade
lineNumbers: false
drawings:
  persist: false
fonts:
  sans: 'Source Sans 3,Noto Sans TC'
  serif: 'Source Serif 4,Noto Serif TC'
  mono: 'IBM Plex Mono'
  weights: '400,500,600,700'
class: text-left
---

<div class="rule-accent" />

# 生成模型導論(二)

## 建構滿足介面的分布

<div class="mt-10 tone-faint">

新進成員訓練・第二堂(120 分鐘)

</div>

<!--
第一堂建立了散度、統一引導式與權重層介入,全程把分布當抽象物件;
本堂的對象是具體構造:每個模型家族是一種「造出這個物件」的方式。
-->

---

# 一個懸而未決的問題

上一堂的每一種損失,寫出來都要呼叫 `logprob(x)`

$$\mathrm{KL}\text{ 系列}\quad \mathcal L_{\text{DPO}}\quad d_\theta=\sigma\!\big(\beta\log\tfrac{p_\theta}{p_{\text{ref}}}\big)$$

<div class="mt-4">

那麼,一個**只會生樣本、不會報密度**的模型,要拿什麼訓練?

</div>

<div class="mt-6">
<ContractCard compact />
</div>

<div class="mt-6">

本堂的任務:用神經網路構造出提供這兩個介面的物件。每一種構造方式對應一個模型家族;介面以什麼形式提供、以什麼代價換得,是家族之間全部的差異。

</div>

<!--
[約 3 分鐘] 開場問題正是第一堂作業附帶的開放問題。
AR、Flow、VAE、EBM、DPM、GAN 各是「如何造出這個物件」的一種答案。
-->

---

# 三個散度,三種補法

每一個缺口,對應一種工程回應

| 散度 | 需要的介面 | 缺什麼 | 代理 | 形成的家族 |
|---|---|---|---|---|
| forward KL | $p_{\text{data}}$.sample + $p_\theta$.logprob | 不缺 | 不需要 | MLE:AR / Flow / VAE / DPM / EBM |
| reverse KL | $p_\theta$.sample + $p_\theta$.logprob + $p_{\text{data}}$.logprob | $p_{\text{data}}$.logprob | reward / energy | RLHF、VI |
| JSD | 兩側 logprob | 資料側必缺;generator 不維護密度時模型側也缺 | 訓練一個分類器 | GAN |

<div class="mt-5">

GAN 用判別器並非設計偏好:資料側的 logprob 本來就缺,而 GAN 的 generator 又不維護自身密度,JSD 需要的兩個 logprob 就一個都不剩;分類器是唯一能把 JSD 變成可計算目標的代理(第一堂的判別器讀法:最優分類器的損失值即 JSD 的仿射函數)。

</div>

<!--
[約 4 分鐘] 這張表是第一堂介面需求表的右半:每個「缺」都對應一種工程回應,
每種回應長成一個家族。表的第一列(不缺)仍分成數個家族,
因為 logprob 介面還有「以什麼形式提供」的次級差異,介面矩陣即按此展開。
VI 一句定義備用:以一個可計算的 q 最小化 reverse KL,ELBO 即其實例,
VAE 的 encoder 正是;energy 代理:以未正規化的能量函數充當 log 密度(差一個常數 log Z)。
-->

---

# 兩種代理,同一個結構

reward model 與判別器並列

| | reward model | GAN 判別器 |
|---|---|---|
| 代理對象 | $p_{\text{data}}$.logprob(偏好版) | $\log(p_{\text{data}}/p_g)$ |
| 輸入輸出 | 單一樣本 → 純量 | 單一樣本 → 純量 |
| 極限 | 逐點評分,無法表達「分布太窄」 | 同左 |

<div class="mt-5">

兩者都代替拿不到的 $p_{\text{data}}$.logprob,也繼承同一個結構極限:介面是逐點的,分布層級的資訊(覆蓋率、多樣性)沒有欄位可以通過。第一堂裡 β 是唯一煞車的論證,對判別器一族同樣成立。

</div>

<!--
第一堂「逐點計分器的結構極限」在此獲得第二個實例;
GAN 的 mode collapse 由同一個極限驅動。
-->

---

# DDO 借用了哪一格

兩個 logprob 都在手上,卻仍造了一個判別器

<div class="mt-2">

DDO 屬於 forward KL 列,本來不需要任何代理。

它做的事是**把 JSD 列的判別器構造搬進來**:不另訓分類器,直接宣告 $\sigma(\beta\log(p_\theta/p_{\text{ref}}))$ 是判別器。

</div>

<div class="mt-4">

| | 抬升資料區 | 壓低過剩區 |
|---|---|---|
| 來源 | forward KL 列的 MLE 項 | JSD 列的判別式負例項 |

</div>

<div class="mt-4">

同時作用於光譜兩端的能力,正來自這次跨列的借用:介面齊備的家族,可以動用其他列的構造。

</div>

<!--
這一頁把第一堂 DDO 段的「為什麼它能兩端同時施力」翻譯成表格語言:
它站在第一列,借第三列的武器;反向(GAN 借 logprob 方法)沒有著力點。
-->

---
layout: section
---

# 分類即介面能力

六個家族,兩個介面,一張矩陣

---

# 家族介面能力矩陣

同一列讀水平取捨,同一欄讀家族差異

<FamilyMatrix />

<!--
[約 3 分鐘] 這張矩陣是本堂的主圖;各家族節標頁的定位小圖即其高亮版。
分類的依據全部是可檢驗的介面性質,不是風格標籤。
-->

---

# GAN 的空格解釋三件事

一個空格,三個後果

<FamilyMatrix focus="GAN" compact />

<div class="mt-4">

1. **訓練只能經過判別器代理**:任何散度都需要密度資訊,GAN 一側拿不出來。
2. **上一堂以 logprob 為前提的方法全數不適用**:統一引導式、DPO、DDO,對 GAN 一條都寫不出來。
3. **放棄正規化密度,換得架構自由**:generator 不受可逆性、序列分解或任何密度簿記的約束,任意一步映射都合法,品質因此能在單步內達成。

</div>

<!--
第 3 點的準確說法:「快」來自 one-step 映射本身;
「一步之內品質高」來自架構不受密度約束。NF 是關鍵對照:密度與一步可以共存。
-->

---

# 同為 exact,差在 sample;同為分解,差在方向

三種 sample 形式的對照

<div class="mt-2">

**AR 與 Normalizing Flow**:logprob 同樣精確,sample 形式不同。

| | AR | NF |
|---|---|---|
| logprob | chain rule 逐項相加 | 變數變換公式 |
| sample | 逐維序列(慢、表達力強) | 一步(快,但受可逆性約束) |

NF 證明了密度與一步生成可以共存;代價是每一層都必須可逆且 Jacobian 可算,表達力被此束縛。

</div>

<div class="mt-4">

**DPM** 把一步生成拆成許多個子問題,每個子問題是一次簡單迴歸:logprob 與品質都保住,代價全數記在抽樣步數上。

</div>

<!--
矩陣的三種 sample 形式(序列/一步/多步)各有代表;「快」與「密度」是獨立的兩個維度,
真正的取捨在表達力、約束與步數之間。
-->

---

# 家族樹

依訓練散度與 logprob 形式分層

<FamilyTree />

<div class="mt-2 text-sm tone-muted text-center">

分支的每一層都是介面問題:訓練散度可不可計算、logprob 以什麼形式提供、sample 要幾步。

</div>

<!--
與傳統「explicit / implicit density」分類的差別:這棵樹的每個節點都是一個可執行的測試。
拿到新模型,查兩個介面,就能定位它在哪一支,以及第一堂哪些方法適用。
-->

---
layout: none
---

<DemoFrame src="interface-contract.html" title="介面契約:逐家族呼叫兩個介面" :maxH="470" />

<!--
[2 分鐘] 展示:
1. 四張模型卡各按一次 sample(),都正常出點。
2. 按 GAN 的 logprob(x):NotImplementedError。矩陣裡的「無」就是這個錯誤。
3. 拉 guide(w) 滑桿:有 logprob 的三張卡即時銳化,GAN 的滑桿是灰的,
   對應上一堂統一引導式的適用範圍。
4. VAE 卡顯示兩個數字:ELBO 與真實 log p,前者永遠不大於後者(下界)。
5. demo 沒有 DPM 卡;口頭補一句:DPM 的行為與 VAE 卡同型(下界),
   差別在可經 probability flow ODE 精確化。
-->

---
layout: section
---

# 生成學習三難

sample quality・mode coverage・sampling speed

---

# 沒有家族三者兼得

六個家族,三條邊

<Trilemma/>

<div class="mt-2">

三個目標同時滿足,目前沒有任何家族做到(Xiao, Kreis & Vahdat, 2022)。每個家族都落在某一條邊上:兩端是它拿到的目標,對面的頂點是它付出的代價,沿邊的位置是兩個目標之間的偏重。

品質與覆蓋那條邊,是上一堂光譜的另一種畫法;速度頂點對應矩陣的 sample 欄。

</div>

<!--
[約 6 分鐘] 改進史大半是在不放掉已有兩角的前提下逼近第三角。逐邊點名,各配一個提問:
GAN 快且銳利——為什麼到不了覆蓋角?(損失無資料項,漏眾數不罰)
VAE、NF 快且覆蓋好——為什麼到不了品質角?(條件均值;可逆性限制表達力)
AR、EBM、DPM 品質覆蓋兼得——為什麼慢?(序列分解、MCMC、多步迭代各自是其本體)
同一條邊上的次序可以問:AR 與 EBM 誰更靠覆蓋角?(兩者都以 MLE 訓練,
EBM 的樣本品質另受抽樣不穩拖累)
若有人以一步蒸餾反駁「三者兼得」:上限來自教師,總成本含教師的訓練與多步推理。
-->

---
layout: center
class: text-center
---

# 休息 10 分鐘

<!--
時間配置:休息的 10 分鐘由③④吸收(③ 15 → 12、④ 12 → 8,節餘轉入⑤),
總長維持兩小時。
-->

---
layout: section
---

# AR

逐維分解,精確 logprob

<div class="mt-4">
<FamilyMatrix focus="AR" compact />
</div>

<div class="mt-3">
<Trilemma focus="AR" compact />
</div>

<!--
AR 在三難上的位置:品質與覆蓋一側,速度以序列抽樣支付,與 DPM / FM、EBM 同一條邊。
-->

---

# AR 的介面實作

一條恆等式,分解出每一項條件機率

$$\log p(x)=\sum_{t}\log p(x_t\mid x_{<t})$$

<div class="mt-4">

chain rule 把聯合分布拆成逐 token 條件機率的連乘;每一項都是一次 softmax 輸出,可以直接讀取。

| 介面 | 形式 | 代價 |
|---|---|---|
| `logprob(x)` | 精確,一次前向即得全部 | 幾乎免費 |
| `sample()` | 逐 token,天生序列 | 長度即延遲 |

</div>

<!--
chain rule 不是近似,是恆等式;AR 的 logprob 是所有家族中最便宜的。
sample 的序列性是同一個分解的另一面:第 t 個 token 的分布依賴前 t−1 個的取值。
-->

---
layout: none
---

<DemoFrame src="ar-2d-interactive.html" title="AR:逐維生成,精確 logprob" :maxH="470" />

<!--
[3 分鐘] 展示腳本:
1. 「生成一點(分步)」兩次:先抽第一維(邊際直方高亮),停頓後抽第二維——
   條件曲線隨第一維的取值不同而不同,序列性肉眼可見。
2. 點畫面兩處查 logprob:兩項相加;密度區讀數高、空白區讀數低。
3. 切換維度順序:同一點的兩個分項改變,總和不變——分解不唯一,密度唯一。
-->

---

# Cross-entropy 與 KL 的關係

兩者只差一項資料本身的熵

$$H(p,q)=H(p)+\mathrm{KL}(p\,\|\,q)$$

<div class="mt-4">

- 分類任務的目標 $p$ 是 one-hot,$H(p)=0$,所以 CE 與 forward KL 是同一個數
- 目標一旦變軟(label smoothing、distillation),兩者開始分離:$H(p)>0$ 成為 CE 的下限
- 語言建模最小化 CE,等價於最小化 forward KL:AR 家族住在 mode-covering 端的原因

</div>

<!--
這條恆等式一行證明:展開 CE 的定義,加減 E_p[log p]。
「loss 還很高」不一定代表模型差:H(p) 那一項是資料本身的熵,誰也壓不掉。
-->

---

# $H(p_{\text{data}})$ 估不出來

資料的熵沒有無偏估計途徑

資料只有樣本、沒有密度。CE 的絕對值因此**無法**回答「離最優還有多遠」。

<div class="mt-3">

四條實務路線:

| 路線 | 作法 |
|---|---|
| 同資料比差值 | $H(p)$ 是共同常數,模型間的 CE 差就是 KL 差 |
| 參考模型正規化 | 以另一個模型的 logprob 為基準(DDO 的 log ratio 即此形) |
| 已知熵的合成資料 | 人造分布,$H(p)$ 可解析計算 |
| 繞開 likelihood | MAUVE、MMD、下游任務指標 |

</div>

<!--
「p_data 沒有 logprob」這件事的又一次現身:第一堂在 reverse KL 與 reward model,
這裡在熵。四條路線中第一條最常用:leaderboard 上比的從來是相對值。
-->

---

# 跨 tokenizer 的比較:BPB

換一個與詞表無關的單位

同一份文本,不同 tokenizer 切出的 token 數不同,per-token 的 CE 之間不可比。換算到位元組:

$$\mathrm{BPB}=\frac{T}{N_{\text{bytes}}}\cdot\log_2 \mathrm{PPL}_{\text{token}}$$

<div class="mt-4">

$T$ 為 token 數、$N_{\text{bytes}}$ 為位元組數:把「每 token 的困惑度」換算成「每位元組的位元數」,單位與 tokenizer 無關。

</div>

<div class="mt-4 tone-muted">

比較不同詞表的模型、或同模型換 tokenizer 前後,以 BPB 為準。

</div>

<!--
推導:總 log loss(以 2 為底)= T·log₂PPL,除以位元組數即每位元組位元數。
資訊論讀法:模型作為壓縮器的碼長。
-->

---

# 訓練目標的精確形狀

forward KL 沿序列分解(對兩邊同時用 chain rule)

$$\mathrm{KL}(p\,\|\,q)=\sum_t\;\mathbb{E}_{x_{<t}\sim {\color{#2563eb}p}}\Big[\mathrm{KL}\big(p(\cdot\mid x_{<t})\,\big\|\,q(\cdot\mid x_{<t})\big)\Big]$$

<div class="mt-4">

注意期望的下標:**前綴取自 $p$**,也就是資料。

teacher forcing(訓練時餵真實前綴)是這個分解的直接實作,而非工程上的權宜。

</div>

<!--
推導(追問時展開):
1. 對 p、q 同時用 chain rule:log(p(x)/q(x)) = Σ_t log(p_t/q_t)。
2. KL = E_{x∼p}[Σ_t log(p_t/q_t)],把和拿出期望。
3. 對第 t 項,x 中 x_{≥t} 的部分先積掉(塔性質),留下對 x_{<t}∼p 的期望,
   內層正是逐步的 KL(p_t‖q_t)。
前綴邊際因此取自 p;這個下標正是 exposure bias 的根源。
-->

---

# Exposure bias:目標從未度量的那條軌跡

訓練與生成走在不同的前綴上

<ExposureBias />

<div class="mt-2 text-sm">

reverse KL 的分解要把**前綴分布與每步引數同時**對調:
$\mathrm{KL}(q\,\|\,p)=\sum_t\mathbb{E}_{x_{<t}\sim q}\big[\mathrm{KL}(q(\cdot\mid x_{<t})\,\|\,p(\cdot\mid x_{<t}))\big]$。
訓練目標裡沒有任何一項在 $q$ 的前綴上取期望;模型自己生成的軌跡,是損失函數的盲區。memory agent 長對話品質漂移的機制之一即在此。DDO 的壓低項恰好在 $p_{\text{ref}}$ 的樣本(即 $q$ 系的軌跡)上施力,補的正是這個盲區。

</div>

<!--
長對話漂移:每一輪回覆都以模型自己過去的輸出為前綴,誤差沿 t 累積,
而訓練從未在這種前綴上校正過條件分布。
量級:behavior cloning 的分析給出誤差隨生成長度平方級 O(εH²) 放大,
demo 中拉長生成步數,軌跡離開資料流形的機會隨之升高即此形狀。
-->

---
layout: none
---

<DemoFrame src="ar-2d-interactive-2.html" title="訓練軌跡與自由生成軌跡的分離" :maxH="470" />

<!--
[3 分鐘] 這頁的模型是現場訓練的(TensorFlow.js):一個小 MLP 讀最近 k 個點回歸下一步位移,
訓練批次只從真實軌跡上取樣,即 teacher forcing;「訓練」核取方塊預設關閉,示範時先勾起來。
左圖同時畫兩條軌道:藍色箭頭是每個真實點上的一步預測(前綴取自資料),
橘紅線是模型自己餵自己的 rollout(前綴取自模型)。第三個面板把兩者畫在同一單位下——
一步 RMSE 一路下降,rollout 平均偏離卻卡在高處,兩線的落差就是 exposure bias,
因為訓練損失從來沒有在橘紅那條軌道上取過期望。
中間面板是誤差複利的直接證據:離流形距離隨 rollout 步數上升,遠離藍色虛線(一步 RMSE)。
示範順序:① 圓 + k=1 訓到 step≈2000,rollout 貼合,建立正常樣態。
② 切 8 字 + k=1:交叉點上單看位置有兩個方向,條件分布雙峰而 MSE 只輸出平均,
藍色箭頭在交叉點直指中間、rollout 在那裡出軌;k 拉到 2(等於知道速度)當場痊癒——
這是資訊不足,還不是 exposure bias,和換什麼架構無關。
③ 切三圈一岔:岔口的位置與速度都相同,該續圈或出岔取決於已繞第幾圈,依賴長度約 25–56 步。
這個病不反映在離流形距離上(每步都貼著某條圓),要看左上角的圈數計數:
k≤8 蓋不住一圈,圈數失控;k=64 蓋過兩圈半,圈數穩定回到 3——
加脈絡有用,且只在依賴長度落進窗內的那一刻生效,即長對話漂移的 2D 版本。
④ 回到 8 字 + k=2 收斂後取消勾選「訓練」凍結模型,σ 拉到 0.06:
推論每步加微擾,誤差複利讓 rollout 明顯散開;再勾回訓練、ε 拉到 0.5(σ 不變)重訓 2000 步,
重新凍結量測——同樣的雜訊下 rollout 穩住,而一步 RMSE 幾乎不變。
ε 是 scheduled sampling:訓練中的自我前綴步也注入同一個 σ,訓練分布因此涵蓋推論分布。
收束:修復條件不是「有沒有用自己的樣本」,而是「訓練時見過的偏差是否涵蓋推論時的偏差」;
⚡ 擾動鈕是同一件事的單次版本,把 rollout 推離流形一次看它回不回得來。
補丁能緩解累積速度,結構原因(目標的期望下標在 p)不動;
且補丁自身有代價——scheduled sampling 假設逐步展開,與 Transformer 的
全前綴平行訓練相衝,實作需 two-pass decoding。
讀數每次重訓、每個起點都會跳動,現場示範請固定起點、比較同一段的量級而非單次數值。
-->

---

# False premise:結構裡沒有拒答

條件分布不知道自己的條件有多罕見

$p(y\mid x)$ 對**任何** $x$ 都良定義,包括 $p(x)\approx 0$ 的荒謬前提。

<div class="mt-4">

- forward KL 訓練獎勵「在資料上覆蓋」,語料裡的問題幾乎都伴隨回答;拒答作為輸出模式,結構上沒有位置,除非 post-training 明文補上(Kalai et al., 2025 從訓練目標與評測誘因分析幻覺的必然性)
- 「$p(y\mid x)$ 算得出來」與「$x$ 值得回答」是兩個獨立命題;false premise 偵測要判定的是後者(基準:(QA)²,Kim et al., 2023)

</div>

<!--
Kalai et al. (2025):預訓練目標 + 只獎勵正確率的評測,共同使「有把握地亂答」
成為最優策略。偵測 false premise 等於在條件分布之外另建一個對 x 本身的判斷,
模型主幹不自帶這個判斷。
-->

---

# AR 的改進史

四步在品質與覆蓋,一步在光譜位置

<Timeline :items="[
  { name: 'n-gram', note: '計數即條件機率;脈絡長度受限於統計強度' },
  { name: 'RNN / LSTM', note: '參數化條件分布,脈絡不再截斷', tag: '品質' },
  { name: 'attention → Transformer', year: '2017', note: '訓練可平行,scaling 自此可行', tag: '品質' },
  { name: 'scaling laws', year: '2020', note: '損失隨規模冪律下降,投資有可預測回報(Kaplan et al.)', tag: '品質・覆蓋' },
  { name: 'instruction tuning / RLHF', year: '2022', note: '往 mode-seeking 端移動:可用性換多樣性', tag: '光譜右移' },
]" />

<!--
每一步用三難的語言讀:前四步都在品質與覆蓋角,速度角原地不動;
最後一步是光譜位置的移動而非三難的突破。
-->

---

# 速度側與應用

序列抽樣的兩種提速

**序列生成的提速**(sample 介面的補強):

- speculative decoding:小模型先猜、大模型驗收,可證明不改變輸出分布
- multi-token prediction:一次前向押注多個 token

<div class="mt-5">

**應用**:對話與 agent、程式生成,以及 LLM-ASR 中 $p(\text{text})$ 的角色(雜訊通道的 prior 項)。

</div>

<!--
speculative decoding 的驗收步驟是精確的 rejection sampling,分布不變、期望步數下降。
LLM-ASR 一列與第一堂的第 4 層對應。
-->

---
layout: section
---

# Normalizing Flow

可逆變換,一步抽樣

<div class="mt-4">
<FamilyMatrix focus="NF" compact />
</div>

<div class="mt-3">
<Trilemma focus="NF" compact />
</div>

<!--
三難上的位置:一步抽樣佔住速度角,logprob 又精確;
表達力受可逆性約束,品質角是它的短邊。
-->

---

# NF 的介面實作

以可逆變換把標準高斯映射到資料分布,變數變換公式給出精確密度

$$\log p_x(x)=\log p_z\big(f^{-1}(x)\big)+\log\big|\det J_{f^{-1}}(x)\big|$$

<div class="mt-3">

- 行列式項是體積修正:空間被拉伸處密度變稀、壓縮處變密,總機率守恆
- `logprob(x)`:逆向映射得 $z$、查 base 密度、累加各層 log-det,精確值三步到手
- `sample()`:$z\sim\mathcal N(0,I)$,前向一步

</div>

<div class="mt-3 aside aside-data text-sm tone-muted">

任意 $D\times D$ 行列式要 $O(D^3)$;NF 以架構設計讓 Jacobian 呈三角形,行列式退化為對角線連乘,降為 $O(D)$。離散 NF 的架構史,大半是三角結構的設計史。

</div>

<!--
兩個硬條件:bijection(同維度、可逆、可微)與 tractable Jacobian。
三角化的三步:NICE(Dinh et al., 2014)additive coupling,det 恆為 1;
RealNVP(Dinh et al., 2016)affine coupling,log|det| = Σs;
Glow(Kingma & Dhariwal, 2018)可學習 invertible 1x1 conv,LU 參數化。
coupling 每層只動一半維度,層間以 permutation 混合。
-->

---
layout: none
---

<DemoFrame src="nf-2d-interactive.html" title="Normalizing Flow:可逆變換與精確密度" :maxH="470" />

<!--
[3 分鐘] 這頁的 flow 是現場訓練的:一疊 affine coupling(RealNVP 型),
直接最小化精確 NLL(forward KL / MLE),無對抗、無下界;損失以 bits/dim 讀出,
與 AR 的 BPB 同單位可比。按「開始訓練」後即可操作。
展示腳本:
1. 拉「層 k」滑桿(或讓它自動播放):高斯網格被前 k 層逐層折成資料分布,
   全程不撕裂、不重疊——可逆性的視覺形式;k=L 的粒子就是生成樣本,一步完成。
2. 中圖熱圖是逐格 inverse pass 算出的精確 log p(x),不是判別器分數、也不是 ELBO——
   全系列 demo 裡唯一的精確密度圖。點擊任一位置即呼叫 logprob 介面,
   讀出數值,並在左圖畫出 x 逐層反推回 z 的路徑。
3. 層數 2 → 8:coupling 單層表達力弱,表達力用深度買(層數 2 蓋不住雙月)。
4. 拓撲稅:選「高斯混合(中心偏移)」,各群之間必然留下細絲;選「圓環」,
   中心漏出一道無法歸零的密度——連續可逆變換是同胚,不能剪開也不能戳洞。
   同題對照:VAE 搭橋而糊、GAN 漏群而塌縮、FM 多步搬運而慢、NF 留細絲。
5. 這個 k 滑桿與 FM demo 的 t 滑桿是同一件事的兩種粒度:NF 是離散層的堆疊、
   FM 是連續時間的場,機率流 ODE 即層數趨於無窮的連續極限。
-->

---

# 兩個介面的價格可以對調:MAF 與 IAF

自迴歸依賴放在哪一側,決定哪個介面便宜

| | MAF | IAF |
|---|---|---|
| 自迴歸參數依賴 | $x_{1:i-1}$(資料側) | $u_{1:i-1}$(雜訊側) |
| logprob | 一次並行前向(快) | 逐維還原 $u$,$O(D)$ 串行(慢) |
| sample | 逐維串行 $O(D)$(慢) | 一次抽 $u$,並行生成(快) |
| 適用 | density estimation、MLE 訓練 | 即時生成、變分後驗 |

<div class="mt-4 text-sm">

同一條 chain rule,擺在資料側或雜訊側,兩個介面的成本表整個對調。Parallel WaveNet 的解法:MAF 型 teacher 高效訓練,蒸餾給 IAF 型 student;student 對自己生成的樣本算 logprob 是並行的($u$ 已知),兩側的快同時到手(van den Oord et al., 2018)。

</div>

<!--
MAF(Papamakarios et al., 2017)、IAF(Kingma et al., 2016)。
「介面成本」語言的教科書案例:同一族模型,價格表對調;
蒸餾繞過兩難的原因是 logprob 只需要對自家樣本便宜。
-->

---

# 連續化:行列式退化為跡

層數趨於無窮的極限是一條 ODE(Neural ODE,Chen et al., 2018)

$$\frac{dz}{dt}=f(z,t),\qquad \frac{d\log p(z(t))}{dt}=-\mathrm{Tr}\!\left(\frac{\partial f}{\partial z}\right)$$

<div class="mt-3">

- 非線性的行列式在連續極限下換成線性的跡:$O(D^3)\to O(D)$,架構約束隨之解除
- 可逆性免費取得:向量場 Lipschitz 連續則軌跡不相交(Picard–Lindelöf),不再需要 coupling 結構
- FFJORD(Grathwohl et al., 2018):Hutchinson 估計跡,免建 Jacobian
- 代價:MLE 訓練的每一步都要呼叫數值 ODE solver,慢且不穩

</div>

<!--
「跡是線性算子」是關鍵:對疊加封閉,任意網路皆可當向量場。
adjoint 法使顯存與深度無關。連續化後,「一步生成」變成「積分一條 ODE」,
步數重新成為成本,與離散 NF 的一步抽樣是不同的取捨。
-->

---

# NF 的特徵性失效與今日角色

OOD 悖論、結構限制,與剩下的戰場

**OOD 悖論**:在複雜影像上以 MLE 訓練的 NF,常給沒見過的更簡單影像更高的 log-likelihood(Nalisnick et al., 2019):forward KL 只要求資料處機率高,不禁止別處更高;低複雜度影像經可逆映射聚在高斯原點附近的高密度區。精確似然與可靠的 OOD 偵測是兩件事。

<div class="mt-3">

**結構限制**:可逆映射不能改變拓撲;維度必須全程保持,無法下採樣壓縮;單層表達力弱,迫使極深堆疊。

</div>

<div class="mt-3">

**今日角色**:語音 vocoder(WaveGlow 並行生成)、科學計算(Boltzmann Generators 一步抽分子平衡態)、變分後驗(IAF)、模擬推斷與宇宙學。

</div>

<!--
OOD 悖論的第三個機制:coupling 的 mask 偏向擬合局部像素相關性,
學到的更像通用壓縮器而非語義模型。
影像戰場失守的三個原因:像素空間 MLE 把容量花在不可見高頻;
確定性軌跡在高斯薄殼上無糾錯能力;維度不變擋住 latent 壓縮路線。
-->

---
layout: section
---

# VAE

latent 變數,下界訓練

<div class="mt-4">
<FamilyMatrix focus="VAE" compact />
</div>

<div class="mt-3">
<Trilemma focus="VAE" compact />
</div>

---

# VAE 的介面實作

以下界取代不可解的邊際似然

引入 latent $z$:$p_\theta(x)=\int p_\theta(x\mid z)\,p(z)\,dz$,積分不可解,改而最大化下界:

$$\log p_\theta(x)\;\ge\;\mathrm{ELBO}=\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]-\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)$$

| 介面 | 形式 |
|---|---|
| `logprob(x)` | 僅下界(ELBO);與真值的差是 $\mathrm{KL}(q_\phi\,\|\,p_\theta(z\mid x))$ |
| `sample()` | $z\sim p(z)$,decoder 一次前向 |

<!--
ELBO 推導:Jensen 不等式(課前自檢第 4 題)套在 log∫ 上。
gap 可再拆兩層:approximation gap(對角高斯變分族本身的表達力上限,
最佳化到完美也不歸零)與 amortization gap(共享 encoder 無法對每筆資料
輸出各自的最優後驗)。
latent 邊際化的記號與第一堂 ICL 隱式貝氏那條積分同語言:對潛在變數積分。
-->

---

# 特徵性失效:過度平滑

Gaussian 重建項與 mode-covering 的疊加

重建項用 Gaussian likelihood 時,$\log p_\theta(x\mid z)$ 等價於負 MSE;
MSE 的最優解是**條件均值**:多個可能輸出的平均,而非其中任何一個。

<div class="mt-4">

疊上 mode-covering 的訓練目標,結果是樣本模糊、細節被抹平:兩個機制往同一個方向推。

</div>

<div class="mt-4 tone-muted">

這是失效型態由訓練目標決定的又一例:模糊寫在 Gaussian × forward KL 這個組合裡,調架構調不掉。

</div>

<!--
一維類比:資料一半是 +1 一半是 −1,MSE 最優預測是 0,誰也不像。
影像上同理:紋理的多種可能被平均成糊。
-->

---
layout: none
---

<DemoFrame src="vae-2d-interactive.html" title="VAE:過度平滑、拓撲、β 的兩端" :maxH="470" />

<!--
[3 分鐘] 觀察順序:
1. mode covering 的過度平滑:樣本雲糊在峰間,對照上一頁。
2. 環狀資料:高斯 prior 蓋不乾淨環形拓撲,雲的中心破洞被填掉。
3. β 過大:重建垮,樣本全糊;β 過小:latent 空洞化,prior 抽樣落空。
   兩種壞法對應 ELBO 兩項的失衡。
補充兩個正式名稱:posterior collapse——encoder 追不上移動中的真後驗,
強 decoder 乾脆忽略 z,之後 encoder 再無梯度(修法:cyclical KL annealing、
free bits);prior holes——aggregate posterior 與 N(0,I) 不重合,
prior 抽樣落入空洞即解碼成糊,環狀資料的破洞正是此現象,
修法之一是在 latent 空間再訓一個先驗模型。
-->

---

# VAE 的改進史與今日角色

三步對付模糊,最後轉為基礎設施

<Timeline dense :items="[
  { name: 'β-VAE', year: '2017', note: '調 KL 項係數,換 disentanglement(Higgins et al.)', tag: '控制' },
  { name: 'VQ-VAE', year: '2017', note: '離散 latent:decoder 無法忽略,繞開 posterior collapse(van den Oord et al.)', tag: '結構' },
  { name: 'VQ-GAN', year: '2021', note: '加對抗損失,重建變銳利(Esser et al.)', tag: '品質' },
  { name: '壓縮器角色', year: '今日', note: 'Stable Diffusion 在 VAE latent 空間跑 diffusion', tag: '基礎設施' },
]" />

<div class="mt-3">

**應用**:表徵學習、異常偵測(ELBO 作為異常分數),以及作為其他生成模型的 latent 基礎設施。

</div>

<!--
改進史的方向一致:對付「糊」。最終定位不在正面戰場,
而是把「快 + 覆蓋 + 可微下界」三個長處打包,當別家的壓縮層。
-->

---
layout: section
---

# GAN

一步映射,沒有密度

<div class="mt-4">
<FamilyMatrix focus="GAN" compact />
</div>

<div class="mt-3">
<Trilemma focus="GAN" compact />
</div>

---

# GAN 的介面實作

只有 sample,沒有密度簿記

$$x = G(z),\qquad z\sim\mathcal N(0,I)$$

<div class="mt-4">

一步映射,只有 `sample()`。$G$ 不維護任何密度,logprob 那一欄是空的:

- 訓練只能透過判別器代理(本堂開頭的代理表)
- 統一引導式、DPO、DDO 全部以 logprob 為前提,對 GAN 無一適用
- 換得的是不受可逆性與序列分解約束的 generator 架構

</div>

<!--
判別器 D 與 G 交替訓練:D 學分辨真假,G 學騙過 D。
第一堂的判別器讀法給了理論對應:最優 D 下,G 的目標值是 2·JSD − 2log2;
實務用的損失與此不同,對照表見損失頁。
-->

---

# 理論上的 JSD,實務上的別種東西

換掉損失,也換掉了對應的散度

$2\,\mathrm{JSD}-2\log 2$ 只在最優判別器處成立,而那裡正是梯度消失的位置(JSD 飽和)。實務改用 non-saturating 損失,對應的散度變成(Arjovsky & Bottou, 2017):

$$\mathrm{KL}\big(p_g\,\|\,p_{\text{data}}\big)-2\,\mathrm{JSD}\big(p_g\,\|\,p_{\text{data}}\big)$$

| | 原始 minimax | non-saturating |
|---|---|---|
| 對應散度 | JSD | reverse KL $-\ 2\cdot$JSD |
| 病理 | 梯度消失 | mode-seeking、易塌縮 |

<div class="mt-3 text-sm tone-muted">

換損失治好了梯度消失,也一併換掉了 mode-covering:每種損失各配一種失效。

</div>

<!--
梯度陳述(Arjovsky & Bottou 2017, Thm 2.5):non-saturating 目標在最優 D 下的
梯度等於 KL(p_g‖p_data) − 2·JSD 的梯度(是梯度相等,不是目標值相等)。
第一項是 reverse KL:mode seeking 的來源;第二項負號與 JSD 反向,
越分得開越受獎勵,是震盪與不穩定的來源——此機制細節口頭講,頁面只留對照表。
-->

---

# Mode collapse 的第一層成因

覆蓋率沒有進入 generator 收到的訊號

generator 的損失裡**沒有資料項**:$G$ 只從 $D$ 的評分獲得訊號,而 $D$ 逐點評分。

<div class="mt-4">

覆蓋率是分布層級的性質;逐點介面沒有承載它的欄位。漏掉一整個眾數時,只要現有樣本能騙過 $D$,損失就不會抗議。

</div>

<div class="mt-4 aside aside-model">

與 reward model 的結構極限同構:兩個逐點代理,同一個盲區。第一堂裡防線只剩 β;這裡連 β 都沒有。

</div>

<div class="mt-3 text-sm tone-muted">

對症的修補是給介面加欄位:minibatch discrimination 把 batch 內樣本相似度的總和直接接進判別器的輸入,塌縮批次立即被抓(Salimans et al., 2016)。

</div>

<!--
更深層的成因(交替最佳化的動力學、D 的容量、G 的參數化)各有文獻。
動力學的一句話圖像:G 過擬合當前 D、塌到單一安全眾數,D 追上後 G 跳到
下一個眾數,循環不收斂;Unrolled GANs(Metz et al., 2017)讓 G 對 k 步後的
D 最佳化,消掉這個 limit cycle。課堂只展開第一層。
-->

---
layout: none
---

<DemoFrame src="gan-2d-interactive.html" title="Mode collapse:判別器地景看得見,generator 收不到" :maxH="470" />

<!--
[5 分鐘] 訓練到塌縮,疊上判別器地景:
被漏掉的眾數在地景上是高值區(D 明知那裡是真資料),
但 G 的梯度只來自它自己樣本所在的位置,那片高值區的資訊傳不過去。
畫面上沒有任何訊號告訴 G「少了一個眾數」,這就是逐點介面的盲區。
-->

---

# 判別器想法的單向轉移

借用只在一個方向上成立

第一堂 DDO 的原型即是這裡的 $d^*=p/(p+q)$:「用自身 logprob 造判別器」把 GAN 的核心構造搬進了 logprob 家族。

<div class="mt-4">

| 方向 | 可行性 |
|---|---|
| 判別器構造 → logprob 家族(DDO) | 可行:log ratio 直接可算,無需另訓網路 |
| logprob 方法 → GAN | 不可行:$p_\theta$ 寫不出來,式子無從成立 |

</div>

<div class="mt-4">

介面多的一方可以借介面少的一方的構造,反向沒有著力點。

</div>

<!--
這是代理表「DDO 借用哪一格」的收尾:第一列可以借第三列的武器,
第三列借不了第一列的,因為武器本身就是那個缺的介面。
-->

---

# GAN 的改進史與今日角色

穩定、品質、規模,最後轉向蒸餾

<Timeline dense :items="[
  { name: 'DCGAN', year: '2015', note: '穩定可複製的卷積配方(Radford et al.)', tag: '品質' },
  { name: 'conditional GAN', note: '條件化生成', tag: '控制' },
  { name: 'WGAN', year: '2017', note: 'Earth Mover 距離替代 JSD,支撐集分離時仍有梯度(Arjovsky et al.)', tag: '穩定' },
  { name: 'StyleGAN', year: '2019', note: '品質與可控性(Karras et al.)', tag: '品質' },
  { name: 'BigGAN', year: '2019', note: '大規模訓練(Brock et al.)', tag: '品質' },
  { name: '蒸餾目標', year: '近年', note: '少從頭訓練;把多步模型壓成一步的對抗式蒸餾', tag: '速度' },
]" />

<div class="mt-2">

**應用**:低延遲與即時場景、超解析、風格與語音轉換、diffusion 的加速蒸餾。

</div>

<!--
WGAN 對付的正是第一堂那條 JSD 飽和曲線。具體對照:兩個平行分布相距 θ 時,
KL 為無窮、JSD 飽和為常數,而 Wasserstein 距離 = |θ| 呈線性,梯度不消失。
今日角色的轉變:一步生成的能力被保留,訓練難的部分讓給別的家族。
-->

---
layout: section
---

# Energy-Based Model

未正規化的密度,MCMC 抽樣

<div class="mt-4">
<FamilyMatrix focus="EBM" compact />
</div>

<div class="mt-3">
<Trilemma focus="EBM" compact />
</div>

<!--
三難上的位置:與 DPM 同在慢側(抽樣靠 MCMC),品質另受訓練不穩拖累;
矩陣裡它佔一種新的 logprob 形式:未正規化。
-->

---

# EBM 的介面實作

能量給出相對機率,$\log Z$ 攔住絕對值

$$p(x)=\frac{e^{-E(x)}}{Z},\qquad Z=\int e^{-E(x)}\,dx$$

任意純量網路都可以當 $E$:低能量即高機率,約束最少的一種參數化。

| 介面 | 形式 |
|---|---|
| `logprob(x)` | 只到未正規化為止:$-E(x)-\log Z$,而 $\log Z$ 算不出來 |
| `sample()` | 原生沒有:靠 Langevin dynamics 逐步迭代逼近 |

<div class="mt-3 text-sm">

$Z$ 有多難:$224\times224\times3$ 的二值影像要對 $2^{150528}\approx10^{45000}$ 個狀態加總。相對比較則不受影響:兩點的 $-E$ 之差就是 log 機率比,$\log Z$ 相消。

</div>

<!--
Langevin:x_{t+1} = x_t − η∇E(x_t) + √(2η)ξ;∇ₓlog Z = 0,只需未正規化的梯度。
「差一個常數的 logprob」是矩陣裡的新格:比 VAE 的下界更弱(絕對值不可得),
比 GAN 的空格強(相對比較、重排序、OOD 偵測都可用)。
-->

---
layout: none
---

<DemoFrame src="ebm-2d-interactive.html" title="EBM:能量地景與 Langevin 抽樣" :maxH="470" />

<!--
[3 分鐘] 展示腳本:
1. 近距雙峰:啟動 20 條鏈,落谷,兩峰都有人口。
2. 點查能量:兩點可比高低,但絕對 logprob 差一個未知的 log Z。
3. 切遠距雙峰:同樣的 Langevin,跨峰計數器幾乎不動——sample 介面的
   成本與 mode mixing 的困難一眼可見。
-->

---

# 沒有 log Z,怎麼訓練

MLE 梯度拆成兩個相位($-\nabla_\theta\log Z$ 化為對模型分布的期望)

$$\nabla_\theta\log p(x)=\underbrace{-\nabla_\theta E(x)}_{\text{壓低真實資料的能量}}\;+\;\underbrace{\mathbb E_{x'\sim p_\theta}\big[\nabla_\theta E(x')\big]}_{\text{拉高自生樣本的能量}}$$

| 方法 | 手法 | 對應的差異度量 |
|---|---|---|
| Contrastive Divergence<br><span class="fine">Hinton, 2002</span> | 鏈從資料點起跑只走 $k$ 步;$\log Z$ 梯度對消 | $\mathrm{KL}(p_{\text{data}}\|p_\theta)-\mathrm{KL}(p_k\|p_\theta)$ |
| Score matching<br><span class="fine">Hyvärinen, 2005</span> | 改對 $x$ 微分:$\nabla_x\log Z=0$,$Z$ 直接消失 | Fisher divergence |
| NCE<br><span class="fine">Gutmann & Hyvärinen, 2010</span> | 與已知雜訊分布做二元分類,$Z$ 當可學純量 | BCE(漸近 KL) |

<!--
negative phase 需要模型自己的樣本:每次參數更新都得跑 MCMC;
實務以 persistent chain / replay buffer 攤銷(Du & Mordatch, 2019)。
denoising score matching:對加噪資料做 score matching,雜訊核的 score
有解析形式,訓練化為預測所加雜訊的監督任務。
「拉高自生樣本能量」與第一堂 DDO 的壓低項是同一種力:
DDO 把這個 negative phase 寫進了 likelihood ratio。
-->

---

# EBM 的特徵性失效

抽樣、混合、穩定性、評估

- **抽樣即成本**:一次生成要數百至數千步 Langevin,每步一次前向加反向;對照一次前向即生成的家族,ImageNet 規模的訓練時間由小時級膨脹至年級
- **Mode mixing**:眾數間隔著高能量障壁時,鏈困在單一能量谷,mixing time 隨障壁高度指數成長,各眾數的比例難以還原
- **訓練發散**:能量無上界約束,OOD 區域會長出訓練資料中不存在的假能量井,sampler 掉入後訓練崩壞;與 BatchNorm 尤其不相容
- **評估困難**:exact likelihood 不存在,只能以 Annealed Importance Sampling 等昂貴近似

<!--
demo 的遠距雙峰情境正是 mode mixing 的可視化。
BN 不相容的機制:真實資料與高噪 MCMC 樣本的批統計劇烈波動,最佳化震盪。
-->

---

# EBM 的改進史與今日角色

從 Hopfield 到 JEM

<Timeline dense :items="[
  { name: 'Hopfield 網路・Boltzmann Machine', year: '1982–85', note: '能量地景支配網路狀態;隨機隱單元(Hopfield;Ackley, Hinton & Sejnowski)', tag: '概念' },
  { name: 'RBM', year: '1986', note: '二部圖限制使 Gibbs sampling 可行(Smolensky)', tag: '可訓練' },
  { name: 'Contrastive Divergence', year: '2002', note: '短鏈繞過平衡態;2006 年疊層預訓練帶動深度學習復興(Hinton)', tag: '訓練' },
  { name: '深度 ConvNet 能量函數', year: '2016', note: '現代網路參數化 E(Xie et al.)', tag: '品質' },
  { name: 'ImageNet 規模', year: '2019', note: 'replay buffer 穩定 SGLD(Du & Mordatch)', tag: '規模' },
  { name: 'JEM', year: '2020', note: '任何 softmax 分類器都暗含 EBM:E(x) = −LogSumExp(logits)(Grathwohl et al.)', tag: '統一' },
]" />

<div class="mt-2 text-sm">

**今日角色**:score $=-\nabla_x E$,能量地景的梯度場;OOD 與異常偵測(energy score)、機器學習勢能面(分子模擬)、序列級重排序(residual EBM 對整句評分,緩解逐 token 生成的誤差累積)。

</div>

<!--
JEM 的讀法:E(x,y) = −f(x)[y],對 y 邊際化得 E(x) = −LogSumExp(logits);
既有分類器免重訓即可做生成與 OOD 偵測。
residual EBM:P(x) ∝ P_LM(x)·e^{−E(x)},以 LM 當雜訊分布用 NCE 訓練,
讓雙向模型參與序列級評分,是第一堂第 4 層(重排序)的訓練期版本。
-->

---
layout: section
---

# DPM / Flow Matching

多步分解,每步一次迴歸

<div class="mt-4">
<FamilyMatrix focus="DPM" compact />
</div>

<div class="mt-3">
<Trilemma focus="DPM" compact />
</div>

---

# DPM 的介面實作

把生成拆成一疊簡單迴歸

AR 沿**序列**分解 forward KL;DPM 沿**雜訊尺度**分解同一個散度:
把資料逐步加入雜訊直到成為純雜訊,模型學每一小步的還原。

<div class="mt-3">

- 每一步是一次簡單迴歸(預測雜訊或速度場),訓練穩定性與 AR 相當
- `logprob(x)`:有變分下界;經 probability flow ODE 可精確計算(Song et al., 2021)
- `sample()`:多步迭代,一步一次前向

</div>

<div class="mt-4 tone-muted">

同一個 forward KL,兩種切法:AR 切在維度之間,DPM 切在訊噪比之間。

</div>

<!--
加噪方向是固定的(不學),學的只有還原方向。
score 的另一個名字:−∇E。denoising score matching 出自 EBM 的訓練工具箱
(Hyvärinen 2005;Song & Ermon 2019),diffusion 學的 s(x,t) 就是各雜訊尺度下
能量地景的梯度場;把離散步驟統一成 SDE 之後,PF-ODE 的精確 logprob
用的正是 NF 段的 instantaneous change-of-variables(跡)。
「把難問題切成一疊簡單迴歸」是 DPM 訓練穩定的來源:
沒有對抗、沒有配分函數、沒有可逆性約束。
-->

---

# 特徵性失效:慢

優勢幾乎全數以抽樣步數支付

一張樣本要跑幾十到上千次前向。

<div class="mt-4">

矩陣裡 DPM 的 sample 欄與三難裡 DPM 的位置說的是同一件事;
於是這個家族的改進史,大半是一部**減步數**的歷史。

</div>

<!--
「慢」不是實作不佳:多步是這個分解方式的本體;
改進史因此有一條清楚的主線:減步數。
-->

---

# 改進史(上):從千步到少步

三篇論文,步數降兩個量級

<Timeline dense :items="[
  { name: 'DDPM', year: '2020', note: '離散時間逐步去噪,千步量級(Ho et al.)', tag: '品質・覆蓋' },
  { name: 'DDIM', year: '2020', note: '非 Markov 過程族,η=0 得確定性抽樣;已訓練的 DDPM 權重免重訓,步數大減(Song et al.)', tag: '速度' },
  { name: 'Score-based SDE', year: '2021', note: '離散步驟統一成連續時間 SDE;probability flow ODE 由此而來(Song et al.)', tag: '理論統一' },
]" />

<!--
DDIM 與 Score-SDE 幾乎同期(2020 年底);概念關係:DDIM 的確定性抽樣
正是 probability flow ODE 的離散化特例,SDE 框架把它解釋清楚。
-->

---

# CFG 與 zero-shot 編輯

統一引導式在這個家族的原始形式(Ho & Salimans, 2022)

$$\log p_w = \log p(x\mid c) + w\big(\log p(x\mid c)-\log p(x)\big)$$

<div class="mt-4">

其中一類 zero-shot 編輯直接套用這個形式:$p_A$ 條件於**原圖**、$p_B$ 為無條件分布、$w$ 控制改動幅度。InstructPix2Pix 的雙 guidance scale(影像一個係數、指令一個係數)是此式加到兩個比值項的推廣,每個條件各配一個係數(Brooks et al., 2023)。

</div>

<div class="mt-3 tone-muted text-sm">

不需重訓、不需配對資料:同一個係數,在影像家族裡調的是「聽指令的程度」。

</div>

<!--
CFG 訓練時隨機丟棄條件,同一個網路同時學到有條件與無條件分數;
推論時外插,每步兩次前向,延遲加倍。線性外插套在確定性 flow 上會把
軌跡推離資料流形(過飽和、結構崩壞),predictor-corrector 類修法把外插改內插。第一堂表裡 CFG for LLM 那一列,原產地在此。
其他編輯法(RePaint 的替換式 inpainting、SDEdit)用的是別的機制,不塞進這條式子。
-->

---

# 改進史(下):換空間、換目標、換步數

latent 空間、速度場、一步蒸餾

<Timeline dense :items="[
  { name: 'Latent Diffusion', year: '2022', note: '在 VAE latent 空間跑 diffusion,算力降一個量級(Rombach et al.)', tag: '速度' },
  { name: 'Flow Matching / Rectified Flow', year: '2023', note: 'CNF 的免模擬訓練法:在插值路徑上直接迴歸速度場;源分布不必是 Gaussian(Lipman et al.;Liu et al.)', tag: '簡化' },
  { name: 'Consistency Models / 對抗式蒸餾', year: '2023', note: '多步 ODE 積分蒸餾成一步(Song et al.);對抗式蒸餾另見 ADD(Sauer et al.)', tag: '速度' },
]" />

<div class="mt-3 text-sm tone-muted">

Latent Diffusion 的壓縮層即 VAE 段的收尾;一步蒸餾的對抗損失即 GAN 段的今日角色:三個家族在這條時間軸上會合。

</div>

<!--
Flow Matching 的訓練是純迴歸:抽一對 (x₀, x₁),在插值路徑上迴歸速度場,
不需要模擬整條軌跡。源分布放開 Gaussian 之後,配對、橋接類任務直接受益。
-->

---

# 另一條提速路線:改權重,不減步數

省下的是每步的第二次前向

對原以 CFG 推論的模型,DDO 微調後**免 guidance** 的品質超過原 CFG 基線,每步省下一次前向(Zheng et al., 2025 專案頁);下表為免 guidance 的前後對照:

| 模型 | 資料集 | FID(前 → 後,無 guidance) |
|---|---|---|
| EDM | CIFAR-10 | 1.79 → 1.30 |
| EDM2 | ImageNet-64 | 1.58 → 0.97 |
| EDM2 | ImageNet 512×512 | 1.96 → 1.26 |
| VAR-d30(AR 家族) | ImageNet 256×256 | 4.74 → 1.79 |

<div class="mt-3 text-sm">

每輪微調成本低於預訓練 epoch 數的 1%,可 self-play 疊代。上一堂兩則定性觀察(MLE 到頂、截斷藏缺陷)的數字版即在此表。

</div>

<!--
數字取自 Zheng et al. (2025) 專案頁。CIFAR-10 的 EDM 基線本即免 guidance,
「省一次前向」的敘述不適用該列。表列同時含 diffusion 與 AR 家族:
方法的適用邊界只由 logprob 介面決定,與家族的其他細節無關,
這正是介面語言的預測力。
-->

---
layout: none
---

<DemoFrame src="flow-matching-2d-interactive.html" title="Flow Matching:同一個散度的另一種分解" :maxH="440" />

<div class="px-6 pt-2 text-sm tone-muted">

應用:影像、影片與音訊生成、分子設計、動作生成。

</div>

<!--
[3 分鐘] 展示:向量場把源分布連續搬運成資料分布;
把源分布換成非 Gaussian,訓練照常進行。
軌跡的平直程度對應少步抽樣的可行性,因果鏈:獨立配對的 (x₀,x₁)
產生交叉路徑,交叉處模型平均相互衝突的速度、軌跡彎曲,積分需小步;
reflow 用自己上一輪的非交叉配對重訓,軌跡拉直後一階 Euler 幾步即可。
-->

---
layout: section
---

# 定位

六個題目,三個座標系

---

# 實驗室題目的完整定位

光譜位置、呼叫的介面、三難取捨

| 題目 | 光譜位置 | 呼叫的介面 | 三難取捨 |
|---|---|---|---|
| prompt engineering | 只換 base 項,係數不動 | sample | 不觸及 |
| memory agent | 同上;條件集合的設計 | sample | 不觸及 |
| 情感支持對話 | 右端,受 $\beta$ 約束 | sample(+logprob 可調) | 覆蓋 vs 品質 |
| false premise 偵測 | 左端訓練的結構後果 | logprob | 不觸及 |
| 信心與正確率 | logprob 讀數的可信度(校準) | logprob | 不觸及 |
| LLM-ASR | log 空間線性組合 | 兩者 | 速度(n-best 的 n) |

<!--
[約 3 分鐘] 開場那張兩欄表的完成態。逐列快速走過,
每一列的術語如今都有定義:光譜(第一堂①)、介面(全課)、三難(本堂④)。
-->

---

# 結語:往哪裡走

進入第 2、3 層的成本

實驗室現行方法多在**第 1 層與第 4 層**,只呼叫 `sample()`:對黑箱 API 也可行,這是它們成為主力的理由。

<div class="mt-4">
<LayerStack :focus="1" />
</div>

<div class="mt-4">

第 2、3 層額外需要的只有 `logprob`;實驗室使用的每一個模型都提供這個介面,而這兩層的方法**不需要任何額外訓練資源**。

</div>

<!--
不是呼籲放棄第 1、4 層,是指出成本結構:進入第 2、3 層,
新增的需求只有一個本來就有的介面。
-->

---
layout: statement
---

# 同一條光譜的兩端

<div class="text-xl leading-relaxed mt-8">

含糊的迴避出自 forward KL 訓練,千篇一律出自 reverse KL 對齊:
同一條光譜的兩端。

</div>

<div class="mt-8 text-base tone-faint">

訓練目標、解碼設定、權重微調,三個層次各有一個可移動的座標;
六個家族,各以自己的介面決定哪些移動可行。

</div>

<!--
第一堂開場的主張至此驗畢:兩種現象、一個連續體、三個施力層次、六種構造。
-->

---

# 參考文獻(1/2)

散度、引導、對齊、條件化

<div class="text-xs leading-relaxed grid grid-cols-2 gap-x-6">
<div>

**散度與機率背景**
- Stanford CS236, Lectures 1–2
- Bishop & Bishop, *Deep Learning: Foundations and Concepts*
- Endres & Schindelin (2003), *A New Metric for Probability Distributions*
- Arjovsky & Bottou (2017), *Towards Principled Methods for Training GANs*

**引導與解碼**
- Holtzman et al. (2020), *The Curious Case of Neural Text Degeneration*
- Li et al. (2023), *Contrastive Decoding*
- Chuang et al. (2024), *DoLa*
- Sanchez et al. (2023), *Stay on Topic with Classifier-Free Guidance*
- Ho & Salimans (2022), *Classifier-Free Diffusion Guidance*
- Karras et al. (2024), *Autoguidance*

</div>
<div>

**對齊與 DDO**
- Ouyang et al. (2022), InstructGPT
- Rafailov et al. (2023), *Direct Preference Optimization*
- Kirk et al. (2024), *Understanding the Effects of RLHF*
- Zheng et al. (2025), *Direct Discriminative Optimization*(ICML)
- Chen et al. (2024), *SPIN*

**條件化與量測**
- Xie et al. (2022), *In-context Learning as Implicit Bayesian Inference*
- Liu et al. (2024), *Lost in the Middle*
- Wang et al. (2023), *Self-Consistency*
- Kalai et al. (2025), *Why Language Models Hallucinate*
- Kim et al. (2023), *(QA)²: Question Answering with Questionable Assumptions*

</div>
</div>

---

# 參考文獻(2/2)

分類、三難,與各家族的演進

<div class="text-xs leading-relaxed grid grid-cols-3 gap-x-5">
<div>

**分類與三難**
- Tomczak, *Deep Generative Modeling*, 2nd ed. (2024)
- Xiao, Kreis & Vahdat (2022), *Generative Learning Trilemma*

**家族演進**
- Kaplan et al. (2020), *Scaling Laws*
- Higgins et al. (2017), β-VAE;van den Oord et al. (2017), VQ-VAE;Esser et al. (2021), VQ-GAN
- Radford et al. (2015), DCGAN;Arjovsky et al. (2017), WGAN;Karras et al. (2019), StyleGAN
- Metz et al. (2017);Salimans et al. (2016)

</div>
<div>

**Normalizing Flow**
- Dinh et al. (2014), NICE;Dinh et al. (2016), RealNVP;Kingma & Dhariwal (2018), Glow
- Kingma et al. (2016), IAF;Papamakarios et al. (2017), MAF;van den Oord et al. (2018), Parallel WaveNet
- Chen et al. (2018), Neural ODE;Grathwohl et al. (2018), FFJORD
- Nalisnick et al. (2019)

**Energy-Based Model**
- Hopfield (1982);Ackley, Hinton & Sejnowski (1985);Smolensky (1986), RBM
- Hinton (2002), CD;Hyvärinen (2005);Gutmann & Hyvärinen (2010), NCE
- Xie et al. (2016);Du & Mordatch (2019);Grathwohl et al. (2020), JEM

</div>
<div>

**Diffusion / Flow Matching**
- Ho et al. (2020), DDPM;Song et al. (2020), DDIM
- Song et al. (2021), Score-SDE;Song & Ermon (2019)
- Rombach et al. (2022), Latent Diffusion
- Lipman et al. (2023), *Flow Matching*;Liu et al. (2023), *Rectified Flow*
- Song et al. (2023), *Consistency Models*;Sauer et al. (2023), ADD
- Brooks et al. (2023), *InstructPix2Pix*

</div>
</div>

---
layout: center
class: text-center
---

# 兩堂課結束

<div class="mt-6 tone-faint">

作業:為自己的題目選一個 AR 以外的家族,寫出它兩個介面的實作形式與特徵性失效,並指出第一堂哪些方法因此適用、哪些失效;一頁,下次 lab meeting 前交。

</div>

<div class="mt-8 text-sm tone-faint">

自學資源:MIT 6.S184(diffusion.csail.mit.edu)・李宏毅《生成式 AI》系列・Jurafsky & Martin, *Speech and Language Processing* 3rd ed. draft

</div>
