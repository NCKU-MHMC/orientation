---
theme: default
title: 生成模型導論(一)分布的度量與調控
titleTemplate: '%s'
transition: fade
lineNumbers: false
drawings:
  persist: false
class: text-left
---

# 生成模型導論(一)

## 分布的度量與調控

<div class="mt-10 text-slate-500">

新進成員訓練・第一堂(120 分鐘)

</div>

<!--
開場前發放課前自檢 handout(docs/課前自檢.md),六題若有不熟,提醒課後補齊。
本堂全程不涉及任何具體模型結構;所有例子取自語言模型的日常使用經驗。
-->

---

# 兩種常見的輸出

同一段開頭:「我最近壓力很大,一直睡不好。」

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="border border-blue-300 rounded-lg p-4 bg-blue-50/50">

**base model 續寫三次**(示意樣本)

- 「…一直睡不好,而且早上起來頭很脹。上禮拜去看了醫生,醫生說」
- 「…一直睡不好。 回覆 #2:我也是,從去年換工作就開始這樣 回覆 #3:+1」
- 「…一直睡不好。睡眠品質與壓力的關係是雙向的,長期壓力會」

</div>
<div class="border border-amber-300 rounded-lg p-4 bg-amber-50/50">

**aligned model 回覆三次**(示意樣本)

- 「聽起來這段時間辛苦了。以下三個建議:1. 固定作息…」
- 「聽到這些感到很心疼。以下幾個方向:1. 建立睡前儀式…」
- 「這段時間辛苦了。可以試試:1. 規律作息…」

</div>
</div>

<div class="mt-5 text-slate-600">

左側把開頭當文本接下去:替說話者把話說完、跑成論壇串、轉進衛教文章,各次分歧而不對話。右側每次都在回答,而且幾乎同一個模板。這兩種行為有共同的來源。

</div>

<!--
[約 3 分鐘] 讓大家先指認現象:左側是 pretraining 後的續寫行為(接寫、格式漂移、
文體轉向),右側是對齊後的回答行為。先不給任何術語,只確認兩種行為都看過。
情感支持對話組對右側樣本應該特別眼熟。
-->

---
layout: statement
---

# 課程主張

<div class="text-xl leading-relaxed mt-8">

base model 的含糊迴避與 aligned model 的千篇一律,
位於同一條 mode-covering–mode-seeking 光譜的兩端。

</div>

<div class="mt-8 text-slate-500 text-base">

本堂課給出這個主張的論證,並指出這個位置可以在哪些層次上移動。

</div>

<!--
這句話現在還無法驗證,mode-covering 與 mode-seeking 都尚未定義。
課程結束時回到這一頁,屆時兩個詞各自有數學內容,主張成為可檢驗的陳述。
-->

---

# 實驗室題目背後的機率問題

六個題目,六個關於分布的陳述

| 題目 | 背後的機率問題 |
|---|---|
| prompt engineering | 選擇條件變數 $c$,操縱 $p(y \mid c)$ |
| memory agent | 建構條件集合;長脈絡下 $p(\text{task} \mid \text{prompt})$ 被稀釋 |
| 情感支持對話 | 泛泛安慰語與對齊後的多樣性塌縮 |
| false premise 偵測 | $p(y\mid x)$ 永遠良定義,即使 $p(x)\approx 0$ |
| 信心與正確率 | 模型報出的機率能不能被信任(校準) |
| LLM-ASR | $p(\text{text}\mid\text{audio})\propto p(\text{audio}\mid\text{text})\,p(\text{text})$ |

<div class="mt-4 text-slate-500 text-sm">

兩堂課處理的正是:分布如何度量、如何調控、如何建造。

</div>

<!--
[約 2 分鐘] 逐列快速唸過,不展開。此刻只需要看出「六個題目都在談分布」。
-->

---

# 判別式與生成式

一個學標籤的條件分布,一個學資料分布本身

<div class="grid grid-cols-2 gap-6 mt-4">
<div class="border rounded-lg p-4">

**判別式模型**

學 $p(y \mid x)$

輸出空間小而封閉:類別、分數、標籤

</div>
<div class="border rounded-lg p-4">

**生成式模型**

學一個可以抽樣的 $p_\theta(x)$,或條件版 $p_\theta(x \mid c)$

目標:逼近未知的 $p_{\text{data}}$

</div>
</div>

<div class="mt-6">

「逼近」馬上引出兩個問題:

1. $p_\theta$ 與 $p_{\text{data}}$ 的差距要用什麼量來度量?
2. 度量所需要的資訊,雙方拿不拿得出來?

</div>

<!--
判別式的輸出空間可以列舉,生成式的輸出空間(所有句子、所有影像)不能;
量化見 2^65536 一例。
-->

---

# 高維分布無法列表

列舉所有狀態的成本隨維度指數成長

<div class="mt-4">

一張 $256 \times 256$ 的二值影像,狀態總數:

$$2^{256\times256} = 2^{65536} \approx 10^{19728}$$

</div>

<div class="mt-4">

把每個狀態的機率存成一張表,宇宙的原子數($\approx 10^{80}$)遠遠不夠。
詞彙表 5 萬的 100-token 句子同理:$50000^{100}$ 個狀態。

</div>

<div class="mt-6 border-l-4 border-blue-400 pl-4 text-slate-700">

因此分布只能以「函數 + 參數」的形式存在,而能對它做的事,取決於這個函數形式願意回答哪些問題。

</div>

<!--
「願意回答哪些問題」即介面。查表法死於維度,參數法活下來,
但參數法的每一種選擇都犧牲了某些查詢能力。
-->

---

# 介面契約

一個分布,至多提供兩個可呼叫的介面

<div class="mt-6">
<ContractCard />
</div>

<div class="mt-6 text-slate-600 text-center">

本堂所有操作都只透過這兩個介面進行。每引入一個方法,先問它呼叫了哪個介面。

</div>

<!--
[約 2 分鐘] 這是整門課的分析工具;頁面右下角常駐這兩個介面的徽章。
sample() 對應「給我一個樣本」;logprob(x) 對應「這個樣本在此分布下的對數密度是多少」。
-->

---

# 介面盤點

資料側與模型側,各自能回答什麼

| 物件 | `sample()` | `logprob(x)` |
|---|---|---|
| $p_{\text{data}}$(資料) | 有:資料集就是樣本的集合 | **無**:資料不附帶密度 |
| $p_\theta$(模型) | 本堂假設有 | 本堂假設有 |

<div class="mt-6">

資料給的是樣本,沒有密度;凡是需要 $p_{\text{data}}$ 密度的量,都卡在這一格。

</div>

<!--
[約 2 分鐘] reverse KL 不可計算、reward model 的存在理由、H(p) 不可估,
追到底都是這一格。模型側「兩個介面都有」是本堂的工作假設。
-->

---
layout: section
---

# 逼近需要選擇散度

量差距的方式決定犯錯的方式

---

# KL divergence

以 $p$ 為權重的對數比期望

$$\mathrm{KL}(p\,\|\,q)=\int p(x)\,\log\frac{p(x)}{q(x)}\,dx$$

<div class="mt-6">

積分由 $p$ 加權:**只在 $p$ 有質量的地方量測差異**。
$p$ 幾乎為零的區域,無論 $q$ 在那裡放了多少機率,都幾乎不進入積分。

</div>

<div class="mt-4">

把 $p_{\text{data}}$ 和 $p_\theta$ 分別放進兩個位置,得到兩個不同的目標:

- $\mathrm{KL}(p_{\text{data}}\,\|\,p_\theta)$:forward KL
- $\mathrm{KL}(p_\theta\,\|\,p_{\text{data}})$:reverse KL

</div>

<!--
不對稱性不需要證明技巧,直接看權重在誰手上。
-->

---

# Forward KL:覆蓋是義務

權重在 $p_{\text{data}}$ 手上

$$\mathrm{KL}(p_{\text{data}}\,\|\,p_\theta)=\int p_{\text{data}}\log\frac{p_{\text{data}}}{p_\theta}$$

<KlZeros mode="forward" />

<div class="mt-2 text-sm">

凡 $p_{\text{data}}>0$ 而 $p_\theta\to 0$ 之處,懲罰無上界,所以 $p_\theta$ 必須覆蓋 $p_{\text{data}}$ 的整個支撐集:**zero-avoiding / mode-covering**。代價是把機率質量攤到峰與峰之間的低密度區。

</div>

<div class="mt-2 border-l-4 border-blue-400 pl-3 text-sm text-slate-700">

以 forward KL 訓練的語言模型寧可對每種說法都給一點機率,也不敢漏掉任何一種:開場看到的含糊、發散的續寫,就是這個目標下的合理行為。

</div>

<!--
數學一句話:log(p/q) 在 q→0 時發散,而該點的權重 p>0,所以積分爆炸。
「寧可全都要」由此而來。MLE 等價於最小化 forward KL。
-->

---

# Reverse KL:放棄不罰

權重換到 $p_\theta$ 手上

$$\mathrm{KL}(p_\theta\,\|\,p_{\text{data}})=\int p_\theta\log\frac{p_\theta}{p_{\text{data}}}$$

<KlZeros mode="reverse" />

<div class="mt-2 text-sm">

$p_\theta$ 不去的地方不進積分,整個丟掉 $p_{\text{data}}$ 的一個眾數不付任何代價;但 $p_\theta$ 涉足 $p_{\text{data}}$ near-zero 區則重罰。**zero-forcing / mode-seeking**。

</div>

<div class="mt-2 border-l-4 border-amber-400 pl-3 text-sm text-slate-700">

對齊後的模型回答收斂到少數幾種安全模板、多樣性下降,是 mode-seeking 目標的行為特徵。

</div>

<!--
與前一頁同一個積分、只換權重。兩種行為(全都要 vs 挑一個)都不是 bug,
是各自目標函數的最優解。
-->

---
layout: none
---

<DemoFrame src="divergence-2d-interactive.html" title="單峰 q 擬合雙峰 p:三種散度,三種解" :maxH="470" />

<!--
[3 分鐘] 展示順序:
1. forward KL:q 拉寬、跨接兩峰,峰間的空隙也被填上機率。
2. reverse KL:q 鎖定其中一峰,另一峰完全放棄。
3. JSD:介於中間的折衷。
收束句:三個解都是「最優」,差別只在最優的定義。
-->

---

# 每個散度需要哪些介面

同一份介面清單,三種可得性

| 散度 | 需要的介面 | 可得性 |
|---|---|---|
| forward KL $\mathrm{KL}(p_{\text{data}}\|p_\theta)$ | $p_{\text{data}}$.sample + $p_\theta$.logprob | 兩者皆有 |
| reverse KL $\mathrm{KL}(p_\theta\|p_{\text{data}})$ | $p_\theta$.sample + $p_\theta$.logprob + $p_{\text{data}}$.logprob | **末項不可得** |
| JSD(由雙側密度的混合構成) | 兩側 logprob | 資料側必缺 |

<div class="mt-5">

forward KL 是唯一所需介面皆可得的散度:期望用 $p_{\text{data}}$ 的樣本近似,被積函數只呼叫 $p_\theta$.logprob。

reverse KL 與 JSD 缺的是同一個介面:$p_{\text{data}}$ 沒有 logprob。

</div>

<!--
[約 2 分鐘] forward KL 展開後 E_{p_data}[log p_data] 是常數(對 θ 而言),
剩下的 −E_{p_data}[log p_θ] 就是 MLE。
-->

---

# JSD 的定義

以混合分布 $m$ 當共同分母

$$\mathrm{JSD}(p\,\|\,q)=\tfrac12\,\mathrm{KL}\!\left(p\,\Big\|\,m\right)+\tfrac12\,\mathrm{KL}\!\left(q\,\Big\|\,m\right),\qquad m=\tfrac{p+q}{2}$$

<div class="mt-5">

與「對稱化 KL」(Jeffreys divergence)$\mathrm{KL}(p\|q)+\mathrm{KL}(q\|p)$ 不同:
Jeffreys 繼承兩側的無窮大;JSD 的分母是混合 $m$,只要 $p$ 或 $q$ 有質量,$m$ 就有質量,因此

</div>

| 性質 | 說明 |
|---|---|
| 有界 | $0\le \mathrm{JSD}\le\log 2$ |
| 對稱 | 定義即對稱 |
| 度量 | $\sqrt{\mathrm{JSD}}$ 滿足度量公理(Endres & Schindelin, 2003) |

<!--
分母有 m 是關鍵:log(p/m) 最多 log 2,不會爆。
有界的另一面是飽和。
-->

---

# 有界的代價:飽和

支撐集一旦分離,曲線就貼著上界

<JsdSaturate />

<!--
支撐集一旦分離,JSD 貼著 log 2,曲線平掉,對參數的梯度趨近零。
高維空間中兩個分布的支撐集幾乎總是近乎不相交,所以這不是邊角案例。
-->

---

# JSD 的判別器讀法

把散度改寫成一個可訓練的分類問題

以等量樣本訓練一個分類器,判斷樣本來自 $p$ 還是 $q$。最優判別器有閉式解:

$$D^*(x)=\frac{p(x)}{p(x)+q(x)}$$

<div class="mt-3">

把 $D^*$ 代回二元分類目標,其值為 $2\,\mathrm{JSD}(p\,\|\,q)-2\log 2$。
換句話說:**JSD 度量的是最優分類器分辨兩個來源的能力**。兩個分布重疊得越好,最優分類器越接近亂猜,JSD 越小。

</div>

<div class="mt-4 border rounded-lg p-3 bg-slate-50">

同一個式子的等價寫法:

$$D^*(x)=\sigma\!\left(\log\frac{p(x)}{q(x)}\right)\qquad\text{(sigmoid 套在 log ratio 上)}$$

</div>

<!--
推導:對每個 x 最大化 D 的 BCE 目標,逐點微分即得 p/(p+q)。
代回:E_p[log D*] + E_q[log(1−D*)],化簡出 2JSD − 2log2。
資訊論形式:JSD(p‖q) = I(X;Z),Z~Bernoulli(1/2) 指示來源;有興趣的同學課後推。
σ(log p/q) = p/(p+q) 一行驗證:σ(t)=1/(1+e^{−t}),代 t=log(p/q) 得 p/(p+q)。
-->

---

# 選擇散度,就是選擇可接受的錯誤

三種選擇,三種失效型態

| | forward KL | JSD | reverse KL |
|---|---|---|---|
| 行為 | mode-covering | 介於其間,但會飽和 | mode-seeking |
| 對稱 | 否 | 是 | 否 |
| 上界 | 無 | $\log 2$ | 無 |
| 失效型態 | 過度平滑、含糊 | 梯度消失或震盪 | 塌縮、多樣性流失 |

<div class="mt-4">
<SpectrumRows :rows="1" mark="objective" />
</div>

<!--
沒有中立的散度。三個失效型態沒有一個是實作瑕疵,全都寫在目標函數裡。
光譜列 1 的 MLE:最大化資料的 log-likelihood,與最小化 forward KL 等價;
RLHF:以人類偏好訊號微調的對齊方法,本堂④的主題。兩詞在此各給一句即可。
-->

---

# 課後練習

以單一高斯擬合雙峰混合,三種散度各解一次

以單一高斯 $q=\mathcal N(\mu,\sigma^2)$ 擬合 1D 雙峰混合
$p=\tfrac12\mathcal N(-2,0.6^2)+\tfrac12\mathcal N(2,0.6^2)$,
分別最小化三種散度,數值解出 $\mu,\sigma$。

<div class="mt-4">
<DivergenceFit />
</div>

<div class="mt-3 text-sm text-slate-500">

預期結果如上圖。動手做一遍,三種行為就不再只是形容詞。

</div>

<!--
提示:一維數值積分用梯形法即可;reverse KL 有兩個局部極小(兩個峰各一)。
-->

---
layout: section
---

# 分布固定之後

目標分布往往不是模型分布本身:更符合條件、更銳利、更安全、更多樣

---

# 引導生成的統一形式

base 項、比值項、係數,三個欄位

<div class="mt-10">
<GuidanceForm />
</div>

<!--
[約 3 分鐘] 這條式子是本節的全部;常見方法逐一填進三個欄位即可。
再正規化:log 空間相加後,機率總和不再是 1,除以配分函數(逐 token 情形就是 softmax)。
-->

---

# 常見方法都是這條式子(上)

解碼期常見的四種手法

| 方法 | base | 比值項 | 係數 | 需要的介面 |
|---|---|---|---|---|
| temperature | $\log p$ | 無 | $1/T$ | logprob(逐 token) |
| CFG for LLM<br><span class="text-xs text-slate-400">Sanchez et al., 2023</span> | $\log p(x\mid c)$ | $\log p(x\mid c)-\log p(x)$ | $w$ | 兩種條件下的 logprob |
| contrastive decoding<br><span class="text-xs text-slate-400">Li et al., 2023</span> | $\log p_{\text{strong}}$ | $\log p_{\text{strong}}-\log p_{\text{weak}}$ | $\lambda$ | 兩個模型的 logprob |
| DoLa<br><span class="text-xs text-slate-400">Chuang et al., 2024</span> | $\log p_{\text{final}}$ | 末層與淺層 logits 之差 | $\lambda$ | 中間層 logits |

<!--
temperature 可以視為比值項退化(把 log p 自己當比值項:(1/T)·log p = log p + (1/T − 1)·log p)。
CFG for LLM:同一模型跑有條件與無條件兩次,差值放大條件的作用。
contrastive decoding:大模型減小模型,削掉「小模型也會犯」的通病。
DoLa:同一模型內部,末層減淺層。
-->

---

# 常見方法都是這條式子(下)

劣化模型、偏好對齊、分布銳化

| 方法 | base | 比值項 | 係數 | 需要的介面 |
|---|---|---|---|---|
| Autoguidance<br><span class="text-xs text-slate-400">Karras et al., 2024</span> | $\log p_\theta$ | $\log p_\theta-\log p_\phi$(劣化版模型) | $w$ | 兩個模型的 logprob |
| RLHF 最優解<br><span class="text-xs text-slate-400">Ouyang et al., 2022;推導見 Rafailov et al., 2023</span> | $\log \pi_{\text{ref}}$ | $r(y)$ | $1/\beta$ | ref 的 logprob + reward |
| DDO 最優解<br><span class="text-xs text-slate-400">Zheng et al., 2025</span> | $\log p_{\text{ref}}$ | $\log p_{\text{data}}-\log p_{\text{ref}}$ | $1/\beta$ | 兩者的 logprob |

<div class="mt-4 text-sm">

top-k / top-p 是同一操作的硬截斷版:不連續,但同樣在移除尾部機率質量。

</div>

<!--
Autoguidance:用刻意劣化的模型當 p_B,把「劣化方向」反向放大,不需要條件標註。
RLHF 列:最優解 π* ∝ π_ref·exp(r/β),取 log 即符合統一引導式。
-->

---
layout: none
---

<DemoFrame src="guidance-playground.html" title="同一根滑桿:temperature、CFG、contrastive decoding" :maxH="470" />

<!--
[3 分鐘] 展示腳本:
1. temperature 情境,拉 w:分布變尖再變平。
2. 切 CFG 情境,同一根桿子做條件強化。
3. 切 prompt engineering 情境:桿子變灰。畫面標註「此手法不在係數的位置上」。
   把這個灰桿留在螢幕上,討論頁還會用到。
4. contrastive 情境(15 秒):大模型減小模型,削掉共同的通病。
5. 對數空間檢視:w 變動時折線嚴格線性位移,「log 空間線性組合」眼見為憑。
-->

---

# 三個結論

座標、時機、適用範圍

1. **係數是光譜上的座標。** $w$ 或 $1/\beta$ 越大,比值項的作用越強;這些控制參數佔的是同一個位置,移動的方向則由各自的比值項決定。

2. **推論期做與訓練期做,差別只在時機。** 同一條式子可以在解碼時套用,也可以內化進權重(第④節)。

3. **適用範圍由介面決定。** 上兩頁的每一列幾乎都要呼叫 logprob;一個不提供 logprob 的模型或黑箱 API,整個框架對它失效。

<div class="mt-5">
<SpectrumRows :rows="2" mark="decoding" />
</div>

<!--
結論 1 是本節的核心:temperature、top-p、CFG 係數、β,全是同一個座標的旋鈕。
光譜第二列(解碼設定)由此補上。
-->

---

# 討論:prompt 在式子的哪個位置

三個欄位,提示詞只動得了一個

$$\log p_{\text{guided}}=\underbrace{\log p_{\text{base}}}_{\text{prompt 置換的是這裡}}+\;w\,(\log p_A-\log p_B)$$

<div class="mt-5">

prompt engineering 改變條件 $c$,等於整個換掉 base 項;係數 $w$ 與比值項完全不動。

因此:輸出太單調、太發散、過度銳化這類**係數層次的問題,無法靠改寫 prompt 解決**,提示詞不在那個位置上。

</div>

<div class="mt-5 text-slate-600">

討論:各自的題目裡,有沒有一個「調了很久 prompt 都沒用」的問題,其實住在係數的位置?

</div>

<!--
[3 分鐘討論] demo 的灰桿就是這一頁的可視化。
常見案例:要求「多給幾種不同建議」收效有限,因為多樣性由熵(係數層)控制,
prompt 只能挪動 base 分布。
-->

---
layout: center
class: text-center
---

# 休息 10 分鐘

<!--
時間配置:休息的 10 分鐘由①吸收(36 → 30:判別式對照頁與高維計數頁各講快一點,
共省 6 分),其餘由③④各緊 2 分,總長維持兩小時。
-->

---
layout: section
---

# 推論期的四層介入

從條件到聚合,每一層呼叫的介面不同

---

# 四層總覽

四個介入點,各自呼叫不同的介面

<LayerStack />

<div class="mt-4 text-sm text-slate-600">

第 1 層與第 4 層只需要 sample:任何黑箱 API 都能做。這正是 prompt engineering 與多數投票類方法對任何服務都可行的原因。

</div>

<!--
[約 2 分鐘] 這張表是本節的地圖。
介面欄延續上一節的分析習慣:看到新方法,先問它呼叫什麼。
第 2、3 層的分界:第 2 層是與內容無關的全域重塑與截斷(整條分布一起變形),
第 3 層是逐 token、內容相依的修改(哪些 token 動、動多少,取決於 token 是誰)。
-->

---

# 第 1 層.改變條件:prompt 即 conditioning

示範收緊的是任務的後驗

$$p(y\mid \text{prompt})=\int p(y\mid \text{task})\;p(\text{task}\mid \text{prompt})\;d\,\text{task}$$

<div class="mt-4">

In-context learning 可讀成隱式貝氏推論(Xie et al., 2022):prompt 裡的示範不改參數,而是收緊模型對「現在在做哪個 task」的後驗。

</div>

<div class="mt-4 border-l-4 border-blue-400 pl-4">

memory agent 的機率語意:記憶系統的工作是**挑選哪些證據進入後驗**;存放只是實作手段。

</div>

<!--
積分式為簡寫,假設 task 給定後 y 與 prompt 條件獨立;
Xie et al. 的原式在積分內保留 p(y|task, prompt)。
Xie et al. (ICLR 2022) 把 ICL 形式化為對潛在 task 變數的貝氏推斷:
prompt 是觀測,回答是對 task 的邊際化。memory agent 組把這條積分式當設計語言:
每一則被取回的記憶都是進入條件的證據,取回策略即後驗塑形。
-->

---

# RAG 與 fine-tuning:兩種安裝條件的方式

條件留在脈絡裡,或攤銷進權重

| | RAG | fine-tuning |
|---|---|---|
| 機率意義 | 顯式條件:$p(y\mid x, \text{檢索到的 } d)$ | 條件攤銷進權重:$p_{\theta'}(y\mid x)$ |
| 失效型態 | 檢索錯,條件就錯;無關文件稀釋後驗 | 分布外遺忘;更新成本高、不可逆 |

<div class="mt-5">

「無關文件稀釋後驗」有量化證據:關鍵資訊放在長脈絡中段時,取用正確率明顯下降(lost-in-the-middle,Liu et al., 2024)。更多條件不等於更好的條件,$p(\text{task}\mid\text{prompt})$ 會被攤平。

</div>

<!--
Liu et al. (2024):同一份文件集,答案所在文件的位置從頭移到中間,
多個模型的正確率呈 U 形下降。對 memory agent 的直接教訓:取回內容的排序本身是條件設計。
-->

---

# 第 2 層.改變抽樣

同一組 logits,兩種重塑方式

<TempTopP />

<div class="mt-3">

temperature 把 logits 除以 $T$,直接調整分布的熵;top-p 截斷尾部後再正規化(Holtzman et al., 2020)。

情感支持系統的抽樣設定是一個設計決策:低 $T$ 安全而單調,高 $T$ 多樣而風險高。預設值不是答案,兩種錯誤的相對代價才是。

</div>

<div class="mt-3">
<SpectrumRows :rows="2" mark="decoding" />
</div>

<!--
第二列光譜(解碼設定)在上一節已畫出;temperature 與 top-p 都是那一列上的移動。
情感支持場景:過於模板化的回應讓使用者感到敷衍(低 T 端),
過於自由的回應可能出現不當建議(高 T 端)。取捨必須明文寫進系統設計。
-->

---

# 第 3 層.改變 logits

在 softmax 之前修改分數

- **constrained decoding / grammar**:在合法 token 子集上重新正規化。要求結構化輸出(JSON、SQL)時,這比在 prompt 裡以指示要求格式可靠:非法 token 的機率被精確歸零,軟性指示做不到這一點。

- **logit bias**:對特定 token 加減常數,即統一引導式中一個手寫的比值項。

- **contrastive decoding、DoLa、CFG for LLM**:上一節表中的三列,安裝位置都在這一層,逐 token 修改 logits 後再 softmax。

<!--
constrained decoding 的機率語意:條件在「輸出屬於文法 L」這個事件上,
p(y|y∈L) 的逐 token 實作。與 prompt 要求格式的差別:前者是精確條件化,後者是軟提示。
-->

---

# 第 4 層.改變樣本的聚合

抽多個樣本,重新估計答案的分布

- **best-of-n**:抽 $n$ 個,用外部評分挑一個
- **self-consistency**:抽多條推理路徑,對最終答案投票
- **MBR**(minimum Bayes risk):選「與其他樣本平均距離最近」的輸出
- **reranking**:以另一個模型重排候選

<div class="mt-4">

self-consistency 的機率語意即 Monte Carlo 邊際化:
$p(a\mid q)=\sum_r p(a\mid r,q)\,p(r\mid q)$,對推理路徑 $r$ 積分(Wang et al., 2023)。

</div>

<!--
這一層只需要 sample(logprob 可選,用於加權)。
與第 1 層合起來看:黑箱 API 能做的事其實不少,第 1 層 + 第 4 層都不需要 logprob。
-->

---

# LLM-ASR:一條 log 空間的線性組合

聲學 likelihood 乘上語言模型 prior

$$p(\text{text}\mid\text{audio})\;\propto\;p(\text{audio}\mid\text{text})\;p(\text{text})$$

<div class="mt-4">

雜訊通道模型:聲學模型提供 likelihood,語言模型提供 prior,相乘即後驗。取 log 之後,這就是統一引導式的形狀。

</div>

| 作法 | 所在層 |
|---|---|
| n-best rescoring、LLM 錯誤更正 | 第 4 層(聚合) |
| speech encoder 接入 LLM | 第 1 層(聲學表徵作為條件) |
| 古典 ASR 的 LM weight 與 insertion penalty | 手工調整的係數 $w$ |

<!--
古典 ASR 解碼器裡 log p_acoustic + λ·log p_LM + 字數懲罰,λ 就是手調係數,
比 guidance 這個詞早了幾十年。LLM-ASR 組的兩條技術路線正好落在第 1、4 兩層。
-->

---
layout: none
---

<DemoFrame src="asr-noisy-channel.html" title="雜訊通道:聲學分數與語言模型分數的線性組合" :maxH="470" />

<!--
[2 分鐘] 拉 LM weight,看 n-best 列表重排:權重過低出現同音錯字,
權重過高輸出被語言模型「腦補」成通順但錯誤的句子。同一個係數,同一種取捨。
畫面底部那一行就是②節的統一引導式:聲學分數是 base 項,LM 分數是比值項,λ 是係數 w。
收束本節:日常使用的方法多在第 1 層,只呼叫 sample;第 2 到 4 層動的是
同一條分布的其他位置,所需介面表上都標好了。
-->

---
layout: section
---

# 權重層的介入

SFT、RLHF、DPO、DDO:把移動寫進參數

---

# SFT:在新資料上重做 MLE

換資料,不換目標函數

$$\max_\theta\;\mathbb{E}_{(x,y)\sim \mathcal D_{\text{SFT}}}\big[\log \pi_\theta(y\mid x)\big]$$

<div class="mt-5">

目標函數與預訓練相同,仍是 forward KL,只是換了資料分布。
因此 SFT 後的模型仍在 mode-covering 端:格式與語氣被塑形,含糊與過度覆蓋的傾向不變。

</div>

<div class="mt-5">
<SpectrumRows :rows="3" mark="weights" />
</div>

<!--
光譜第三列(權重微調)從這裡開始畫。SFT 佔住左端。
-->

---

# RLHF 的目標與閉式解

reward 的期望,減去與參考模型的 KL

$$\max_\pi\;\mathbb{E}_{y\sim\pi}\big[r(y)\big]-\beta\,\mathrm{KL}\big(\pi\,\|\,\pi_{\text{ref}}\big)$$

<div class="mt-3">

這個目標有閉式最優解(推導見 Rafailov et al., 2023 附錄;把解代回目標即可驗證):

$$\pi^*(y)\;\propto\;\pi_{\text{ref}}(y)\,\exp\!\big(r(y)/\beta\big)$$

取 log 即第②節表中的 RLHF 列。而整個最佳化問題等價於

$$\min_\pi\;\mathrm{KL}\big(\pi\,\|\,\pi^*\big)$$

<div class="text-slate-600">

KL 的第一個位置放的是 $\pi$,這正是 reverse KL:對齊訓練住在 mode-seeking 端。

</div>

</div>

<!--
等價性推導:把 π* 代回,E_π[r] − β·KL(π‖π_ref) = −β·KL(π‖π*) + 常數。
所以 RLHF 是「以 π* 為目標的 reverse KL 投影」。
mode-seeking 的一切性質(挑模板、丟多樣性)自動繼承。
-->

---

# DPO:把 reward 消掉

模型自己的 log ratio 就是隱式 reward

閉式解可以反過來解出 reward:

$$r(y)=\beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)}+\text{const}$$

<div class="mt-3">

把這個表達式代入偏好資料的 Bradley–Terry 損失,reward model 從式子裡消失(Rafailov et al., 2023):

$$\mathcal L_{\text{DPO}}=-\mathbb{E}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)\right]$$

</div>

<div class="mt-4">

整條損失只呼叫 $\pi_\theta$ 與 $\pi_{\text{ref}}$ 的 logprob 介面;最佳化的仍是同一個 reverse KL 目標。

</div>

<!--
三行推導:π* ∝ π_ref·exp(r/β) ⇒ r = β log(π*/π_ref) + const;
Bradley–Terry:P(y_w ≻ y_l) = σ(r(y_w) − r(y_l));const 相消。
「隱式 reward = β·log ratio」與 DDO 的隱式判別器是同一種參數化。
-->

---

# reward model 補的是哪一格

reverse KL 缺的那一個介面

| 散度 | 缺的介面 | 補上它的東西 |
|---|---|---|
| reverse KL | 目標分布的 logprob | **reward model** |

<div class="mt-5">

需求表裡標「不可得」的那一格,由 reward model 填補:它從人類偏好標註學出純量分數 $r(y)$,而 $r/\beta$ 正好充當目標分布相對 $\pi_{\text{ref}}$ 的 log 密度比。

</div>

<div class="mt-4">

要注意目標分布在此換了對象:對齊瞄準的已非 $p_{\text{data}}$,而是偏好加權後的 $\pi^*\propto\pi_{\text{ref}}\,e^{r/\beta}$。缺 logprob 的困境不變,代理的品質決定對齊的品質:reward model 學壞了,$\pi^*$ 就指向錯的地方(reward hacking)。

</div>

<!--
這一格在①的介面需求表中標為不可得;reverse KL 一族的所有方法都得先造一個代理。
代理不只 reward model 一種,能量函數、分類器都可以。
若有人問「兩端的散度連引數都不同,光譜還成立嗎」:成立,光譜刻畫的是散度的
「方向」——由誰加權、對誰 zero-forcing——而非固定的分布對;mode-seeking 是方向的
性質,對任何目標分布都導致收斂到該目標的少數眾數,行為特徵(模板化、多樣性下降)
不因目標分布換人而改變。
-->

---

# β 是唯一的煞車

單調來自目標函數的形式

「對齊讓模型更安全,但更單調」有其數學形式,並非經驗巧合:

<div class="mt-3">

- reverse KL 目標的最優解**只在 reward 高的區域放質量**;多樣性沒有出現在目標函數裡
- 唯一抑制塌縮的是 $\beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$,而它約束的是**與參考模型的距離**,不是多樣性本身
- 實測:RLHF 後輸出多樣性系統性下降(Kirk et al., 2024)

</div>

<div class="mt-4 border-l-4 border-amber-400 pl-4">

情感支持系統的安全與多樣共用同一個 $\beta$:調鬆帶來變化也帶來風險,調緊帶來安全也帶來模板。這是一個旋鈕,兩個願望。

</div>

<!--
Kirk et al. (2024):RLHF 模型在 summarization 等任務上 per-input 與 cross-input
多樣性皆下降,泛化能力上升。「多樣性與 β 鬆緊同向」是由目標函數形式得到的
理論預期,該文並未系統性掃 β,課堂上不要說成實測結論。
開場那個「千篇一律」的示意樣本,機制在此。
-->

---

# 逐點計分器的結構極限

reward model 與 judge 共有的盲區

reward model 對**單一樣本**給分:$r(y)\in\mathbb R$。

<div class="mt-3">

「這個分布太窄」是**分布層級**的性質,單點分數的介面裡沒有承載它的欄位:
每個模板回答逐點看都得高分,計分器對「全部都是同一種回答」無從抗議。

</div>

<div class="mt-4">

LLM-as-judge 同樣逐點評審,同樣的極限;把 judge 換強並不補上這個欄位。

</div>

<div class="mt-4 text-slate-600">

因此塌縮的防線只剩 β 項對參考模型距離的約束。

</div>

<!--
形式化:任何 r: Y→R 的期望 E_π[r] 對 π 的重複度不敏感,
除非 r 本身以分布為輸入(而它不是)。
-->

---

# 兩件已經在手上的事實

一個判別器閉式解,一條引導式

<div class="mt-4 grid grid-cols-1 gap-5">

<div class="border rounded-lg p-4">

**事實一(第①節)** 分辨兩個分布的最優判別器是

$$d^*(x)=\sigma\!\left(\log\frac{p_{\text{data}}(x)}{q(x)}\right)$$

</div>

<div class="border rounded-lg p-4">

**事實二(第②節)** 統一引導式方法表的末列:

$$\log p_{\text{guided}}=\log p_{\text{ref}}+\tfrac1\beta\big(\log p_{\text{data}}-\log p_{\text{ref}}\big)$$

</div>

</div>

<div class="mt-5 text-slate-600">

接下來的方法只用這兩件事,不引入任何新原理。

</div>

<!--
[約 1 分鐘] 兩個事實直接陳述,學員此刻應該都認得。
DDO 的全部原料就這兩行。
-->

---

# DDO:用自己的 logprob 當判別器

判別器不必是另一個網路

任何提供 logprob 的分布,都可以直接**宣告**自己是判別器(Zheng et al., 2025):

$$d_\theta(x)\;=\;\sigma\!\left(\beta\,\log\frac{p_\theta(x)}{p_{\text{ref}}(x)}\right)$$

<div class="mt-3">

- $\beta$ 是必要的縮放:log ratio 的逐維差異隨維度累積、量級可達數十至數百,直接進 sigmoid 會使梯度消失
- 以標準 BCE 訓練:真樣本標 1,參考模型樣本標 0
- BCE 的最優判別器是 $\sigma(\log(p_{\text{data}}/p_{\text{ref}}))$;對照兩式,$\beta=1$ 時最優解就是 $p_\theta=p_{\text{data}}$,一般 $\beta$ 給出 $p_\theta^*\propto p_{\text{ref}}^{\,1-1/\beta}p_{\text{data}}^{\,1/\beta}$

</div>

<div class="mt-4 border-l-4 border-emerald-500 pl-4 text-slate-700">

DPO 把 reward 參數化成 log ratio;DDO 把**判別器**參數化成 log ratio。同一著棋,下在不同的棋盤。

</div>

<!--
與 DPO 對照:DPO 的隱式 reward = β log(π_θ/π_ref),DDO 的隱式判別器 = σ(β log(p_θ/p_ref)),
參數化手法完全平行。資料需求不同:DPO 要偏好對,DDO 只要原始訓練資料 + ref 樣本。
-->

---

# DDO 的機制

兩種樣本,一個 BCE 損失

<DdoMechanism />

<div class="mt-2 text-sm text-slate-600 text-center">

無需額外判別器網路、無需交替訓練、無需對抽樣過程反向傳播。

</div>

<!--
三個「無需」對照的是傳統對抗訓練的三大負擔。
虛線:一輪訓練後把 p_θ 存成新的 p_ref,再來一輪(self-play);
同型的 self-play 微調另見 SPIN(Chen et al., 2024)。
-->

---

# 梯度在做什麼

正負號由 $p_\theta$ 與 $p_{\text{data}}$ 的差決定

$$\nabla_\theta L\;=\;\int \big(1-d_\theta(x)\big)\,\big(p_\theta(x)-p_{\text{data}}(x)\big)\,\nabla_\theta\log p_\theta(x)\,dx\qquad(\text{取 }\beta=1)$$

<div class="mt-4">

| 區域 | 符號 | 作用 |
|---|---|---|
| $p_\theta<p_{\text{data}}$(蓋不夠) | 負 | **抬升**該處密度 |
| $p_\theta>p_{\text{data}}$(蓋過頭) | 正 | **壓低**該處密度 |

</div>

<div class="mt-4">

MLE 的梯度只有第一種作用:在資料點上抬密度,對模型自己多出來的機率質量沒有任何移除機制。DDO 的損失同時含 $\mathbb E_{p_{\text{data}}}[\cdot]$(抬)與 $\mathbb E_{p_{\text{ref}}}[\cdot]$(壓),「遠離自己的樣本」的精確意義在此。

</div>

<!--
直觀:判別器訓練把「模型樣本」當負例,等於在自己過度自信的區域施加向下的力。
(1 − d_θ) 是權重:已經分得很開的區域力道小,分不開的區域力道大。
推導步驟(β=1;追問時用):BCE 對 θ 微分 → σ 的導數帶出 d_θ(1−d_θ) →
E_{p_data} 與 E_{p_ref} 兩項合併,利用 p_ref·d_θ = (1−d_θ)·p_θ(β=1 才成立)
得 (p_θ − p_data) 加權形式;一般 β 的梯度是 β·[p_ref·d_θ − p_data(1−d_θ)]∇log p_θ,
完整推導見 Zheng et al. (2025) 附錄。
-->

---
layout: none
---

<DemoFrame src="mle-vs-ddo-gradient.html" title="MLE 與 DDO 的梯度場" :maxH="470" />

<!--
[3 分鐘] 左:MLE 梯度場,只有指向資料點的吸力;模型多餘的質量原地不動。
右:DDO 梯度場,資料區吸、過剩區推。
讓學員找「MLE 動不了、DDO 在動」的區域,就是 p_θ > p_data 的地方。
量化證據看畫面上的兩個讀數:右側 reverse KL 讀數下降明顯快於左側,
forward KL 則兩側皆降——「同時抓兩端」以數字兌現。
-->

---

# DDO 在光譜上的位置

抬升與壓低分屬光譜兩端

<div class="mt-4">
<SpectrumRows :rows="3" />
</div>

<div class="mt-6">

抬升項做的是 forward KL 的事(把質量搬向資料),壓低項做的是 reverse KL 的事(從過剩區撤出質量):**DDO 同時作用於光譜兩端**,而多數方法只能佔一端。

</div>

<!--
第三列至此完整:左端 SFT,右端 DPO/DDO(小 β),而 DDO 特殊在兩端同時施力。
-->

---

# DPO、DDO 與統一引導式

同一種參數化,兩個用途

| | DPO | DDO |
|---|---|---|
| 隱式參數化 | reward $=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ | 判別器 $=\sigma\!\big(\beta\log\frac{p_\theta}{p_{\text{ref}}}\big)$ |
| 目標 | 偏好學習 | 分布對齊 |
| 資料 | 成對人類標註 | 原始訓練資料,無需配對 |

<div class="mt-4">

DDO 的最優解(Zheng et al., 2025):

$$p_\theta^*\;\propto\;p_{\text{ref}}^{\,1-1/\beta}\;p_{\text{data}}^{\,1/\beta}$$

正是統一引導式方法表末列的指數形式。guidance 在推論期執行這次銳化,DDO 把同一個操作寫進權重,推論時不再需要第二次前向。

</div>

<!--
取 log 驗證:log p* = log p_ref + (1/β)(log p_data − log p_ref) + const,
與②表末列逐項相同。「訓練期與推論期只差時機」在此兌現。
-->

---

# 訓練到頂之後

MLE 收斂之後,指標仍有空間

Zheng et al. (2025) 微調多個提供 logprob 的生成模型,兩則觀察與本堂論證直接相關:

<div class="mt-4">

1. **繼續用 MLE 訓練,指標不動甚至劣化。** forward KL 目標已到達其可達的極限;卡住的原因在目標函數,調參數解不開。

2. **有些模型此前靠 top-k / top-p 撐指標。** 截斷實際上降低了有效溫度,把分布的缺陷藏起來而非修好;換上 DDO 後,不加任何截斷的原始分布品質即提升。

</div>

<!--
兩則觀察分別對應:①的「失效型態寫在目標函數裡」與②③的「截斷是硬移除尾部質量」。
出處均為 Zheng et al. (2025)。
-->

---

# 兩種輸出,同一條光譜

開場的兩欄,各自的位置

<div class="mt-2">
<SpectrumRows :rows="3" />
</div>

<div class="mt-6">

含糊的續寫出自 forward KL 訓練,同質的回答出自 reverse KL 對齊:**同一條光譜的兩端**,而位置可以在三個層次上移動:

- 訓練目標(選哪個散度)
- 解碼設定(temperature、top-p、guidance 係數)
- 權重微調(SFT、DPO、DDO 的 $\beta$)

</div>

<!--
開場的兩欄示意樣本至此有了完整的解釋鏈:現象 → 散度 → 光譜位置 → 三個可動層次。
-->

---

# 本堂的假設,與兩個未解的問題

兩個介面齊備,是全部論證的前提

**本堂自始至終的假設**:$\pi_{\text{ref}}$(以及 $p_\theta$)同時提供 `sample()` 與 `logprob(x)`。

<div class="mt-5">

兩個此刻沒有答案的問題:

1. reverse KL 一族缺 $p_{\text{data}}$.logprob,各方法拿什麼**代理**去補?本堂只見過 reward model 一種。
2. 一個**完全不提供 logprob** 的模型,要拿什麼訓練?本堂的每一種損失對它都寫不出來。

</div>

<!--
兩個問題留作懸念即可,課堂上不展開、不解答。
-->

---

# 作業

選一個自己正在做的題目,寫下三件事

1. **機率形式**:$x$、$y$、$c$ 各是什麼?在學的分布是哪一個?
2. **光譜位置**:目前的方法把分布往哪端推?被什麼參數控制?
3. **介面清單**:用到的每個方法,各呼叫了 `sample()` 還是 `logprob(x)`?

<div class="mt-6 text-slate-500">

下次課前交一頁。第三題若有方法兩個介面都不需要,重看第③節的四層總表。
完整書目見課程 repo 的文獻頁。

</div>

<!--
驗收重點在第三題:介面分析是否已成為反射。
批改時比對第二堂結尾的定位表。
-->
