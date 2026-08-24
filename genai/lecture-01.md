---
theme: default
title: 生成模型導論(一) 分布的度量與調控
titleTemplate: '%s'
transition: fade
lineNumbers: false
drawings:
  persist: false
download: true
exportFilename: genai-lecture-01
fonts:
  sans: 'Source Sans 3,Noto Sans TC'
  serif: 'Source Serif 4,Noto Serif TC'
  mono: 'IBM Plex Mono'
  weights: '400,500,600,700'
class: text-left
---

<div class="rule-accent" />

# 生成模型導論(一)

## 分布的度量與調控

<!--
聽眾是剛進實驗室的碩一新生,修過深度學習導論,還沒做過研究,
大約一年後才會定自己的題目。全程不預設任何人手上已經有題目,
所有例子取自平常使用 LLM 的經驗。

課前發放課前自檢 handout(docs/課前自檢.md),六題若有不熟,提醒課後補齊。

開場:這兩堂課從頭到尾只談一件事,機率分布。今天這一堂處理兩個問題,
分布之間的差距怎麼量,量完之後怎麼把分布推到想要的位置。
全程不會出現任何具體的模型結構,不談 Transformer,也不談 diffusion 的網路長什麼樣;
所有例子都取自平常使用語言模型的經驗。
-->

---

# 兩種常見的輸出

同一段開頭:「我最近壓力很大，一直睡不好。」

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="card card-data">

**base model 續寫三次**(示意樣本)

- 「...一直睡不好，而且早上起來頭很脹。上禮拜去看了醫生，醫生說...」
- 「...一直睡不好。 回覆 #2：我也是，從去年換工作就開始這樣。 回覆 #3：+1」
- 「...一直睡不好。睡眠品質與壓力的關係是雙向的，長期壓力會...」

</div>
<div class="card card-model">

**aligned model 回覆三次**(示意樣本)

- 「聽起來這段時間辛苦了。以下三個建議：1. 固定作息...」
- 「聽到這些感到很心疼。以下幾個方向：1. 建立睡前儀式...」
- 「這段時間辛苦了。可以試試：1. 規律作息...」

</div>
</div>

<div class="mt-5 tone-muted">

左側把開頭當文本接下去：替說話者把話說完、跑成論壇串、轉進衛教文章，各次分歧而不對話。右側每次都在回答，而且幾乎同一個模板。這兩種行為有共同的來源。

</div>

<!--
[約 3 分鐘] 先看現象,不給任何術語。

同一段開頭丟給兩種模型。左邊是 base model,只做過 pretraining:
它把這段開頭當成一份文本繼續寫下去。第一次替說話的人把話說完,
第二次跑成論壇的回覆串,第三次轉進衛教文章的語氣。
三次的走向互相分歧,而且沒有一次在跟人對話。
右邊是對齊過的模型:三次都在回答,都先同理一句,再給一份編號清單,連句型都幾乎一樣。

右邊那種回覆,平常用助理型的 LLM 應該都看過;
實驗室的情感支持對話題目處理的就是它。左邊那種續寫比較少見,
因為公開服務多半已經對齊過,所以先在這裡把樣本擺出來。
兩種行為看起來相反,但它們有共同的來源。
-->

---
layout: statement
---

# 課程重點

<div class="text-xl leading-relaxed mt-8">

<span class="underset">base model 的含糊迴避<div class="underset-lg">---mode-covering---</div></span> 與 <span class="underset">aligned model 的千篇一律<div class="underset-lg">---mode-seeking---</div></span><br/>
是同一個量的兩個極端：模型分布的覆蓋程度，也就是機率質量攤在多少種輸出上。

</div>

<div class="mt-8 tone-faint text-base">

本堂課給出這個主張的論證，並指出覆蓋程度可以在哪些層次上調整。

</div>

<!--
這一頁的主張現在還無法驗證,mode-covering 與 mode-seeking 兩個詞都還沒有定義。
先把話放在這裡:左邊那種什麼都寫一點、什麼都不聚焦的行為,
和右邊那種永遠同一個模板的行為,是同一個量的兩個極端,
這個量就是模型把機率質量攤在多少種輸出上。

今天要做兩件事:給出這個主張的論證,並指出覆蓋程度可以在哪些層次上被調整。
等到兩個詞各自有了數學內容,這句話就成為可以檢驗的陳述。
-->

---

# 實驗室題目背後的機率問題

實驗室相關的六個題目，六個關於分布的陳述

| 題目 | 背後的機率問題 |
|---|---|
| prompt engineering | 選擇條件變數 $c$，操縱 $p(y \mid c)$ |
| memory agent | 建構條件集合；長脈絡下 $p(\text{task} \mid \text{context})$ 被稀釋 |
| 情感支持對話 | 泛泛安慰語與對齊後的多樣性塌縮 |
| false assumptions 偵測 | $p(y\mid x)$ 永遠良定義，即使 $p(x)\approx 0$ |
| 信心與正確率 | 模型報出的機率能不能被信任(校準) |
| ASR | 嘗試解構出 $p(\text{feature}\mid\text{audio})$ 來降低 ASR 建模難度：<br/>$p(\text{text}\mid\text{audio} )=\int_{\text{feature}}p(\text{text}\mid\text{feature},\text{audio})p(\text{feature}\mid\text{audio})$；|

<div class="mt-4 tone-faint text-sm">

兩堂課處理的正是：分布如何度量、如何調控、如何建造。

</div>

<!--
[約 2 分鐘] 逐列快速唸過,不展開。

這六個是實驗室現在在跑的題目,今天不必挑,也不必聽懂細節。
新生大約一年後才定自己的研究題目,現在只要知道這些題目共用同一組機率語言。

prompt engineering 做的是選條件變數 c,操縱 p(y|c)。
memory agent 是在建構條件的集合,而長脈絡下 p(task|prompt) 會被稀釋。
情感支持對話碰到的是泛泛的安慰語,以及對齊之後的多樣性塌縮。
false premise 偵測要注意 p(y|x) 永遠良定義,即使 x 本身的機率趨近於零。
信心與正確率問的是模型自己報出來的機率能不能信,也就是校準。
LLM-ASR 直接建模 p(text|audio),外掛的語言模型是在 log 空間加權相加。

此刻只需要看出一件事:六個題目都在談分布。今天處理其中兩件,度量與調控。
-->

---

# 判別式與生成式

一個學標籤的條件分布，一個學資料分布本身

<div class="grid grid-cols-2 gap-6 mt-4">
<div class="card">

**判別式模型**

學 $p(y \mid x)$

輸出空間小而封閉：類別、分數、標籤

</div>
<div class="card">

**生成式模型**

學一個可以抽樣的 $p_\theta(x)$,或條件版 $p_\theta(x \mid c)$

目標：逼近未知的 $p_{\text{data}}$

</div>
</div>

<div class="mt-6">

「逼近」馬上引出兩個問題：

1. $p_\theta$ 與 $p_{\text{data}}$ 的差距要用什麼量來度量？
2. 度量所需要的資訊,雙方拿不拿得出來？

</div>

<!--
判別式模型學 p(y|x),輸出空間小而封閉:類別、分數、標籤,可以整個列舉出來。
生成式模型學一個可以抽樣的 p_θ(x),或條件版 p_θ(x|c),目標是逼近未知的 p_data。
兩者的差別不在網路,在輸出空間能不能列舉:所有句子、所有影像的集合列不出來,
下一頁把這個「列不出來」量化。

「逼近」兩個字馬上帶出兩個問題:差距要用什麼量來度量;
以及度量需要的資訊,p_data 和 p_θ 雙方拿不拿得出來。前半堂就在回答這兩個問題。
-->

---

# 高維分布無法列表

列舉所有狀態的成本隨維度指數成長

<div class="mt-4">

一張 $256 \times 256$ 的二值影像，狀態總數：

$$2^{256\times256} = 2^{65536} \approx 10^{19728}$$

</div>

<div class="mt-4">

把每個狀態的機率存成一張表：宇宙的原子數($\approx 10^{80}$)遠遠不夠。
詞彙表 5 萬的 100-token 句子同理：$50000^{100}$ 個狀態。

</div>

<div class="mt-6 aside aside-data tone-muted">

因此分布只能以「函數 + 參數」的形式存在，而能對它做的事取決於這個函數形式能回答哪些問題。

</div>

<!--
一張 256×256 的二值影像,狀態總數是 2 的 65536 次方,大約 10 的 19728 次方。
把每個狀態的機率存成一張表,宇宙的原子數大約 10 的 80 次方,遠遠不夠。
詞彙表五萬、長度一百個 token 的句子同理,50000 的 100 次方個狀態。

查表法死於維度,所以分布只能以「函數加參數」的形式存在。
但參數化不是免費的:每一種函數形式都只能回答某些查詢、放棄另一些。
一個分布能被拿來做什麼,取決於它能回答哪些問題。
-->

---

# 介面契約

一個分布,至多提供兩個可呼叫的介面

<div class="mt-6">
<ContractCard />
</div>

<div class="mt-6 tone-muted text-center">

本堂所有操作都只透過這兩個介面進行。每引入一個方法,先問它呼叫了哪個介面。

</div>

<!--
[約 2 分鐘] 這是整堂課的分析工具。

一個分布最多提供兩個可呼叫的介面。
sample() 是「給我一個樣本」,呼叫它得到一個 x,但不附帶任何數值。
logprob(x) 是「這個 x 在此分布下的對數密度是多少」,x 由外部給定,回傳一個數。

接下來每引入一個方法,第一個問題都是:它呼叫了哪個介面。
頁面右下角常駐這兩個介面的徽章,方便隨時對照。
-->

---

# 兩個介面的動作

sample() 交出一個樣本，logprob(x) 交出指定點的對數密度

<div class="mt-4">
<ContractAnim />
</div>

<!--
[約 1 分鐘] 看動畫。

抽樣時落點的疏密由密度決定,但過程不回報任何數值,拿不到密度。
查密度時 x 由外部給定,模型把該點的曲線高度取對數後回報,但這不會產生新樣本。

兩件事互不蘊含:會抽樣不代表能報密度,能報密度也不代表抽樣容易。
-->

---

# 介面盤點

資料側與模型側，各自能回答什麼

| 物件 | `sample()` | `logprob(x)` |
|---|---|---|
| $p_{\text{data}}$(資料) | 有：資料集就是樣本的集合 | **無**：資料不附帶密度 |
| $p_\theta$(模型) | 本堂假設有 | 本堂假設有 |

<div class="mt-6">

資料給的是樣本，沒有密度 <!--凡是需要 $p_{\text{data}}$ 密度的量,都卡在這一格。-->

</div>

<!--
[約 2 分鐘] 這張表只有一格是「無」,而今天有一半的內容卡在那一格。

資料側:sample 有,資料集本身就是一堆樣本;logprob 沒有,資料不附帶密度值。
一張照片、一句話擺在那裡,沒有人告訴我們它在真實分布下的機率是多少。
模型側:今天假設兩個介面都有,這是全部論證的工作假設。

reverse KL 為什麼不能直接算、reward model 為什麼必須存在、資料的熵 H(p) 為什麼估不出來,
追到底都是資料側缺 logprob 這一格。
-->

---
layout: section
---

# 逼近需要選擇散度

量差距的方式決定犯錯的方式

<!--
這一節回答的是:兩個分布的差距要怎麼量。
量差距的方式不只一種,而選了哪一種,等於選了願意犯哪一種錯。
-->

---

# KL divergence

以 $p$ 為權重的對數比期望

$$\mathrm{KL}(p\,\|\,q)=\int p(x)\,\log\frac{p(x)}{q(x)}\,dx$$

<div class="mt-6">

積分由 $p$ 加權：**只在 $p$ 有質量的地方量測差異**。
$p$ 幾乎為零的區域，無論 $q$ 在那裡放了多少機率，都幾乎不進入積分。

</div>

<div class="mt-4">

把 $p_{\text{data}}$ 和 $p_\theta$ 分別放進兩個位置，得到兩個不同的目標：

- $\mathrm{KL}(p_{\text{data}}\,\|\,p_\theta)$: Forward KL
- $\mathrm{KL}(p_\theta\,\|\,p_{\text{data}})$: Reverse KL

</div>

<!--
KL 散度是對數比的期望,權重是前面那個分布 p。
整個積分由 p 加權,這一點決定它後面所有的性質:只在 p 有質量的地方量測差異。
p 幾乎為零的區域,不管 q 在那裡放了多少機率,都幾乎不進入積分。

KL 不對稱,所以把 p_data 和 p_θ 放進兩個位置,會得到兩個不同的目標:
p_data 在前是 forward KL,p_θ 在前是 reverse KL。
不對稱性不需要任何證明技巧,直接看權重在誰手上。
-->

---

# Forward KL：覆蓋是義務

權重在 $p_{\text{data}}$ 手上

$$\mathrm{KL}(p_{\text{data}}\,\|\,p_\theta)=\int p_{\text{data}}\log\frac{p_{\text{data}}}{p_\theta}$$

<KlZeros mode="forward" />

<div class="mt-2 text-sm">

凡 $p_{\text{data}}>0$ 而 $p_\theta\to 0$ 之處，懲罰無上界，所以 $p_\theta$ 必須覆蓋 $p_{\text{data}}$ 的整個支撐集：**zero-avoiding / mode-covering**。代價是把機率質量攤到峰與峰之間的低密度區。

</div>

<div class="mt-2 aside aside-data text-sm tone-muted">

以 forward KL 訓練的語言模型對每種說法都留一點機率,不讓任何一種降到零：開場看到的含糊、發散的續寫,就是這個目標下的合理行為。

</div>

<!--
權重在 p_data 手上。凡是 p_data 大於零、而 p_θ 趨近於零的地方,
log(p/q) 發散,而該點的權重是正的,積分就爆掉,懲罰沒有上界。

所以最小化 forward KL 的模型必須覆蓋 p_data 的整個支撐集,一個峰都不能漏,
這是 zero-avoiding,也叫 mode-covering。
代價是機率質量會被攤到峰與峰之間的低密度區,而那裡其實沒有資料。

以這個目標訓練出來的語言模型,對每一種說法都留一點機率,不讓任何一種降到零。
開場左欄那三段分歧的續寫,是這個目標之下的合理行為,不是模型壞掉。
MLE 等價於最小化 forward KL,介面表那一頁會把這個等價關係算出來。
-->

---

# Reverse KL：放棄不罰

權重換到 $p_\theta$ 手上

$$\mathrm{KL}(p_\theta\,\|\,p_{\text{data}})=\int p_\theta\log\frac{p_\theta}{p_{\text{data}}}$$

<KlZeros mode="reverse" />

<div class="mt-2 text-sm">

$p_\theta$ 不去的地方不進積分，整個丟掉 $p_{\text{data}}$ 的一個眾數不付任何代價；但 $p_\theta$ 涉足 $p_{\text{data}}$ near-zero 區則重罰。**zero-forcing / mode-seeking**。

</div>

<div class="mt-2 aside aside-model text-sm tone-muted">

對齊後的模型回答收斂到少數幾種安全模板、多樣性下降，是 mode-seeking 目標的行為特徵。

</div>

<!--
同一個積分,只把權重換到 p_θ 手上。

p_θ 不去的地方不進積分,所以整個丟掉 p_data 的一個眾數,不付任何代價。
反過來,只要 p_θ 跑到 p_data 接近零的區域,log 比值就爆,重罰。
結果是 p_θ 被逼著待在 p_data 有質量的地方,而且挑一個峰待著就夠:
zero-forcing,也叫 mode-seeking。

對齊之後的模型回答收斂到少數幾種安全模板、多樣性下降,就是這個行為特徵。
全都要和挑一個,兩種行為都不是 bug,是各自目標函數的最優解。
-->

---

# JSD 的定義

以混合分布 $m$ 當共同分母

$$\mathrm{JSD}(p\,\|\,q)=\tfrac12\,\mathrm{KL}\!\left(p\,\Big\|\,m\right)+\tfrac12\,\mathrm{KL}\!\left(q\,\Big\|\,m\right),\qquad m=\tfrac{p+q}{2}$$

<div class="mt-5">

與「對稱化 KL」(Jeffreys divergence)$\mathrm{KL}(p\|q)+\mathrm{KL}(q\|p)$ 不同：
Jeffreys 繼承兩側的無窮大；<br/>
JSD 的分母是混合 $m$，只要 $p$ 或 $q$ 有質量, $m$ 就有質量，因此

</div>

| 性質 | 說明 |
|---|---|
| 有界 | $0\le \mathrm{JSD}\le\log 2$ |
| 對稱 | 定義即對稱 |
| 度量 | $\sqrt{\mathrm{JSD}}$ 滿足度量公理([Endres & Schindelin, 2003](https://doi.org/10.1109/TIT.2003.813506)) |

<!--
JSD 先把 p 和 q 平均成混合分布 m,再分別量 p 到 m、q 到 m 的 KL,取平均。

要跟「對稱化 KL」分開。Jeffreys divergence 是 KL(p‖q) 加 KL(q‖p),
把兩邊的無窮大都繼承下來,一樣會爆。
JSD 的分母是 m,只要 p 或 q 其中一個有質量,m 就有質量,
log(p/m) 最多是 log 2,不會發散。

由此得到三個性質:有界在 0 到 log 2 之間;定義本身對稱;
開根號之後滿足度量公理(Endres & Schindelin, 2003)。
有界是好處,但有界的另一面是飽和。
-->

---

# 有界的代價：飽和

支撐集一旦分離,曲線就貼著上界

<JsdSaturate />

<!--
看曲線:兩個分布的支撐集一旦分離,JSD 就貼著上界 log 2,曲線整個平掉。
平掉的意思是對參數的梯度趨近於零,模型既不知道往哪走,也不知道還差多遠。

這不是邊角案例。高維空間裡兩個分布的支撐集幾乎總是近乎不相交,
維度一高,飽和幾乎是常態。原始 GAN 訓練不穩定,常見的解釋之一就在這裡。
-->

---

# JSD 的判別器讀法

把散度改寫成一個可訓練的分類問題

以等量樣本訓練一個分類器，判斷樣本來自 $p$ 還是 $q$。最優判別器有閉式解:

$$D^*(x)=\frac{p(x)}{p(x)+q(x)}$$

<div class="mt-3">

把 $D^*$ 代回二元分類目標,其值為 $2\,\mathrm{JSD}(p\,\|\,q)-2\log 2$。
換句話說：**JSD 度量的是最優分類器分辨兩個來源的能力**。兩個分布重疊得越好，最優分類器越接近亂猜，JSD 越小。

</div>

<div class="mt-4 card">

同一個式子的等價寫法:

$$D^*(x)=\sigma\!\left(\log\frac{p(x)}{q(x)}\right)\qquad\text{(sigmoid 套在 log ratio 上)}$$

</div>

<!--
把散度改寫成一個可訓練的分類問題。

用等量樣本訓練一個分類器,判斷樣本來自 p 還是 q。最優分類器有閉式解,
D*(x) = p(x)/(p(x)+q(x))。推導只要對每個 x 逐點最大化 BCE 目標,微分即得。

把 D* 代回二元分類目標,值是 2·JSD(p‖q) − 2log2。
所以 JSD 度量的就是最優分類器分辨兩個來源的能力:
兩個分布重疊得越好,最優分類器越接近亂猜,JSD 越小。

下面那個等價寫法要記住:D* = σ(log(p/q)),sigmoid 套在 log ratio 上。
一行驗證:σ(t)=1/(1+e^{−t}),代 t=log(p/q) 得 p/(p+q)。

資訊論形式:JSD(p‖q) = I(X;Z),Z 是 Bernoulli(1/2) 的來源指示變數,
有興趣的同學課後推。
-->

---

# 選擇散度,就是選擇可接受的錯誤

三種選擇,三種失效型態

| | forward KL | JSD | reverse KL |
|---|---|---|---|
| 行為 | mode-covering | 介於其間，但會飽和 | mode-seeking |
| 對稱 | 否 | 是 | 否 |
| 上界 | 無 | $\log 2$ | 無 |
| 失效型態 | 過度平滑、含糊 | 梯度消失或震盪 | 塌縮、多樣性流失 |

<div class="mt-4">
<SpectrumRows :rows="1" mark="objective" />
</div>

<div class="mt-3 text-sm tone-faint">

圖中的橫向位置就是覆蓋程度：左端把質量攤給每一種子集，右端集中在少數幾種。

</div>

<!--
三種選擇,三種失效型態。

forward KL:mode-covering,不對稱,沒有上界,失效型態是過度平滑、含糊。
JSD:介於兩者之間,對稱,上界 log 2,失效型態是梯度消失或訓練震盪。
reverse KL:mode-seeking,不對稱,沒有上界,失效型態是塌縮、多樣性流失。

三個失效型態沒有一個是實作瑕疵,全都寫在目標函數裡,調參數調不掉。沒有中立的散度。

圖的橫向位置就是覆蓋程度:左端把質量攤給每一種說法,右端集中在少數幾種。
第一列先填訓練目標。MLE 在左端,剛剛算過它等價於最小化 forward KL;
RLHF 在右端,是以人類偏好訊號微調的對齊方法,本堂第四節的主題,這裡各給一句即可。
-->

---
layout: none
---

<DemoFrame src="divergence-2d-interactive.html" title="單峰 q 擬合雙峰 p:三種散度,三種解" :maxH="500" />

<!--
[3 分鐘] 用單峰的 q 擬合雙峰的 p,三種散度各解一次。

展示順序:
1. forward KL:q 被拉寬、跨接兩個峰,連峰間的空隙都被填上機率,而那裡沒有資料。
2. reverse KL:q 鎖定其中一個峰,另一個峰完全放棄,損失不會抗議。
3. JSD:落在中間的折衷。

收束:三個解都是最優解,差別只在最優的定義。換一個散度,就換一種錯誤。
-->

---

# 每個散度需要哪些介面

同一份介面清單，三種可得性

| 散度 | 需要的介面 | 可得性 |
|---|---|---|
| forward KL $\mathrm{KL}(p_{\text{data}}\|p_\theta)$ | $p_{\text{data}}$.sample + $p_\theta$.logprob | 兩者皆有 |
| reverse KL $\mathrm{KL}(p_\theta\|p_{\text{data}})$ | $p_\theta$.sample + $p_\theta$.logprob + $p_{\text{data}}$.logprob | **末項不可得** |
| JSD(由雙側密度的混合構成) | 兩側 logprob | 資料側必缺 |

<div class="mt-5">

forward KL 是唯一所需介面皆可得的散度：期望用 $p_{\text{data}}$ 的樣本近似，被積函數只呼叫 $p_\theta$.logprob。

reverse KL 與 JSD 缺的是同一個介面：$p_{\text{data}}$ 沒有 logprob。

</div>

<!--
[約 2 分鐘] 把三個散度拿到介面表上核對。

forward KL:期望對 p_data 取,用資料集的樣本近似即可;
被積函數裡只有 log p_θ,呼叫 p_θ.logprob。兩個介面都拿得到。
展開來看:E_{p_data}[log p_data] 對 θ 而言是常數,
剩下的 −E_{p_data}[log p_θ] 就是 MLE。最大概似估計就是 forward KL,不是另一種方法。

reverse KL:期望對 p_θ 取,需要 p_θ 的 sample 與 logprob,這兩個有;
但被積函數裡有 log p_data,要資料側的密度,這一項拿不到。
JSD 兩側的 logprob 都要,一樣缺資料側。

兩者缺的是同一格。後面出現的 reward model 與判別器,都是在補這一格。
-->

---

# 課後練習

以單一高斯擬合雙峰混合，三種散度各解一次

以單一高斯 $q=\mathcal N(\mu,\sigma^2)$ 擬合 1D 雙峰混合
$p=\tfrac12\mathcal N(-2,0.6^2)+\tfrac12\mathcal N(2,0.6^2)$，分別最小化三種散度並解出 $\mu,\sigma$。

<div class="mt-4">
<DivergenceFit />
</div>

<div class="mt-3 text-sm tone-faint">

預期結果如上圖。動手做一遍，三種行為就不再只是形容詞。

</div>

<!--
課後練習:用單一高斯 q 擬合 1D 雙峰混合 p,分別最小化三種散度,數值解出 μ 與 σ。

提示:一維數值積分用梯形法就夠;reverse KL 有兩個局部極小,兩個峰各一個,
初始值不同會收到不同的解,這件事本身就是 mode-seeking 的直接證據。
圖上是預期結果。動手做一遍,三種行為就不再只是形容詞。
-->

---
layout: section
---

# 分布固定之後

目標分布往往不是模型分布本身：更符合條件、更銳利、更安全、更多樣

<!--
第二節換一個問題:分布已經訓練好了,但想要的往往不是模型分布本身,
而是更符合條件、更銳利、更安全或更多樣的版本。這一節看這些操作有沒有共同形式。
-->

---

# 引導生成的統一形式

base 項、比值項、係數，三個欄位

<div class="mt-10">
<GuidanceForm />
</div>

<!--
[約 3 分鐘] 這一節的內容就是這一條式子。

在對數空間裡,引導後的分布等於 base 項加上係數乘比值項。
三個欄位:base 是被推動的起點,比值項是推動的方向,係數是推動的幅度。

加完之後要再正規化,因為 log 空間相加後機率總和不再是 1,要除以配分函數;
逐 token 的情形,這一步就是 softmax。

接下來的工作很單純:把常見方法逐一填進這三個欄位。
-->

---

# 常見方法都是這條式子(上)

解碼期常見的四種手法

| 方法 | base | 比值項 | 係數 | 需要的介面 |
|---|---|---|---|---|
| temperature | $\log p$ | 無 | $1/T$ | logprob(逐 token) |
| CFG for LLM<br><span class="fine">[Sanchez et al., 2023](https://arxiv.org/abs/2306.17806)</span> | $\log p(x\mid c)$ | $\log p(x\mid c)-\log p(x)$ | $w$ | 兩種條件下的 logprob |
| contrastive decoding<br><span class="fine">[Li et al., 2023](https://arxiv.org/abs/2210.15097)</span>,<br/> Autoguidance<br><span class="fine">[Karras et al., 2024](https://arxiv.org/abs/2406.02507)</span> | $\log p_{\text{strong}}$ | $\log p_{\text{strong}}-\log p_{\text{weak}}$ | $\lambda$ | 兩個模型的 logprob |
| DoLa<br><span class="fine">[Chuang et al., 2024](https://arxiv.org/abs/2309.03883)</span> | $\log p_{\text{final}}$ | 末層與淺層 logits 之差 | $\lambda$ | 中間層 logits |

<!--
解碼期的四種手法。

temperature:base 是 log p,沒有比值項,係數 1/T。
也可以視為比值項退化的情形,把 log p 自己當比值項:
(1/T)·log p = log p + (1/T − 1)·log p。只需要逐 token 的 logprob。

CFG for LLM(Sanchez et al., 2023):同一個模型跑兩次,有條件與無條件各一次,
比值項是兩者之差,係數 w 放大條件的作用。

contrastive decoding(Li et al., 2023):大模型減小模型,
削掉「小模型也會犯」的那類通病,留下大模型獨有的判斷。
Autoguidance(Karras et al., 2024):p_B 用一個刻意訓壞的模型,
比值項是好模型減劣化模型,等於把劣化的方向反向放大。好處是不需要條件標註。

DoLa(Chuang et al., 2024):不必第二個模型,同一個模型內部末層減淺層。

四種方法,四種比值項,同一條式子。
-->

---

# 常見方法都是這條式子(下)

劣化模型、偏好對齊、分布銳化

| 方法 | base | 比值項 | 係數 | 需要的介面 |
|---|---|---|---|---|
| RLHF 最優解<br><span class="fine">[Ouyang et al., 2022](https://arxiv.org/abs/2203.02155);推導見 [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)</span> | $\log \pi_{\text{ref}}$ | $r(y)$ | $1/\beta$ | ref 的 logprob + reward |
| DDO 最優解<br><span class="fine">[Zheng et al., 2025](https://arxiv.org/abs/2503.01103)</span> | $\log p_{\text{ref}}$ | $\log p_{\text{data}}-\log p_{\text{ref}}$ | $1/\beta$ | 兩者的 logprob |

<div class="mt-4 text-sm">

top-k / top-p 是同一操作的硬截斷版：不連續，但同樣在移除尾部機率質量。

</div>

<!--
RLHF 最優解(Ouyang et al., 2022;推導見 Rafailov et al., 2023):
π* 正比於 π_ref 乘 exp(r/β),取 log 就是 base 加上 (1/β) 乘 reward。
這一列的比值項是一個 reward 函數,不是兩個模型的 log 密度差,第四節會回來處理。

DDO 最優解(Zheng et al., 2025):base 是 log p_ref,
比值項是 log p_data 減 log p_ref,係數 1/β。

最後一行:top-k 與 top-p 是同一個操作的硬截斷版本,不連續,
但做的事一樣,都在移除尾部的機率質量。
-->

---
layout: none
---

<DemoFrame src="guidance-playground.html" title="同一根滑桿:temperature、CFG、contrastive decoding" :maxH="500" />

<!--
[3 分鐘] 同一根滑桿,四個情境。

1. temperature 情境,這格的 w 就是 1/T:往左拉 T 升高、分布攤平,往右拉 T 降低、分布尖化,看熵怎麼被係數控制。
2. 切 CFG 情境:同一根桿子,這次做的是條件強化。
3. 切 prompt engineering 情境:桿子變灰,畫面標註「此手法不在係數的位置上」。
   這根灰桿留在螢幕上,討論頁還要用。
4. contrastive 情境,十五秒:大模型減小模型,削掉共同的通病。
5. 切對數空間檢視:w 變動時折線嚴格線性位移,
   「log 空間的線性組合」在這裡眼見為憑。
-->

---

# 三個結論

位置、時機、適用範圍

1. **係數決定移動的幅度。** $w$ 或 $1/\beta$ 越大, 比值項的作用越強；各方法的控制參數都填在式子的同一欄，移動的方向則由各自的比值項決定。

2. **推論期做與訓練期做,差別只在時機。** 同一條式子可以在解碼時套用，也可以內化進權重(第④節)。

3. **適用範圍由介面決定。** 上兩頁的每一列幾乎都要呼叫 logprob；不提供 logprob 的模型或黑箱就無法在推論期操作分布。

<div class="mt-5">
<SpectrumRows :rows="2" mark="decoding" />
</div>

<!--
第一,係數決定移動的幅度。w 或 1/β 越大,比值項的作用越強。
temperature、top-p、CFG 係數、RLHF 的 β,全都填在式子的同一欄,
差別只在各自的比值項指向哪個方向。這是本節的核心。

第二,推論期做與訓練期做,差別只在時機。同一條式子可以在解碼時套用,
也可以內化進權重,第四節會看到後者。

第三,適用範圍由介面決定。前兩頁的每一列幾乎都要呼叫 logprob;
碰到一個不提供 logprob 的模型,或只回傳文字的黑箱 API,整個框架對它失效。

圖的第二列,解碼設定,由此補上。
-->

---

# 討論:prompt 在式子的哪個位置

三個欄位,提示詞只動得了一個

$$\log p_{\text{guided}}=\underbrace{\log p_{\text{base}}}_{\text{prompt 置換的是這裡}}+\;w\,(\log p_A-\log p_B)$$

<div class="mt-5">

prompt engineering 改變條件 $c$，等於整個換掉 base 項；係數 $w$ 與比值項完全不動。

因此：輸出太單調、太發散、過度銳化這類**係數層次的問題，無法靠改寫 prompt 解決**，提示詞不在那個位置上。

</div>

<div class="mt-5 tone-muted">

討論：平常用 LLM 時，有沒有遇過「怎麼改 prompt 都沒改善」的情況，其實出在係數這一欄？

</div>

<!--
[3 分鐘討論] demo 那根灰桿就是這一頁的可視化。

prompt engineering 改變的是條件 c,等於整個換掉 base 項;係數與比值項完全不動。
所以輸出太單調、太發散、過度銳化這一類係數層次的問題,改寫 prompt 解決不了。

常見案例:要求「多給幾種不同的建議」通常收效有限,
因為多樣性由熵控制,而熵是係數那一欄的事,prompt 只能挪動 base 分布。

新生還沒有自己的題目,所以這裡問的是使用經驗:請大家想一個平常用 LLM 時
「怎麼改 prompt 都沒改善」的例子,再判斷它是不是出在係數這一欄。
常見的有三種:要求回答多樣一點、要求不要那麼囉唆、要求輸出穩定的格式。
第三種可以靠第 3 層的 constrained decoding 解決,前兩種靠 prompt 解決不了。
-->

---
layout: center
class: text-center
---

# 中場 Q & A

<!--
休息十分鐘。

時間配置:這十分鐘由第一節吸收(36 分鐘壓到 30 分鐘,
判別式對照頁與高維計數頁各講快一點,共省 6 分鐘),
其餘由第三、四節各緊 2 分鐘,總長維持兩小時。
-->

---
layout: section
---

# 推論期的四層介入

從條件到聚合，每一層呼叫的介面不同

<!--
第三節把推論期能做的事分成四層。重點是每一層呼叫的介面不同,
所以能不能用,取決於手上的模型給不給那個介面。
-->

---

# 四層總覽

四個介入點,各自呼叫不同的介面

<LayerStack />

<div class="mt-4 text-sm tone-muted">

第 1 層與第 4 層只需要 sample:任何黑箱 API 都能做。這正是 prompt engineering 與多數投票類方法對任何服務都可行的原因。

</div>

<!--
[約 2 分鐘] 這張表是本節的地圖。

第 1 層改變條件,第 2 層改變抽樣,第 3 層改變 logits,第 4 層改變樣本的聚合。
介面欄延續前面的分析習慣:看到新方法,先問它呼叫什麼。

第 2、3 層的分界要說清楚。第 2 層是與內容無關的全域重塑與截斷,整條分布一起變形;
第 3 層是逐 token、內容相依的修改,哪些 token 動、動多少,取決於這個 token 是誰。

第 1 層與第 4 層只需要 sample,任何黑箱 API 都能做。
prompt engineering 與多數投票類的方法對任何服務都可行,原因在此。
-->

---

# 第 1 層.改變條件:prompt 即 conditioning

示範收緊的是任務的後驗

$$p(y\mid \text{prompt})=\int p(y\mid \text{task})\;p(\text{task}\mid \text{prompt})\;d\,\text{task}$$

<div class="mt-4">

In-context learning 可讀成隱式貝氏推論([Xie et al., 2022](https://arxiv.org/abs/2111.02080))：prompt 裡的示範不改參數，而是收緊模型對「現在在做哪個 task」的後驗。

</div>

<div class="mt-4 aside aside-data">

memory agent 的機率語意:記憶系統的工作是**挑選哪些證據進入後驗**;存放只是實作手段。

</div>

<!--
prompt 的機率語意就是 conditioning。

把回答對潛在的 task 變數做邊際化:p(y|prompt) 等於對 task 積分,
被積的是 p(y|task) 乘 p(task|prompt)。
In-context learning 可以讀成隱式的貝氏推論(Xie et al., ICLR 2022):
prompt 裡的示範不改任何參數,改的是模型對「現在在做哪個 task」的後驗。

積分式是簡寫,假設 task 給定後 y 與 prompt 條件獨立;
Xie 等人的原式在積分內保留 p(y|task, prompt)。

實驗室的 memory agent 題目:記憶系統的工作是挑選哪些證據進入這個後驗,
存放只是實作手段。
每一則被取回的記憶都是一項證據,取回策略本身就是後驗塑形。
-->

---

# RAG 與 fine-tuning：兩種安裝條件的方式

條件留在脈絡裡,或攤銷進權重

<div class="grid gap-5 mt-3" style="grid-template-columns: 1.75fr 1fr">
<div>

| | RAG | fine-tuning |
|---|---|---|
| 機率意義 | 顯式條件：$p(y\mid x, \text{檢索到的 } d)$ | 條件攤銷進權重：$p_{\theta'}(y\mid x)$ |
| 失效型態 | 檢索錯, 條件就錯：無關文件稀釋後驗 | 分布外遺忘：更新成本高、不可逆 |

<div class="mt-4">

「無關文件稀釋後驗」有量化證據:關鍵資訊放在長脈絡中段時,取用正確率明顯下降(lost-in-the-middle, [Liu et al., 2024](https://arxiv.org/abs/2307.03172))。更多條件不等於更好的條件，$p(\text{task}\mid\text{prompt})$ 會被攤平。

</div>

</div>
<div>

<img src="/public/assets/lost-in-the-middle.png" class="w-full" alt="lost-in-the-middle 的 U 形曲線" />

<div class="fine mt-1">

20 份檢索文件，答案所在位置由開頭移到中段，正確率從 75.8% 掉到 54%，中段甚至低於完全不給文件的 closed-book 基線(紅虛線)。

</div>

</div>
</div>

<!--
同樣是安裝條件,兩種裝法。
RAG 把條件顯式留在脈絡裡,p(y|x, 檢索到的 d);
fine-tuning 把條件攤銷進權重,變成 p_{θ'}(y|x)。

失效型態不同。RAG 是檢索錯條件就錯,而且無關的文件會稀釋後驗;
fine-tuning 是分布外遺忘,更新成本高而且不可逆。

「稀釋後驗」有量化證據:Liu et al. (2024) 的 lost-in-the-middle,
同一份文件集,把答案所在的文件從開頭移到中段,多個模型的正確率呈 U 形下降。
更多條件不等於更好的條件,p(task|prompt) 會被攤平。
對 memory agent 這類題目的直接教訓:取回內容的排序本身就是條件設計。
-->

---

# 第 2 層.改變抽樣

同一組 logits,兩種重塑方式

<TempTopP/>

<div class="mt-3">

temperature 把 logits 除以 $T$，直接調整分布的熵；top-p 截斷尾部後再正規化([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751))。

抽樣設定是一個設計決策：低 $T$ 安全而單調，高 $T$ 多樣而風險高。

</div>

<div class="mt-3">
<SpectrumRows :rows="2" mark="decoding" />
</div>

<!--
同一組 logits,兩種重塑方式。
temperature 把 logits 除以 T,直接調整分布的熵,整條分布一起變形。
top-p 截斷尾部後再正規化(Holtzman et al., 2020),被截掉的機率質量精確歸零。

情感支持系統的抽樣設定是一個明確的設計決策:
低 T 安全但單調,過於模板化的回應會讓使用者覺得敷衍;
高 T 多樣但風險高,可能出現不當建議。
預設值不會替我們回答這個問題,要回答的是兩種錯誤的相對代價,
而且這個取捨要明文寫進系統設計。

temperature 與 top-p 都是第二列上的移動。
-->

---

# 第 3 層.改變 logits

在 softmax 之前修改分數

- **constrained decoding / grammar**: 在合法 token 子集上重新正規化。要求結構化輸出(JSON、SQL)時，這比在 prompt 裡以指示要求格式可靠：非法 token 的機率被精確歸零，軟性指示做不到這一點。

- **logit bias**: 對特定 token 加減常數，即統一引導式中一個手寫的比值項。

- **contrastive decoding、DoLa、CFG for LLM**: 上一節表中的三列，透過逐 token 修改 logits 後再 softmax。

<!--
在 softmax 之前修改分數。

constrained decoding 或 grammar:只在合法的 token 子集上重新正規化。
機率語意是條件在「輸出屬於文法 L」這個事件上,p(y | y∈L) 的逐 token 實作。
要求 JSON 或 SQL 這類結構化輸出時,它比在 prompt 裡用指示要求格式可靠:
非法 token 的機率被精確歸零,軟性指示做不到這一點。

logit bias:對特定 token 加減一個常數,就是統一引導式裡一個手寫的比值項。

contrastive decoding、DoLa、CFG for LLM:前面表中的三列,安裝位置都在這一層,
逐 token 修改 logits 之後再 softmax。
-->

---

# 第 4 層.改變樣本的聚合

抽多個『完整』樣本，重新估計答案的分布

- **Best-of-n**: 抽 $n$ 個，用外部評分挑一個
- **Self-consistency**: 抽多條推理路徑，對最終答案投票
- **MBR**(minimum Bayes risk): 選「與其他樣本平均距離最近」的輸出
- **Reranking**: 以另一個模型重排候選

<div class="mt-4">

Self-consistency 的機率語意即 Monte Carlo 邊際化：
$p(a\mid q)=\sum_r p(a\mid r,q)\,p(r\mid q)$，對推理路徑 $r$ 積分 ([Wang et al., 2023](https://arxiv.org/abs/2203.11171))。

</div>

<!--
這一層不動分布本身,改的是怎麼從多個樣本聚合出答案。

best-of-n:抽 n 個,用外部評分挑一個。
self-consistency:抽多條推理路徑,對最終答案投票。
MBR:選與其他樣本平均距離最近的輸出。
reranking:用另一個模型重排候選。

self-consistency 的機率語意就是 Monte Carlo 邊際化:
p(a|q) 等於對推理路徑 r 求和 p(a|r,q)·p(r|q)(Wang et al., 2023)。
投票是在對 r 積分,不是啟發式。

這一層只需要 sample,logprob 可選,用於加權。
與第 1 層合起來看:黑箱 API 能做的事其實不少,這兩層都不需要 logprob。
-->

---

# classifier guidance:分類器提供比值項

無條件模型與分類器兩個介面都拿得到,貝氏反轉才寫得出來

$$\log p_w(x)\;=\;\log p(x)\;+\;w\,\log p(c\mid x)$$

<div class="mt-4">

由貝氏, $\log p(c\mid x)=\log p(x\mid c)-\log p(x)+\text{const}$：分類器交出來的就是那個比值項。base 項是無條件模型，係數是 guidance scale $w$；安裝位置在第 3 層，逐步修改分數 ([Dhariwal & Nichol, 2021](https://arxiv.org/abs/2105.05233))。

</div>

| $w$ | 得到的分布 |
|---|---|
| $0$ | 無條件分布 $p(x)$ |
| $1$ | 貝氏後驗 $p(x\mid c)$，因為 $p(x)\,p(c\mid x)=p(x,c)$ |
| $>1$ | 外插：條件更銳利，多樣性下降 |

<div class="mt-3 aside aside-third text-sm tone-muted">

代價是一個額外介面：$p(c\mid x)$ 要另外訓練。CFG 把比值項改由同一個模型跑有條件與無條件兩次算出，這個額外介面就不需要了([Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598))。

</div>

<!--
這一頁是貝氏反轉唯一站得住的用法,條件是兩個介面都在手上:
無條件模型給 p(x),另外訓練的分類器給 p(c|x)。

三個 w 值要講清楚:
w=0 沒有引導;w=1 時 p(x)·p(c|x) 就是聯合分布 p(x,c),正規化後正是後驗 p(x|c),
不是近似,是恆等式;w>1 是外插,把無條件分布反向扣掉一部分,
等價寫法是 p_w ∝ p(x)^(1−w)·(π_c·p(x|c))^w,與 DDO 最優解同一種幾何內插的形狀。
外插的效果就是保真度換多樣性,實務上正是這樣調的。

介面代價:分類器必須看得懂加噪後的 x,所以要在每個雜訊尺度上另訓一個。
CFG 的作法是讓同一個模型同時學有條件與無條件分數,比值項自己算得出來,
省掉這個額外網路,代價是每步兩次前向。

反例對照(有人問再講):端到端 ASR 沒有生成式的 p(audio|text),
所以那邊寫不出這種貝氏反轉,只能在 log 空間直接把分數加起來。
-->

---
layout: none
---

<DemoFrame src="classifier-guidance.html" title="classifier guidance:w=0 無條件,w=1 後驗,w>1 外插" :maxH="500" />

<!--
[2 分鐘] 三張圖由左到右:無條件分布、分類器、引導後的分布。左邊兩張與 w 無關。

展示順序:
1. w=0:右圖與左圖相同,讀數的「目標類別質量」等於該類別的先驗(A 是 45%、B 是 35%、C 是 20%)。
2. 拉到 w=1:右上角的 L1 距離歸零,這時的分布就是貝氏後驗,一點誤差都沒有。
3. 繼續拉到 3、4:質量塌向分類器最有把握的區域,樣本雲肉眼可見地縮小,
   格點熵下降、目標類別質量升到九成以上。保真度換多樣性,在這裡是看得到的。
4. 換類別 C(先驗只有 0.20):同一根 w,窄類別被拉起來的幅度更明顯。

中間那張圖是分類器的後驗,虛線是 0.5 決策邊界。要強調它就是比值項:
由貝氏它等於 log p(x|c) − log p(x) + const,不是另一種新東西。

本節收束:classifier guidance 安裝在第 3 層,逐步修改分數;
日常在用的方法多半在第 1 層,只呼叫 sample。第 2 到第 4 層動的是同一條分布的其他位置,
各自需要什麼介面,四層總表上都標好了。
-->

---
layout: section
---

# 後訓練的介入

SFT、RLHF、DPO、DDO：把移動寫進參數

<!--
第四節把同樣的移動寫進參數:SFT、RLHF、DPO、DDO。
問題和前面一樣:目標函數是哪個散度,需要哪些介面。
-->

---

# SFT:在新資料上重做 MLE

換資料,不換目標函數

$$\max_\theta\;\mathbb{E}_{(x,y)\sim \mathcal D_{\text{SFT}}}\big[\log \pi_\theta(y\mid x)\big]$$

<div class="mt-5">

目標函數與預訓練相同，仍是 forward KL，只是換了資料分布，透過顯式把分布遷移到一個較小的子集上,能減輕原先過度通用的特性。但有對新子集 mode-covering 的可能性。

</div>

<div class="mt-5">
<SpectrumRows :rows="3" mark="weights" />
</div>

<!--
SFT 就是在新資料上重做一次 MLE,最大化每組 (x,y) 的 log π_θ(y|x)。

目標函數與預訓練完全相同,仍然是 forward KL,換掉的只有資料分布。
所以 SFT 之後的模型仍然落在 mode-covering 端:格式與語氣被塑形了,
但含糊、過度覆蓋的傾向不變,因為那是目標函數的性質,不是資料的性質。

第三列,權重微調,從這裡開始畫,SFT 位在左端。
-->

---

# RLHF 的目標與閉式解

reward 的期望,減去與參考模型的 KL

$$\max_\pi\;\mathbb{E}_{y\sim\pi}\big[r(y)\big]-\beta\,\mathrm{KL}\big(\pi\,\|\,\pi_{\text{ref}}\big)$$

<div class="mt-3">

這個目標有閉式最優解(推導見 [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) 附錄；把解代回目標即可驗證)：

$$\pi^*(y)\;\propto\;\pi_{\text{ref}}(y)\,\exp\!\big(r(y)/\beta\big)$$

取 log 即第②節表中的 RLHF 列。而整個最佳化問題等價於

$$\min_\pi\;\mathrm{KL}\big(\pi\,\|\,\pi^*\big)$$

<div class="tone-muted">

KL 的第一個位置放的是 $\pi$，這正是 reverse KL：對齊訓練落在 mode-seeking 端。

</div>

</div>

<!--
RLHF 的目標:最大化 reward 的期望,減去 β 乘上與參考模型的 KL。

這個目標有閉式最優解,π* 正比於 π_ref 乘 exp(r/β);
把解代回目標即可驗證,推導見 Rafailov et al. (2023) 附錄。
取 log 就是第二節表中的 RLHF 列。

更要緊的是整個最佳化問題等價於 min KL(π ‖ π*)。
代回去算:E_π[r] − β·KL(π‖π_ref) = −β·KL(π‖π*) + 常數。

看 KL 的第一個位置放的是誰:是 π,被訓練的模型自己。這正是 reverse KL。
對齊訓練落在 mode-seeking 端,而 mode-seeking 的一切性質,
挑模板、丟多樣性,全部自動繼承過來。
-->

---

# DPO:把 reward 消掉

模型自己的 log ratio 就是隱式 reward

閉式解可以反過來解出 reward：

$$r(y)=\beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)}+\text{const}$$

<div class="mt-3">

把這個表達式代入偏好資料的 Bradley–Terry 損失，reward model 從式子裡消失([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)):

$$\mathcal L_{\text{DPO}}=-\mathbb{E}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)\right]$$

</div>

<div class="mt-4">

整條損失只呼叫 $\pi_\theta$ 與 $\pi_{\text{ref}}$ 的 logprob 介面；最佳化的仍是同一個 reverse KL 目標。

</div>

<!--
DPO 從同一個閉式解出發,反過來解出 reward:r = β·log(π*/π_ref) + 常數。

把這個表達式代進偏好資料的 Bradley–Terry 損失,
P(y_w 勝 y_l) = σ(r(y_w) − r(y_l)),兩個常數相消,
reward model 從式子裡整個消失(Rafailov et al., 2023),剩下的就是投影片上這條損失。

兩件事要注意。第一,整條損失只呼叫 π_θ 與 π_ref 的 logprob,不需要額外網路。
第二,消掉 reward model 並沒有換掉目標,最佳化的仍是同一個 reverse KL 目標,
所以塌縮的傾向也一併繼承,DPO 不會比 RLHF 更不容易單調。
-->

---

# reward model 補的是哪一格

reverse KL 缺的那一個介面

| 散度 | 缺的介面 | 補上它的東西 |
|---|---|---|
| reverse KL | 目標分布的 logprob | **reward model** |

<div class="mt-5">

需求表裡標「不可得」的那一格，由 reward model 填補:它從人類偏好標註學出純量分數 $r(y)$，而 $r/\beta$ 正好充當目標分布相對 $\pi_{\text{ref}}$ 的 log 密度比。

</div>

<div class="mt-4">

要注意目標分布在此換了對象：對齊瞄準的已非 $p_{\text{data}}$，而是偏好加權後的 $\pi^*\propto\pi_{\text{ref}}\,e^{r/\beta}$。<br/>缺 logprob 的困境不變，代理的品質決定對齊的品質：reward model 學壞了，$\pi^*$ 就指向錯的地方(reward hacking)。

</div>

<!--
回到第一節那張介面需求表:reverse KL 缺的是目標分布的 logprob,標「不可得」的那一格。
reward model 就是來填這一格的。它從人類偏好標註學出純量分數 r(y),
而 r/β 正好充當目標分布相對 π_ref 的 log 密度比。

有一件事要講清楚:目標分布在這裡換了對象。
對齊瞄準的已經不是 p_data,而是偏好加權後的 π*,正比於 π_ref 乘 exp(r/β)。
但困境沒變,目標分布的 logprob 仍然拿不到,只能造一個代理。
代理的品質就決定對齊的品質:reward model 學壞了,π* 就指向錯的地方,
這就是 reward hacking。代理不只 reward model 一種,能量函數、分類器都可以充當。

若有人問「兩端的散度連引數都不同,這個對照還成立嗎」:成立。
這裡比較的是散度的方向,由誰加權、對誰 zero-forcing,而不是固定的一對分布。
mode-seeking 是方向的性質,對任何目標分布都導致收斂到該目標的少數眾數,
模板化、多樣性下降這些行為特徵,不因目標分布換人而改變。
-->

---

# β 是唯一抑制塌縮的項

單調來自目標函數的形式

「對齊讓模型更安全，但更單調」可以從目標函數推出，不是經驗巧合:

<div class="mt-3">

- reverse KL 目標的最優解**只在 reward 高的區域放質量**；多樣性沒有出現在目標函數裡
- 唯一抑制塌縮的是 $\beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$,而它約束的是**與參考模型的距離**，不是多樣性本身
- 實測：RLHF 後輸出多樣性系統性下降([Kirk et al., 2024](https://arxiv.org/abs/2310.06452))

</div>

<div class="mt-4 aside aside-model">

$\beta$ 的取捨問題：調鬆帶來變化也帶來不受控的風險，調緊帶來安全也帶來模板。

</div>

<!--
「對齊讓模型更安全,但更單調」可以從目標函數推出來,不是經驗上的巧合。

第一,reverse KL 目標的最優解只在 reward 高的區域放質量,
多樣性根本沒有出現在目標函數裡。
第二,唯一在抑制塌縮的是 β·KL(π‖π_ref),而它約束的是與參考模型的距離,
不是多樣性本身;只因為參考模型比較散,才順帶保住一些多樣性。
第三,實測上 RLHF 之後輸出多樣性系統性下降(Kirk et al., 2024)。

Kirk 等人量的是 summarization 等任務上 per-input 與 cross-input 的多樣性,
兩者皆下降,同時泛化能力上升。
要提醒的是,「多樣性與 β 鬆緊同向」是由目標函數形式得到的理論預期,
該文並未系統性掃 β,課堂上不要說成實測結論。

情感支持系統的安全與多樣共用同一個 β:調鬆帶來變化也帶來風險,
調緊帶來安全也帶來模板,同一個參數同時決定這兩件事。
開場右欄那三則幾乎一樣的回覆,機制就在這裡。
-->

---

# 逐點計分器的結構極限

reward model 與 judge 共有的限制

reward model 對**單一樣本**給分：$r(y)\in\mathbb R$。

<div class="mt-3">

「這個分布太窄」是**分布層級**的性質，單點分數的介面裡沒有承載它的欄位：<br/>
<div class="ml-10">每個模板回答逐點看都得高分,計分器對「全部都是同一種回答」無從抗議。
</div>
</div>

<div class="mt-4">

LLM-as-judge 同樣逐點評審，同樣的極限；把 judge 換強並不補上這個欄位。

</div>

<div class="mt-4 tone-muted">

因此抑制塌縮的只剩 β 項對參考模型距離的約束。

</div>

<!--
reward model 對單一樣本給分,r(y) 是一個實數。

「這個分布太窄」是分布層級的性質,而單點分數的介面裡沒有欄位承載它:
每一個模板回答逐點看都得高分,計分器對「全部都是同一種回答」無從抗議。

形式化一句:任何 r: Y→ℝ 的期望 E_π[r] 對 π 的重複度不敏感,
除非 r 本身以分布為輸入,而它不是。

LLM-as-judge 也是逐點評審,受同一個限制。把 judge 換成更強的模型並不補上這個欄位,
因為缺的是介面,不是能力。

所以抑制塌縮的只剩 β 那一項對參考模型距離的約束。
-->

---

# 兩件已經在手上的事實

一個判別器閉式解，一條引導式

<div class="mt-4 grid grid-cols-1 gap-5">

<div class="card">

**事實一(第①節)** 分辨兩個分布的最優判別器是

$$d^*(x)=\sigma\!\left(\log\frac{p_{\text{data}}(x)}{q(x)}\right)$$

</div>

<div class="card">

**事實二(第②節)** 統一引導式方法表的末列:

$$\log p_{\text{guided}}=\log p_{\text{ref}}+\tfrac1\beta\big(\log p_{\text{data}}-\log p_{\text{ref}}\big)$$

</div>

</div>

<div class="mt-5 tone-muted">

接下來的方法只用這兩件事，不引入任何新原理。

</div>

<!--
[約 1 分鐘] 兩件事直接陳述,此刻應該都認得。

事實一,第一節:分辨兩個分布的最優判別器是 σ(log(p_data/q))。
事實二,第二節:統一引導式方法表的末列,
log p_guided = log p_ref + (1/β)(log p_data − log p_ref)。

接下來的方法只用這兩件事,不引入任何新原理。
-->

---

# DDO：用自己的 logprob 當判別器

判別器不必是另一個網路

任何提供 logprob 的分布，都可以直接**宣告**自己是判別器([Zheng et al., 2025](https://arxiv.org/abs/2503.01103)):

$$d_\theta(x)\;=\;\sigma\!\left(\beta\,\log\frac{p_\theta(x)}{p_{\text{ref}}(x)}\right)$$

<div class="mt-3">

- $\beta$ 是必要的縮放：log ratio 的逐維差異隨維度累積、量級可達數十至數百,直接進 sigmoid 會使梯度消失
- 以標準 BCE 訓練：真樣本標 1, 參考模型樣本標 0
- BCE 的最優判別器是 $\sigma(\log(p_{\text{data}}/p_{\text{ref}}))$：對照兩式, $\beta=1$ 時最優解就是 $p_\theta=p_{\text{data}}$, 一般 $\beta$ 給出 $p_\theta^*\propto p_{\text{ref}}^{\,1-1/\beta}p_{\text{data}}^{\,1/\beta}$

</div>

<div class="mt-4 aside aside-third tone-muted">

DPO 把 reward 參數化成 log ratio；DDO 把**判別器**參數化成 log ratio。參數化手法相同，套用的對象不同。

</div>

<!--
關鍵的一步:任何提供 logprob 的分布,都可以直接宣告自己是判別器
(Zheng et al., 2025)。不必另外訓練一個判別網路,
把 σ(β·log(p_θ/p_ref)) 當成判別器就好。

三點說明。
β 是必要的縮放:log ratio 的逐維差異隨維度累積,量級可達數十甚至數百,
直接丟進 sigmoid 會飽和,梯度消失。
訓練用標準 BCE:真實資料標 1,參考模型抽出來的樣本標 0。
BCE 的最優判別器是 σ(log(p_data/p_ref));和宣告的形式對照,
β = 1 時最優解就是 p_θ = p_data,一般的 β 給出
p_θ* 正比於 p_ref^(1−1/β)·p_data^(1/β)。

與 DPO 對照:DPO 的隱式 reward 是 β·log(π_θ/π_ref),
DDO 的隱式判別器是 σ(β·log(p_θ/p_ref)),參數化手法完全平行,套用的對象不同。
資料需求也不同:DPO 要成對的偏好標註,DDO 只要原始訓練資料加參考模型的樣本。
-->

---

# DDO 的機制

兩種樣本，一個 BCE 損失

<DdoMechanism />

<div class="mt-2 text-sm tone-muted text-center">

無需額外判別器網路、無需交替訓練、無需對抽樣過程反向傳播。

</div>

<!--
看圖:兩種樣本,一個 BCE 損失。
真實資料當正例,參考模型的樣本當負例,判別器就是模型自己的 log ratio,
梯度直接回到 p_θ 的參數。

三個「無需」對照的是傳統對抗訓練的三大負擔:
不需要額外的判別器網路、不需要生成器與判別器交替訓練、
不需要對抽樣過程反向傳播。

圖上的虛線是 self-play:一輪訓練結束後把 p_θ 存成新的 p_ref,再訓練一輪。
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

MLE 的梯度只有第一種作用：在資料點上抬密度，對模型自己多出來的機率質量沒有任何移除機制。DDO 的損失同時含 $\mathbb E_{p_{\text{data}}}[\cdot]$(抬)與 $\mathbb E_{p_{\text{ref}}}[\cdot]$(壓)，「遠離自己的樣本」的精確意義在此。

</div>

<!--
取 β=1 的梯度長這樣,重點只在符號。

括號裡的 (p_θ − p_data) 決定方向:
模型蓋不夠的地方,p_θ 小於 p_data,符號為負,梯度把該處密度抬升;
模型蓋過頭的地方,p_θ 大於 p_data,符號為正,梯度把該處密度壓低。
前面的 (1 − d_θ) 是權重:已經分得很開的區域力道小,分不開的區域力道大。

對照 MLE:MLE 的梯度只有第一種作用,在資料點上抬密度,
對模型自己多出來的機率質量沒有任何移除機制,因為資料裡沒有負例。
DDO 的損失同時含 E_{p_data}(抬)與 E_{p_ref}(壓),
「遠離自己的樣本」的精確意義在此。

推導步驟,追問時再用:BCE 對 θ 微分,σ 的導數帶出 d_θ(1−d_θ),
兩項期望合併時利用 p_ref·d_θ = (1−d_θ)·p_θ(這一步 β=1 才成立),
得到 (p_θ − p_data) 的加權形式。
一般 β 的梯度是 β·[p_ref·d_θ − p_data(1−d_θ)]∇log p_θ,
完整推導見 Zheng et al. (2025) 附錄。
-->

---
layout: none
---

<DemoFrame src="mle-vs-ddo-gradient.html" title="MLE 與 DDO 的梯度場" :maxH="500" />

<!--
[3 分鐘] 左右對照。

左邊是 MLE 的梯度場,只有指向資料點的吸力,模型多餘的質量原地不動。
右邊是 DDO 的梯度場,資料區吸、過剩區推。
請大家找「MLE 動不了、DDO 在動」的區域,那就是 p_θ 大於 p_data 的地方。

量化證據看畫面上的兩個讀數:右側的 reverse KL 下降明顯快於左側,
forward KL 則兩側皆降。「同時作用於兩端」在這裡以數字兌現。
-->

---

# DDO 在覆蓋程度上的位置

抬升與壓低分屬兩端

<div class="mt-4">
<SpectrumRows :rows="3" ddo />
</div>

<div class="mt-6">

抬升項做的是 forward KL 的事(把質量搬向資料)，壓低項做的是 reverse KL 的事(從過剩區撤出質量)：
<div class="text-center text-xl mt-5">

**DDO 同時作用於兩端**，多數方法只能佔一端。
</div>
</div>

<!--
抬升項做的是 forward KL 的事,把質量搬向資料;
壓低項做的是 reverse KL 的事,從過剩區撤出質量。
所以 DDO 同時作用於兩端,而多數方法只能佔一端。

第三列到這裡完整:左端 SFT、右端 DPO,而 DDO 是圖上那條橫跨兩端的帶,
左箭頭是抬升項、右箭頭是壓低項。其他方法都只佔一端,這是 DDO 與它們的差別。
-->

---

# DPO、DDO 與統一引導式

同一種參數化，兩個用途

| | DPO | DDO |
|---|---|---|
| 隱式參數化 | reward $=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ | 判別器 $=\sigma\!\big(\beta\log\frac{p_\theta}{p_{\text{ref}}}\big)$ |
| 目標 | 偏好學習 | 分布對齊 |
| 資料 | 成對人類標註 | 原始訓練資料，無需配對 |

<div class="mt-4">

DDO 的最優解 ([Zheng et al., 2025](https://arxiv.org/abs/2503.01103)):

$$p_\theta^*\;\propto\;p_{\text{ref}}^{\,1-1/\beta}\;p_{\text{data}}^{\,1/\beta}$$

正是統一引導式方法表末列的指數形式。guidance 在推論期執行這次銳化，而 DDO 則把同一個操作寫進權重，這使得推論時不再需要第二次前向。

</div>

<!--
同一種參數化,兩個用途。
DPO 把 reward 參數化成 β·log(π_θ/π_ref),用途是偏好學習,資料要成對人類標註。
DDO 把判別器參數化成 σ(β·log(p_θ/p_ref)),用途是分布對齊,
只要原始訓練資料,不必配對。

DDO 的最優解是 p_θ* 正比於 p_ref^(1−1/β)·p_data^(1/β)。取 log 驗證一次:
log p* = log p_ref + (1/β)(log p_data − log p_ref) + 常數,
與第二節方法表的末列逐項相同。

差別在時機:guidance 在推論期執行這次銳化,每一步都要多跑一次前向;
DDO 把同一個操作寫進權重,推論時不再需要第二次前向。
-->

---

# 訓練到頂之後

MLE 收斂之後，指標仍有空間

[Zheng et al. (2025)](https://arxiv.org/abs/2503.01103) 微調多個提供 logprob 的生成模型，兩則觀察與本堂論證直接相關：

<div class="grid grid-cols-2 gap-6 mt-3">
<div>

1. **繼續用 MLE 訓練，指標不動甚至劣化。** forward KL 目標已到達其可達的極限；卡住的原因在目標函數，調參數解不開。

2. **有些模型此前靠 top-k / top-p 撐指標。** 截斷降低了有效溫度，把分布的缺陷藏起來而非修好；換上 DDO 後，不加任何截斷的原始分布品質即提升。

</div>
<div>

<img src="/public/assets/ddo-iter.png" class="w-full" alt="DDO 微調的 FID 曲線" />

<div class="fine mt-2">

class-conditional CIFAR-10 第一輪微調(圖 6b、6c)。灰色虛線是繼續用 MLE 訓練, FID 由 1.85 起不降反升; 彩色線是不同超參數設置的 DDO, 1500 步內壓到 1.6 附近。

</div>

</div>
</div>

<!--
Zheng et al. (2025) 微調多個提供 logprob 的生成模型,兩則觀察與今天的論證直接相關。

第一,MLE 收斂之後繼續用 MLE 訓練,指標不動,有時甚至劣化。
forward KL 目標已經到達它可達的極限,卡住的原因在目標函數,調參數解不開,
要換的是目標函數。

第二,有些模型此前是靠 top-k 或 top-p 撐指標。
截斷實際上降低了有效溫度,把分布的缺陷藏起來而沒有修好;
換上 DDO 之後,不加任何截斷的原始分布品質就提升了。
這一則呼應前面說過的:截斷是硬移除尾部質量,被藏起來的問題還在。

兩則出處都是 Zheng et al. (2025)。

圖是該文的圖 6(b)(c),class-conditional CIFAR-10 的第一輪微調曲線。
兩個超參數:β 是判別器裡 log ratio 的縮放,α 是壓低項(參考模型樣本那一項)的權重,
損失是 −p_data·log σ(β·r_θ) − α·p_ref·log(1 − σ(β·r_θ))。
(b) 固定 α=4.0 掃 β,(c) 固定 β=0.05 掃 α,兩種掃法效果相近。

灰色虛線要照原文講:該文說明繼續用 MLE 訓練無法改善、甚至劣化,
一部分原因是沒有保留原本的 optimizer 狀態,更根本的原因才是 forward KL 目標本身的極限。
不要把劣化整段都算到目標函數頭上。

另外提醒曲線末端會回升:α、β 給得太大或訓練太久都會過頭,
與②節「係數決定移動的幅度」是同一件事。
-->

---

# 兩種輸出, 覆蓋程度的兩端

開場的兩欄，各自的位置

<div class="mt-2">
<SpectrumRows :rows="3" ddo />
</div>

<div class="mt-6">

含糊的續寫出自 forward KL 訓練，同質的回答出自 reverse KL 對齊，分居覆蓋程度的兩端。這個位置可以在三個層次上調整:

- 訓練目標(選哪個散度)
- 解碼設定(temperature、top-p、guidance 係數)
- 權重微調(SFT、DPO、DDO 的 $\beta$)

</div>

<!--
回到開場的兩欄。
左欄含糊、發散的續寫來自 forward KL 訓練;
右欄同質、模板化的回答來自 reverse KL 對齊。
兩者分居覆蓋程度的兩端,而這個位置有三個層次可以調整:

訓練目標,選哪一個散度;
解碼設定,temperature、top-p、guidance 的係數;
權重微調,SFT、DPO、DDO 的 β。

現象、散度、覆蓋程度、三個可調層次,到這裡串成一條完整的解釋。
-->

---

# 本堂的假設, 與兩個未解的問題

兩個介面齊備，是全部論證的前提

**本堂自始至終的假設**：$\pi_{\text{ref}}$(以及 $p_\theta$) 同時提供 `sample()` 與 `logprob(x)`。

<div class="mt-5">

兩個此刻沒有答案的問題：

1. reverse KL 一族缺 $p_{\text{data}}$.logprob，各方法拿什麼**代理**去補？本堂只見過 reward model 一種。
2. 一個**完全不提供 logprob** 的模型，要拿什麼訓練？

</div>

<!--
先把假設攤開:今天自始至終假設 π_ref 以及 p_θ 同時提供 sample() 與 logprob(x)。
所有論證都建立在這個前提上。

兩個此刻沒有答案的問題。
第一,reverse KL 一族缺的是 p_data 的 logprob,各方法拿什麼代理去補?
今天只見過 reward model 一種。
第二,一個完全不提供 logprob 的模型,要拿什麼訓練?
今天寫過的每一種損失,對它都寫不出來。

這兩個問題今天不展開、不解答。
-->

---

# 本堂總結

機率花式操作技巧

1. **介面決定可能性。** 一個分布至多提供 `sample()` 與 `logprob(x)`。  
$p_{\text{data}}$ 只有前者，缺的失的 `logprob(x)` 正是模型的擬何目標。

2. **選散度就是選錯誤型態。** forward KL 過度覆蓋、reverse KL 塌縮、JSD 飽和，三種失效都寫在目標函數裡。

3. **引導只有一條式子。** $\log p_{\text{guided}}=\log p_{\text{base}}+w\,(\log p_A-\log p_B)$.  
temperature、CFG、classifier guidance、RLHF、DDO 都只是填欄位。

4. **推論期分四層。** 條件、抽樣、logits、聚合；第 1、4 層只需要 `sample()`，第 2、3 層要 `logprob(x)`。

5. **覆蓋程度可以在三個層次上調整。** 訓練目標、解碼設定、權重微調。

<div class="mt-5 tone-faint">

遇到沒看過的方法，第一個問題永遠一樣：它呼叫了哪一個介面。

</div>

<!--
[約 2 分鐘] 收尾,逐條唸過即可,五條在前面都出現過。

第 1 條:介面盤點那張表裡,資料側的 logprob 是「無」,
reverse KL 不可算、reward model 要存在、H(p) 估不出來,全都由那一格而來。
第 2 條:三種失效型態都是目標函數的最優解,不是實作瑕疵。
第 3 條:式子的三個欄位,base、比值項、係數;prompt 只能換 base。
第 4 條:黑箱 API 能做第 1 層與第 4 層,第 2、3 層要有 logprob 才進得去。
第 5 條:開場那兩欄輸出的位置,由這三個層次決定。

如果只帶走一件事,帶第 1 條:看到新方法先問它呼叫哪一個介面。
完整書目見課程 repo 的文獻頁。
-->
