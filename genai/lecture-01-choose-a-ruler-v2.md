---
theme: seriph
title: 生成模型入門 · 第一堂:選一把尺(v2)
titleTemplate: '%s'
background: none
class: text-center
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
colorSchema: dark
themeConfig:
  primary: '#b48cff'
transition: slide-left
fonts:
  sans: Noto Sans TC
  mono: IBM Plex Mono
duration: 180min
mdc: true
---

<div class="eyebrow">生成模型入門 · Lecture 01 / 02 · v2</div>

# 選一把尺

## 生成即分布逼近,而「逼近」並非唯一定義的操作

<div class="pt-6 text-sm opacity-70">
實驗室新生課程 · 180 分鐘<br>
先修:機率、KL 散度、Jensen 不等式、反向傳播
</div>

<div class="abs-br m-6 text-xs opacity-50">
下堂課:在軸上滑動 · 引導生成
</div>

<!--
這門課不是生成模型通識,是實驗室現有研究的底層說明書。

骨幹放散度,不放模型分類法。分類法只給一張清單,散度給的是可推導的因果。

v2 的變動:③ 補上完整的家族分類與體質對照,④ 把 VAE 與 GAN 從各一頁擴成各一段。
如果只有 120 分鐘,砍法是 ③ 的 Trilemma 與 ④ 的「改進」兩頁各留 1 分鐘帶過。

開場先不要講大綱,直接做 ① 的兩個現場示範。
-->

---
layout: section
class: sec-intro
---

# ① 兩個失敗現場

## 現場示範,不看投影片

<!--
0–12 分。

示範 1:問一個 base model 一個有標準答案的問題,它給出「這取決於很多因素……」
示範 2:對一個對齊過的模型用同一個 prompt 取樣十次,得到十個幾乎相同的回答

示範完再翻下一頁。
-->

---
layout: center
---

<FailureScenes />

<div v-click class="mt-8 text-center text-xl">

**這兩種失敗,是同一種嗎?**

</div>

<!--
讓學生先猜。多數人會說「不一樣,一個是太笨、一個是被管太嚴」。

不要立刻否定。說:接下來 40 分鐘我要證明它們是同一件事。
-->

---
layout: center
class: text-center
---

<div class="eyebrow">本堂論題</div>

<div class="text-2xl leading-relaxed mt-6 mb-8 px-10">

base model 的空泛 hedging,<br>
與對齊後模型的千篇一律,<br>
是**同一條軸的兩端**。

</div>

<div v-click>
  <SpectrumAxis :rows="0" />
  <div class="text-xs opacity-60 mt-2">這條軸今天畫最上面一列,下堂課補完下面兩列。</div>
</div>

<!--
第 12 分鐘提出論題,下堂課最後一頁回收。

現在這條軸還是空的。今天結束時,最上面一列會被填滿;下堂課你會看到下面兩列其實是上面那一列的「推論時版本」與「權重版本」。
-->

---

# 你們手上的題目,其實都是機率問題

<div class="text-sm">

| 實驗室題目 | 它其實是什麼機率問題 | 在軸上的位置 |
|---|---|---|
| prompt engineering | 選擇條件變數 $c$,操控 $p(y \mid c)$ | 只換條件,不動目標函數 |
| memory agent | 條件集合的建構;長 context 下 $p(\text{task} \mid \text{prompt})$ 被稀釋 | 同上 |
| 情感支持對話 | 通用安慰語 vs. 對齊後的多樣性塌陷 | 偏右,且被 $\beta$ 綁住 |
| 虛假前提檢測 | $p(y\mid x)$ 永遠 well-defined,即使 $p(x)\approx 0$ | 左端的結構性後果 |
| confidence vs. accuracy | 校準;predictive entropy vs. semantic entropy | 量測工具本身 |
| LLM-ASR | $p(\text{text}\mid\text{audio})\propto p(\text{audio}\mid\text{text})\,p(\text{text})$ | 對數空間的線性組合 |

</div>

<div v-click class="mt-4 p-3 border-l-4 border-amber-400 text-sm">
右邊那一欄現在看不懂沒關係。這是本課的<b>驗收清單</b>,下堂課最後一頁逐格回填。
</div>

<!--
這頁只放一次,不要逐列講解,30 秒掃過即可。目的是讓每個人在第一分鐘就看到「這堂課跟我有關」。

點名兩個人問:你的題目在哪一列?
-->

---
layout: section
class: sec-ruler
---

# ② 逼近,需要一個評估基準

## 這個基準不只一種,而且不對稱

<!--
12–50 分,本堂最重的一段。

紀律提醒(給我自己):散度要當成一個問題的答案來講,別當定義念。每一個性質講完後兩分鐘內,必須落地成一個學生見過的具體現象。
-->

---

# 生成 = 從 $p(x)$ 取樣,但 $p$ 不能建表

<div class="grid grid-cols-2 gap-6 items-center mt-6">

<div>

一張 $512\times512$ 的 RGB 影像:

<div class="text-center my-5 p-4 rounded border border-violet-400">
<div class="text-3xl" style="font-family: var(--mono); color: var(--violet)">256<sup>786432</sup></div>
<div class="text-xs opacity-60 mt-2">種可能的取值</div>
</div>

宇宙的原子數約 $10^{80}$。這張表**不可能存在**。

</div>

<div v-click>

所以我們不存表,改成學一個參數化的 $p_\theta$,讓它逼近 $p_{\text{data}}$。

<div class="mt-5 p-3 border-l-4 border-amber-400 text-sm">

**「逼近」是什麼意思?**

兩個分布之間的差異,必須先定義一個評估基準才能量化。

</div>

<div class="mt-4 text-sm opacity-80">
而這樣的基準不只一種,而且<b>不對稱</b>。
</div>

</div>

</div>

<!--
這頁 2 分鐘。重點只有一句:因為建表不可能,所以生成問題被迫變成「分布逼近」問題;
而分布逼近問題被迫要選一個散度。

散度的選擇是這門課唯一的主角。
-->

---

# KL 散度:積分的權重決定了它懲罰什麼

$$\mathrm{KL}(p\|q)=\int p(x)\,\log\frac{p(x)}{q(x)}\,dx$$

<div v-click class="mt-4 text-center text-lg">

權重是 <span style="color: var(--violet)"><katex-elem expr="p" /></span>,所以它**只在 $p$ 有質量的地方施加懲罰**。

</div>

<v-clicks>

<div class="grid grid-cols-2 gap-5 mt-5 text-sm">

<div class="p-4 rounded border border-cyan-400">

### Forward KL · $\mathrm{KL}(p_{\text{data}}\|p_\theta)$

$p>0$ 而 $q\to 0$ 的地方,$\log\frac{p}{q}\to+\infty$

→ $q$ **必須蓋住 $p$ 的全部 support**

<div class="mt-2" style="color: var(--cyan)">zero-avoiding · mode-covering</div>

</div>

<div class="p-4 rounded border border-pink-400">

### Reverse KL · $\mathrm{KL}(p_\theta\|p_{\text{data}})$

權重換成 $q$,$q=0$ 的地方整項歸零

→ $p$ 的其他模式**被忽略也不受懲罰**

<div class="mt-2" style="color: var(--pink)">zero-forcing · mode-seeking</div>

</div>

</div>

</v-clicks>

<!--
請學生盯著積分的權重看。不對稱就寫在式子的第一個符號上,不用另外當成一條「性質」去背。

一句話總結:forward KL 問「資料出現的地方,你有沒有給機率?」;reverse KL 問「你放機率的地方,資料真的在嗎?」
-->

---

# 配圖 B-1 · 用單一高斯 $q$ 擬合雙峰 $p$

<div class="text-xs opacity-60 mb-1">下面的曲線是數值最小化各自的散度解出來的,不是示意圖。</div>

<DivergenceFit :w="0.5" :curves="['forward']" annotate="forward" />

<div v-click class="mt-3 p-3 border-l-4 border-cyan-400 text-sm">
forward KL 會把可觀的機率質量配置在<b>資料實際不存在的谷底</b>,以換取不在任何一個峰上取 0。<br>
<span class="opacity-70">「糊」就是這樣來的。base model 那句「這取決於很多因素」也是。</span>
</div>

<!--
解出來的參數:μ=0, σ=1.70。兩個真峰的寬度只有 0.55。

這裡停 30 秒讓學生看那座橋。橋上沒有任何真實資料,但 forward KL 逼它必須存在。
-->

---

# 換一個散度,同一個 $q$ 的最佳解就換了位置

<DivergenceFit :w="0.5" :curves="['forward', 'reverse']" annotate="reverse" />

<div v-click class="mt-3 p-3 border-l-4 border-pink-400 text-sm">
reverse KL 的解直接放棄左峰,收縮到右峰內部:<b>銳利,但漏掉一半的 support。</b><br>
<span class="opacity-70">對齊後模型「十次取樣、十個一樣」,就是這麼來的。</span>
</div>

<div v-click class="mt-2 text-center text-base">

同一個模型族、同一份資料,**只換了目標函數**,得到兩個完全相反的失效模式。

</div>

<!--
到這裡論題的一半已經證完了:hedging 與塌陷不是兩種病,是同一個選擇的兩個方向。

參數:forward μ=0 σ=1.70 / reverse μ=1.60 σ=0.58。
-->

---

# JSD:澄清一 · 它不是「對稱化的 KL」

<div class="grid grid-cols-2 gap-6 text-sm mt-3">

<div class="p-4 rounded border border-pink-400">

### Jeffreys(真的對稱化)

$$\mathrm{KL}(p\|q)+\mathrm{KL}(q\|p)$$

繼承**兩邊**的無限大。<br>
support 不重疊 → $\infty$,且梯度不帶任何方向資訊。

</div>

<div class="p-4 rounded border border-amber-400">

### JSD(換掉分母)

$$\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m)$$
$$m=\tfrac{p+q}{2}$$

分母是混合分布,$m\ge\tfrac12 p$ 且 $m\ge\tfrac12 q$

</div>

</div>

<div v-click class="mt-5 text-center">

於是 **恆為有界**:$\;0\le\mathrm{JSD}(p\|q)\le\log 2$,而且 $\sqrt{\mathrm{JSD}}$ 滿足度量公理,包含三角不等式。

</div>

<!--
關鍵在「分母換成 m」這個動作:log(p/m) 最大只能是 log 2,因為 m 至少是 p 的一半。

有界聽起來是優點。下一頁說明它為什麼同時是災難。
-->

---

# JSD:澄清二 · 有界 = 會飽和 = 梯度消失

<JsdSaturate />

<div v-click class="mt-2 p-3 border-l-4 border-amber-400 text-sm">
兩個分布一旦分開到 support 幾乎不重疊,JSD 就貼上 <katex-elem expr="\log 2" /> <b>不再變動</b>:曲線變平,梯度趨近 0。<br>
<span class="opacity-70">「你離目標很遠」與「你離目標非常遠」給出同一個 loss,模型就不知道該往哪走。</span>
</div>

<div v-click class="mt-2 text-xs opacity-70">
這正是原始 GAN 的病理,也是 non-saturating loss 與 WGAN 的動機。④ 會回到這一點。
</div>

<!--
現場拖那個滑桿。從 d=0 拖到 d=6:KL 一路飆到 27,JSD 停在 0.693 動也不動。

讓學生自己說出「這樣沒辦法訓練」。
-->

---

# JSD:澄清三 · 它就是「一個最佳分類器能多會分」

<div class="text-sm mt-3">

給定 $p$ 與 $q$,最佳判別器是

$$D^*(x)=\frac{p(x)}{p(x)+q(x)}$$

代回 GAN 的目標,得到

$$V(D^*,G)=2\,\mathrm{JSD}(p\|q)-2\log 2$$

</div>

<div v-click class="mt-5 p-3 border-l-4 border-violet-400 text-sm">

所以 JSD 度量的是:**最佳判別器分辨樣本來源時,能取得多大的優勢。**

形式上它等於 $I(X;Z)$,其中 $Z\sim\text{Bernoulli}(1/2)$ 是「來自 $p$ 還是 $q$」的標籤。

</div>

<div v-click class="mt-3 text-center text-sm opacity-80">
分不出來 → 互資訊 0 → JSD 0。<b>「像不像」被翻譯成了「猜不猜得到」。</b>
</div>

<!--
這一頁是 ③ 的伏筆:判別器就是 JSD 的操作型定義,不是 GAN 的裝飾品。

$D^*$ 的推導只需要對 a·log y + b·log(1−y) 求極值,y* = a/(a+b),一行就完成,可以請學生口頭推。

如果時間緊,I(X;Z) 那句可以只提一次不展開。
-->

---

# 澄清四(加碼)· 「JSD 在中間」到底是什麼意思?

<div class="text-xs opacity-60 mb-1">把左峰的權重從 50% 降到 30%,三個散度各重解一次:</div>

<DivergenceFit :w="0.3" :curves="['forward', 'jsd', 'reverse']" />

<div v-click class="mt-3 p-3 border-l-4 border-amber-400 text-sm">
兩峰一旦不對稱,<b>JSD 的解會倒向大峰</b>:它放棄小峰,跟 reverse KL 收斂到同一個地方。<br>
而 forward KL 從不倒向任一側,reverse KL 一律倒向單側。
</div>

<div v-click class="mt-2 text-center text-sm">

「在中間」不等於「取平均」,而是<b>「依權重倒向其中一側」</b>。

</div>

<!--
這頁是本次備課實際算出來的,不是教科書上的圖。w=0.5 時 JSD 解 μ=0 σ=1.66(covering);
w=0.3 時直接跳到 μ=1.60 σ=0.58,與 reverse KL 完全相同。

教學價值:讓「折衷」這個詞變得可證偽。也順帶說明為什麼 GAN 的行為比 VAE 難預測。
-->

---
layout: center
class: text-center
---

<div class="text-2xl px-10 leading-relaxed">

選擇散度 = **選擇你願意犯哪一種錯**。

</div>

<div class="text-base opacity-70 mt-6">沒有中立的散度。</div>

<div v-click class="mt-8 text-sm">
  <SpectrumAxis :rows="1" />
</div>

<!--
本段核心訊息。寫在黑板上,整堂課不要擦掉。

現在軸的最上面一列可以填了,但先只填「訓練目標」這一列。
-->

---

# 三個散度的對照表

<div class="text-sm">

| | Forward KL | JSD | Reverse KL |
|---|---|---|---|
| 行為 | mode-covering | 中間,但會飽和 | mode-seeking |
| 對稱 | 否 | 是 | 否 |
| 上界 | 無 | $\log 2$ | 無 |
| **失效模式** | 過度平滑、hedging、覆蓋過廣 | 梯度消失或震盪 | 模式塌縮、多樣性喪失 |
| 你在哪裡看過 | base model 的空泛回答 | GAN 訓練不收斂 | RLHF 後的樣板化回覆 |

</div>

<div class="mt-5 grid grid-cols-2 gap-4 text-sm">

<div class="p-3 border-l-4 border-cyan-400">
<b>互動 demo</b><br>
<a href="/demos/divergence-2d-interactive.html" target="_blank">散度的選擇 · 2D 互動版</a><br>
<span class="text-xs opacity-70">三個散度同時量測,但只優化其中一個</span>
</div>

<div class="p-3 border-l-4 border-violet-400">
<b>課後練習(不佔課堂時間)</b><br>
用單一高斯擬合 1D 雙峰混合,分別最小化三個散度,數值求解 <katex-elem expr="\mu,\sigma" />。<br>
<span class="text-xs opacity-70">做過一次就不會再搞混。</span>
</div>

</div>

<!--
最後一列「你在哪裡看過」是這張表的重點,前四列只是把剛才講的收起來。

課後練習其實就是今天那張 B-1 的產生方式,鼓勵他們自己重跑一次。
-->

---
layout: section
class: sec-compute
---

# ③ 散度能不能算,決定它變成哪一個家族

## 全課骨架

<!--
50–80 分。

這段講完,學生對 AR / VAE / GAN / DPM 之間的關係應該就不會再亂了。
-->

---

# 一個散度要能算,你得先拿得到兩樣東西

<div class="grid grid-cols-2 gap-6 mt-6 text-sm">

<div class="p-4 rounded border border-cyan-400">

### 取樣 (sampling)

從某個分布抽出樣本 $x\sim p$

<div class="mt-3 text-xs opacity-75">
<katex-elem expr="p_{\text{data}}" />:有資料集,✓<br>
<katex-elem expr="p_\theta" />:模型自己跑一次,✓
</div>

</div>

<div class="p-4 rounded border border-pink-400">

### 密度 (density)

給定 $x$,算出 $p(x)$ 這個**數值**

<div class="mt-3 text-xs opacity-75">
<katex-elem expr="p_\theta" />:likelihood-based 模型算得出,✓<br>
<katex-elem expr="p_{\text{data}}" />:<b style="color: var(--pink)">永遠拿不到 ✗</b>
</div>

</div>

</div>

<div v-click class="mt-6 text-center text-base">

$\log p_{\text{data}}(x)$ 拿不到。**整個生成模型的分類法,都是從這一個缺口推導出來的。**

</div>

<!--
先讓學生確認這四格。特別是右下角:我們有一大堆從 p_data 抽出來的樣本,
但我們永遠不知道任何一個樣本「本來應該有多大機率」。

接下來一頁,把三個散度各自往這四格上一放,家族就自己掉出來了。
-->

---

# 於是三個散度,長成三個家族

<ComputeMap />

<div v-click class="mt-5 p-3 border-l-4 border-amber-400 text-base">
GAN 的判別器<b>不是設計上的巧思,是計算上的必然</b>。<br>
<span class="text-sm opacity-75">JSD 兩邊的密度都拿不到,只剩下「訓一個分類器去逼近那個比值」這條路。</span>
</div>

<!--
本堂最該被記住的一頁。

講法:先蓋住「由誰代理」與「於是變成」兩欄,只念前兩欄,讓學生自己推。多數人推得出 GAN 那一列。

推不出來也沒關係,重點是他們親身體驗到「這是被逼出來的,不是被發明出來的」。
-->

---

# 完整的分類樹:分岔點只有一個問題

<FamilyTree />

<div v-click class="mt-2 p-3 border-l-4 border-cyan-400 text-sm">
Goodfellow 那棵樹之所以是這個形狀,不是因為有人想把模型分成六類,而是因為
<b>「<katex-elem expr="p_\theta(x)" /> 寫不寫得出來」這一問只有三種答案</b>:寫得出、寫不出但有下界、完全寫不出。
</div>

<!--
帶讀:
- 顯式可精確:AR 用鏈鎖法則拆成 ∏p(xᵢ|x_<ᵢ);Flow 用可逆變換加 change of variables。
  代價是架構被密度可算性綁架(必須自迴歸、或必須可逆)。
- 顯式近似:積分算不動,退而最佳化一個下界(VAE)或一整條變分過程(diffusion)。
- 隱式:放棄寫密度,只造取樣機,用另一個網路當評估基準。

出處:Goodfellow, NIPS 2016 Tutorial (arXiv:1701.00160)。
-->

---

# 五個家族的體質對照

<FamilyMatrix />

<div v-click class="mt-3 text-sm opacity-80">
沒有一個家族五項全滿。<b>短板的位置,就是那個家族選定的散度的性質。</b>
虛線格代表<b>該欄的高低是框架內的設計選擇</b>,不是家族的體質。
</div>

<!--
刻意用三格量表而不是形容詞。「偏糊」「銳利」「極高」放在同一張表裡沒有可比性,點數有。

Diffusion 與 Flow Matching 放同一列,不是為了省空間。它們是同一個框架:
學一個時間相關的向量場(或等價的 score),把簡單先驗沿一條機率路徑搬到資料分布。
Song et al. 的 probability-flow ODE 說明 diffusion 的 SDE 有同 marginal 的 ODE 形式;
Lipman et al. 進一步證明 diffusion 用的路徑只是 conditional flow matching 的一個特例。
所以「取樣速度」那一格畫成虛線區間:步數取決於路徑與取樣器,不是取決於你叫它哪個名字。

提問:「AR 又穩、密度又精確,為什麼大家還要搞別的?」
答:取樣是序列式的,一張 512×512 影像等於 26 萬次前向;而且沒有天然的潛在空間可以編輯。

30 秒掃過即可,重點在下一頁的三角。
-->

---

# 為什麼短板一定存在:生成三難

<Trilemma/>

<div class="mt-3 text-center text-sm">
黃色箭頭是重點:<b>三角上的位置不是永久的</b>。latent diffusion、蒸餾、rectified flow 都在沿這條邊往「取樣速度」推。
</div>

<!--
Xiao, Kreis & Vahdat, "Tackling the Generative Learning Trilemma with Denoising Diffusion GANs",
ICLR 2022 (arXiv:2112.07804)。

這頁的教學功能是把「取捨」從一句安慰話變成一個結構性陳述:
你選的散度決定你坐在哪一條邊上,而不是你的網路不夠大。

近年幾乎所有生成模型的工程進展,都可以問一句:「它在攻這個三角的哪一邊?」
latent diffusion 在低維空間跑、蒸餾把步數壓下來、rectified flow 把路徑拉直,
這三個都是把同一個點沿「品質—覆蓋」這條邊往速度方向推;diffusion-GAN 混血則同時攻速度與覆蓋。
-->

---
layout: center
---

<div class="px-8">

<div class="eyebrow mb-4">替下堂課鋪路</div>

<div class="text-lg leading-relaxed">

reverse KL 那一列的「**reward 代理**」,<br>
與 GAN 那一列的「**判別器代理**」,<br>
其實是同一件事的兩種說法:

</div>

<div v-click class="mt-6 p-4 border-l-4 border-violet-400 text-lg">
兩者都在補同一個拿不到的東西:<katex-elem expr="\log p_{\text{data}}" />。
</div>

<div v-click class="mt-5 text-sm opacity-70">
下堂課的 capstone(DDO)就站在這句話上:如果 likelihood-based 模型本來就算得出 <katex-elem expr="\log p_\theta" />,那判別器其實可以不用另外蓋一個網路。
</div>

</div>

<!--
30 秒,講完就走,不展開。

這是刻意留的懸念,下堂課回收。
-->

---
layout: section
class: sec-family
---

# ④ 家族巡禮

## 每個家族的典型失效,都來自它選定的訓練目標

<!--
80–150 分。AR 25 / VAE 20 / GAN 20 / DPM 5。

紀律:每個模型只講「它選了哪個散度」與「那個選擇帶來什麼失效模式」。
架構細節只在它與散度有因果關係時才講。
-->

---

# AR / LLM:它就是 forward KL 的直接產物

<div class="mt-4">

$$\log p(x)=\sum_t \log p(x_t \mid x_{\lt t})$$

</div>

<div v-click class="mt-6">

### CE 與 KL 的關係

$$H(p,q)=-\mathbb{E}_{x\sim p}[\log q(x)]=\underbrace{H(p)}_{\text{與 }\theta\text{ 無關}}+\mathrm{KL}(p\|q)$$

</div>

<div v-click class="mt-5 p-3 border-l-4 border-cyan-400 text-sm">
<b>分類任務</b>:目標是 one-hot,<katex-elem expr="H(p)=0" />,所以 cross-entropy <b>就是</b> forward KL,兩者恆等。<br>
一旦目標變軟(label smoothing、知識蒸餾),兩者才分家。
</div>

<!--
你們每天在跑的 next-token CE loss,不是「一個常用的 loss」,它是 forward KL 本人。

所以第 ② 段講的所有 mode-covering 性質,全部原封不動適用於你手上的 LLM。這句話要講重一點。
-->

---

# $H(p)$ 消不掉,而這正是 ③ 那個缺口的另一面

$$H(p)=-\mathbb{E}_{x\sim p}[\log p(x)]$$

<div v-click class="mt-3 text-sm">

$\log$ 裡面是 $p$ **自己**。我們只有樣本,沒有密度 → **無法蒙地卡羅估計**。

<div class="mt-2 opacity-75">所以 CE 的絕對數值沒有意義,只有差值有意義。</div>

</div>

<div v-click class="mt-5">

### 四條實務路徑

<div class="grid grid-cols-2 gap-3 text-sm mt-2">
<div class="p-3 rounded border border-gray-500">① 讓它成為<b>共享常數</b>:只在同一份資料上比差值</div>
<div class="p-3 rounded border border-gray-500">② 相對於<b>參考模型</b>正規化 → <katex-elem expr="\log\frac{p_\theta}{p_{\text{ref}}}" /></div>
<div class="p-3 rounded border border-gray-500">③ 用已知 <katex-elem expr="H(p)" /> 的<b>合成資料</b></div>
<div class="p-3 rounded border border-gray-500">④ 繞開 likelihood:MAUVE / MMD / 下游指標</div>
</div>

</div>

<div v-click class="mt-3 text-xs opacity-70">
② 這條路下堂課會變成主角:那個量叫 likelihood ratio,它天生扣掉了「這筆資料本身有多難」。
</div>

<!--
confidence vs. accuracy 那組請特別記 ②:你拿 sequence log-prob 當信心,
其實混進了「這句話本身有多罕見」。用 reference model 相除就能扣掉。

這是下堂課最後那個現成 research idea 的起點。
-->

---

# 跨 tokenizer 比較:bits per byte

<div class="text-sm">

鏈鎖法則會 **telescope**:字串的 log-likelihood 與你怎麼切 token 無關,會變的只有分母。

</div>

$$\mathrm{BPB}=\frac{T}{N_{\text{bytes}}}\cdot\log_2 \mathrm{PPL}_{\text{token}}$$

<div v-click class="mt-5">

### 三個必查 caveat

<div class="text-sm">

| | 要檢查什麼 |
|---|---|
| **無損** | `decode(encode(s)) == s` 必須成立 |
| **上界** | 算的是 canonical 切法,是真實字串機率的**上界**;不同 tokenizer 鬆緊不同 |
| **一致** | 固定 chunking、stride 與 BOS 處理 |

</div>

</div>

<div v-click class="mt-3 text-xs opacity-70">
下堂課 ⑤ 回收:跨模型比較 perplexity 幾乎都是錯的,除非分母換成 bytes。
</div>

<!--
LLM-ASR 組會直接用到:換 LM 之後,LM weight 這個超參數要能移植,尺度就得統一。

不要在這頁停太久,3 分鐘。
-->

---

# 鏈鎖分解:請盯著期望值的下標

$$\mathrm{KL}(p\|q)=\sum_t \mathbb{E}_{x_{\lt t}\sim \textcolor{#ff6b9d}{p}}\Big[\mathrm{KL}\big(p(\cdot\mid x_{\lt t})\,\big\|\,q(\cdot\mid x_{\lt t})\big)\Big]$$

<div v-click class="mt-6 text-center text-lg">

前綴 <katex-elem expr="x_{<t}" /> 取自 <b style="color: var(--pink)"><katex-elem expr="p" /></b>,**不是** $q$。

</div>

<div v-click class="mt-6 text-sm opacity-80 text-center">
這一個下標,決定了 AR 模型的訓練方式與它的招牌 bug。整個下堂課也是從這裡開始。
</div>

<!--
這頁只有一行式子,刻意的。讓他們盯著那個下標看 30 秒。

然後翻頁,一次收三個結論。
-->

---

# 配圖 B-2 · 兩條軌道

<TwoTracks />

<div v-click class="mt-2 p-3 border-l-4 border-pink-400 text-sm">
訓練時,每一步的前綴都是<b>從語料抓來的正確前綴</b>;推論時,前綴是<b>模型自己剛剛生出來的</b>。<br>
兩條軌道從第二步開始就分開了,而 loss 只認得上面那條。
</div>

<!--
這張圖下堂課 DDO 段會原樣回放。請他們拍照。

問學生:如果你想讓下面那條軌道也被測量,你會怎麼做?先不要回答,下堂課講。
-->

---

# 一個下標,三個結論

<v-clicks>

<div class="p-4 rounded border border-cyan-400 text-sm mb-3">

### 1 · teacher forcing 不是訓練技巧

它是 forward KL 分解式的**字面實作**。式子裡寫了前綴取自 $p$,你就只能餵真實前綴。

</div>

<div class="p-4 rounded border border-pink-400 text-sm mb-3">

### 2 · exposure bias 就在同一行式子裡

推論時前綴來自 <katex-elem expr="q" />,但訓練目標**從未測量過任何模型自己生成的前綴**。<br>
→ 訓練與推論優化的不是同一個泛函 → **memory agent 的長對話漂移**

</div>

<div class="p-4 rounded border border-violet-400 text-sm">

### 3 · 把下標從 $p$ 換成 $q$,就得到 reverse KL 的分解

scheduled sampling / RL / DPO / DDO 的共同點**不是「用了 RL」**,<br>
而是 **把 loss 搬到了下方那條軌道上。**

</div>

</v-clicks>

<!--
第 2 點直接對應 memory agent 組的問題:長對話漂移不是 context window 的問題,
是訓練目標從來沒看過自己生成的長前綴。

第 3 點是整個下堂課的入口,講完停 15 秒。
-->

---

# forward KL 在 token 層級的實作機制

<TokenBars />

<div v-click class="mt-1 text-sm">

語料裡每個前綴通常**只有一個續寫**,所以每個位置的目標是 one-hot,
一個高變異但**無偏**的估計。模型仍能收斂到條件分布,因為 CE 是 **proper scoring rule**。

</div>

<!--
「只有一個續寫卻能學到分布」這件事學生常卡住:因為同一個前綴在整份語料裡會出現很多次,
不同次的續寫不同,平均起來就是條件分布。

proper scoring rule 一句話帶過:最小化期望 CE 的唯一解就是真實條件機率。
-->

---
layout: center
---

<div class="px-6">

<div class="eyebrow mb-3">往下推一步 · 虛假前提檢測</div>

<div class="text-xl leading-relaxed">

用 forward KL 訓練的模型,<br>
**結構上就沒有「拒絕回答」這個選項**,<br>
除非後訓練另外教它。

</div>

<div v-click class="mt-7 p-4 border-l-4 border-amber-400 text-base">

$p(y\mid x)$ **良好定義**,與 $x$ **值得被回答**,是兩件完全不同的事。

<div class="text-sm opacity-75 mt-2">
「玉山的第三個火山口叫什麼?」<katex-elem expr="p(x)\approx 0" />,但 <katex-elem expr="p(y\mid x)" /> 照樣算得出來,而且模型從沒被教過在這裡要停。
</div>

</div>

</div>

<!--
虛假前提組的理論基礎就是這一頁。

配套閱讀:Kalai et al. (2025) Why Language Models Hallucinate;(QA)² benchmark 建議實際跑一次。
-->

---

# AR 段的兩個實務陷阱

<div class="grid grid-cols-2 gap-5 mt-5 text-sm">

<div class="p-4 rounded border border-pink-400">

### tokenization 把機率切碎

同一個語意,不同切法 → 不同的 token 序列 → 不同的 sequence probability

<div class="mt-3 opacity-75">
<b>confidence 估計最常踩的坑。</b><br>
下堂課的 semantic entropy 就是為了繞開它。
</div>

</div>

<div class="p-4 rounded border border-amber-400">

### 序列 likelihood 的長度偏誤

$\log p(x)$ 是**負值累加**,句子越長分數越低

<div class="mt-3 opacity-75">
直接拿來排序 → 永遠選最短的。<br>
需要長度正規化,而正規化方式本身就是一個設計選擇。
</div>

</div>

</div>

<div v-click class="mt-5 text-center text-sm opacity-80">
兩個陷阱的共同點:<b>你以為你在量「模型多有把握」,其實你量到的是「這串 token 有多長、有多罕見」。</b>
</div>

<!--
confidence vs. accuracy 那組現在應該已經知道自己第一步要做什麼了:先把量測工具修好,再談相關性。
-->

---
layout: center
class: text-center
---

<div class="eyebrow">④-B</div>

# VAE

<div class="text-lg opacity-80 mt-3">同一個 forward KL,但邊際化不可解,只能退而最佳化一個下界</div>

<!--
100–120 分。

VAE 這一段的重點不是架構,是「邊際化算不動」這個計算事實怎麼一路決定了它的所有毛病。
-->

---

# 動機:潛在變數模型,與它的邊際化困難

<div class="text-sm">

假設資料 $x$ 由看不見的低維因子 $z$ 生成:

</div>

$$p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz, \qquad p(z) = \mathcal{N}(0, I)$$

<v-clicks>

<div class="grid grid-cols-2 gap-5 mt-4 text-sm">

<div class="p-3 rounded border border-cyan-400">
<b>取樣方向:容易</b><br>
<span class="opacity-75">抽 <katex-elem expr="z \sim p(z)" />,過 decoder,完成。</span>
</div>

<div class="p-3 rounded border border-pink-400">
<b>訓練方向:困難</b><br>
<span class="opacity-75">MLE 需要 <katex-elem expr="\log p_\theta(x)" />,但這個積分在 <katex-elem expr="z" /> 連續、decoder 非線性時 intractable。</span>
</div>

</div>

<div class="mt-4 p-3 border-l-4 border-amber-400 text-sm">
樸素蒙地卡羅無效:<katex-elem expr="p_\theta(x)\approx\frac1K\sum_k p_\theta(x\mid z_k),\ z_k\sim p(z)" />。
高維時幾乎所有 <katex-elem expr="z_k" /> 都與這個 <katex-elem expr="x" /> 無關,估計量的變異數過大。
</div>

</v-clicks>

<div v-click class="mt-3 text-center text-base">

**出路**:另外訓練一個推斷網路 <katex-elem expr="q_\phi(z\mid x)" />,直接指出哪些 $z$ 有可能生成這個 $x$。

</div>

<!--
強調「取樣方向容易、推斷方向困難」的不對稱。

encoder 不是為了壓縮而生,是為了讓 MLE 變得可算而生。這個因果順序講清楚,
學生才不會把 VAE 當成「autoencoder 加了噪聲」。
-->

---

# ELBO:一行 Jensen,換一個可訓練的下界

$$
\begin{aligned}
\log p_\theta(x) &= \log\, \mathbb{E}_{q_\phi(z\mid x)}\!\left[\frac{p_\theta(x\mid z)\,p(z)}{q_\phi(z\mid x)}\right]
\;\ge\; \mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{p_\theta(x\mid z)\,p(z)}{q_\phi(z\mid x)}\right] \\[4pt]
&= \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{重建項}} \;-\; \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{正則項}} \;=\; \mathrm{ELBO}
\end{aligned}
$$

<div v-click class="mt-4 p-3 border-l-4 border-violet-400 text-sm">

更有用的是等式形式:

$$\log p_\theta(x) = \mathrm{ELBO} + \mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)$$

間隙**恰好等於 $q_\phi$ 對真後驗的 KL**。$q$ 越接近真後驗,下界越緊。

</div>

<!--
Jensen 那一步放大寫一次,請一位學生說出為什麼不等號方向是 ≥
(log 凹 → E[log] ≤ log E,取在這個形式後成為下界)。這是先修內容,要能當場答出來。

第二個式子是很多教材略過、但概念上最重要的一條。下一頁把它畫出來。
-->

---

# 下界有多鬆?間隙是可以指認的

<ElboGap />

<div v-click class="mt-2 p-3 border-l-4 border-pink-400 text-sm">
你優化的是 ELBO,不是 <katex-elem expr="\log p_\theta(x)" />。而<b>間隙的大小取決於你選的 <katex-elem expr="q" /> 家族</b>,不取決於 decoder 有多大。<br>
<span class="opacity-70">所以 VAE 報出來的 likelihood 一律是保守估計,拿去跟 AR 的精確 likelihood 比是不公平的。</span>
</div>

<!--
這頁對做評估的人特別重要:VAE 的 test log-likelihood 是下界,AR 的是精確值,
兩者直接比大小沒有意義。

想收緊間隙的方向:IWAE(多樣本下界)、normalizing flow posterior、更深的 encoder。
不必展開,提名字即可。
-->

---

# Reparameterization:把隨機性搬到計算圖外面

<ReparamGraph />

<div v-click class="mt-2 text-sm">

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon,\quad \varepsilon \sim \mathcal{N}(0, I)$$

</div>

<!--
一句話總結:不是對骰子微分,而是把骰子丟到模型外面,模型只負責平移和縮放骰子的結果。

高斯 q 對高斯先驗的 KL 有閉式解,所以整個 ELBO 端到端可微。

延伸(有人問再說):離散 z 不能這樣做 → Gumbel-softmax 或 VQ-VAE 的 straight-through。
這是等一下 VQ-VAE 的伏筆。
-->

---

# β-VAE:把 ELBO 的拉鋸變成一顆旋鈕

$$\mathcal{L}_{\beta} = \underbrace{\mathbb{E}_{q_\phi}\big[\log p_\theta(x\mid z)\big]}_{\text{重建}} \;-\; \beta \cdot \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{正則}}$$

<div class="text-sm mt-4">

| $\beta$ | 行為 | 代價 |
|---|---|---|
| $\beta = 0$ | 退化成普通 autoencoder,重建極準 | 潛在空間四散;從先驗取樣落在沒學過的區域,**生成崩壞** |
| $\beta = 1$ | 標準 VAE(正宗 ELBO) | 重建與生成的折衷 |
| $\beta \gg 1$ | 潛在空間被硬壓成 $\mathcal{N}(0,I)$ | 重建劣化、樣本糊成一團;**posterior collapse** 前兆 |

</div>

<div class="mt-4 text-sm">
<a href="/demos/vae-2d-interactive.html" target="_blank">Demo · β-VAE 2D</a>:現場轉這顆旋鈕,看圓環拓撲蓋不乾淨、以及 β 兩端各自怎麼崩壞。
</div>

<!--
β-VAE 原始動機是 disentanglement(Higgins et al., ICLR 2017),
但教學上它是把 ELBO 兩項拉鋸可視化的最好工具。

順帶一提:這個 β 與 RLHF 目標裡那個 β 是同一個角色:正則強度,把新分布拉向參考分布。
⑤ 會正式回收。
-->

---

# VAE 的四個缺陷,各自來自 ELBO 的某一項

<VaeDefects />

<!--
講法:先指式子,再指缺陷。不要讓學生把這四個當成四件獨立的事背下來。

缺陷 ① 的兩層成因要講清楚:
- 損失層:高斯 likelihood 等價 MSE,一對多時最優解是平均。平均臉沒有毛孔。
- 路線層:forward KL 本來就 mode-covering,罰漏不罰糊。
demo 裡高斯混合中間那座「橋」是兩股力的合成。

posterior collapse 的語音錨點:用 autoregressive decoder 做 VAE 時,decoder 自己就能把資料
建得很好,於是直接忽略 z,KL 項歸零、z 失效。這是語音表示學習的實際痛點。
-->

---

# VAE 的後續改進:每一項都對準一個缺陷

<div class="text-xs">

| 缺陷 | 改進方向 | 代表工作 | 一句話 |
|---|---|---|---|
| ① 樣本糊 | 離散潛在空間 + 強 decoder | **VQ-VAE** (2017) / **VQ-VAE-2** (2019) | $z$ 改成 codebook 查表,生成交給 AR prior;催生 DALL·E 路線 |
| ① 樣本糊 | 疊多層潛在變數 | **NVAE** (2020)、**VDVAE** (2021) | hierarchical VAE:多層 $z$ 逐級細化 |
| ① 樣本糊 | 借判別器的感知基準 | **VAE-GAN** (2016) | 重建項換成對抗式的 feature-level loss |
| ④ prior hole | 學出來的先驗 | **VampPrior** (2018) | 先驗改成 pseudo-input 的 posterior 混合,把洞填掉 |
| ③ collapse | KL annealing / free bits | (訓練技巧系) | 前期關小 KL,或給每一維 KL 保底 |
| ② 間隙 | 更緊的下界 / 更靈活的 $q$ | **IWAE** (2015) | 多樣本重要性加權,間隙隨樣本數收斂 |

</div>

<div v-click class="mt-3 p-3 border-l-4 border-cyan-400 text-sm">
<b>VAE 的現代角色</b>:Stable Diffusion 的第一層就是一個對抗式訓練的 VAE,先把影像壓進 latent space,
diffusion 才在低維空間跑。這條研究線沒有消失,它變成了別人的第一層。
</div>

<!--
不必逐列細講,挑兩個各講 30 秒:

- VQ-VAE:把「連續 z + 高斯先驗」整組換成「離散 codebook + 學出來的 AR 先驗」,
  一次繞開糊(likelihood 交給強 decoder 與 prior)與 prior hole(先驗是學的)。
- NVAE / VDVAE:證明純 VAE 認真做架構也能生成高解析人臉;更重要的是 hierarchical 這個形狀
  就是 diffusion 的前身,下堂課的伏筆。

SD 的 VAE 其實是 VAE-GAN 混合(KL 正則 + patch 判別器 + LPIPS),正好呼應本堂兩條路線的合流。
-->

---
layout: center
class: text-center
---

<div class="eyebrow">④-C</div>

# GAN

<div class="text-lg opacity-80 mt-3">兩邊的密度都拿不到,只好把評估基準本身訓練出來</div>

<!--
120–140 分。
-->

---

# 核心想法:不寫密度,改訓練一個評估基準

<AdversarialLoop />

<div v-click class="mt-2 text-sm">

$$\min_G \max_D \; \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p(z)}[\log(1 - D(G(z)))]$$

</div>

<!--
關鍵對比句:VAE 用一把固定的基準(likelihood / MSE);GAN 把基準本身也做成神經網路,
跟著資料一起學。

基準會學到人眼在意的特徵(紋理、銳利度),這是 GAN 樣本銳利的來源;
基準會動,這是 GAN 不穩的來源。一體兩面。

實作用 non-saturating loss:G 改成最大化 log D(G(z))。原因下一頁講。
-->

---

# GAN:JSD 是理論,實務上用的不是 JSD

<div class="text-sm mt-2">

$V=2\,\mathrm{JSD}-2\log 2$ **只在最佳判別器下成立**,而那正是梯度消失的地方(② 那張滑桿圖)。

改用 non-saturating loss 之後,生成器實際在最小化的是

</div>

$$\mathrm{KL}(p_g \| p_{\text{data}}) \;-\; 2\,\mathrm{JSD}(p_g \| p_{\text{data}})$$

<div v-click class="grid grid-cols-2 gap-4 mt-4 text-sm">

<div class="p-3 rounded border border-pink-400">
第一項是 <b>reverse KL</b><br>
<span class="opacity-75">→ mode-seeking 的來源</span>
</div>

<div class="p-3 rounded border border-amber-400">
第二項帶<b>負號</b><br>
<span class="opacity-75">→ 訓練不穩的來源</span>
</div>

</div>

<div v-click class="mt-4 p-3 border-l-4 border-violet-400 text-sm">
為了修梯度消失而換掉的那個 loss,<b>同時也換掉了 mode-covering。</b><br>
<span class="opacity-70">GAN 的兩種 loss,分別坐在兩種失效模式上。</span>
</div>

<!--
Arjovsky & Bottou (2017) 是這頁的依據。

這頁是整個 ④ 段最能展示「目標函數決定失效模式」的一頁:換 loss 就是換你願意犯的錯。
-->

---

# mode collapse:第一層原因就足以解釋

<v-clicks>

<div class="p-4 rounded border border-pink-400 text-sm mb-3">

### 1 · 生成器的 loss 裡根本沒有資料

$\mathbb{E}_{p_{\text{data}}}[\log D(x)]$ 這一項**不含 $G$**。而 $D(x)$ 是**逐點**判斷:
「覆蓋度」是分布層級的性質,**判別器的介面裡沒有這個欄位。**

</div>

<div class="text-sm opacity-80">

其餘三層(不必全講):

2. non-saturating loss 本身已含 reverse KL(上一頁)
3. min-max 沒有位能函數 → 繞圈 → mode hopping;$D$ 會遺忘,塌縮後缺乏回復機制
4. 一步生成的幾何代價:只蓋單一模式的映射既平滑又便宜,塌縮是低成本解

</div>

</v-clicks>

<div v-click class="mt-5 p-3 border-l-4 border-amber-400 text-sm">
下堂課會看到:RLHF 的 reward model 有<b>一模一樣</b>的盲點。它也是逐點評分器,永遠不會說「你的回應分布太窄了」。LLM-as-judge 同理。
</div>

<!--
第 1 層是本頁唯一必講的。用一句話問學生:「判別器有沒有辦法告訴生成器『你漏了一群』?」

答案是沒有,因為它一次只看一個點。這個介面限制在 RLHF 裡原封不動地重演。
-->

---
layout: center
---

<div class="px-8 text-center">

<div class="eyebrow mb-4">Demo · GAN 2D · 5 分鐘</div>

<div class="text-lg leading-relaxed">

高斯混合 ×6 訓練到 collapse,<br>
打開「左圖疊上 $D$ 地景」:

</div>

<div v-click class="mt-7 text-xl" style="color: var(--amber)">

畫面上有任何東西告訴你「有兩群被漏掉了」嗎?

</div>

<div v-click class="mt-6 text-base">

**沒有。**

<div class="text-sm opacity-75 mt-3">
被漏掉的那幾群在判別器地景上是<b>亮</b>的,但生成點看不見那份亮度。<br>
亮著,但沒人去。這就是 mode collapse。
</div>

</div>

<div class="mt-6 text-sm">
<a href="/demos/gan-2d-interactive.html" target="_blank">開啟 Demo · GAN 2D 互動版</a>
</div>

</div>

<!--
一定要現場操作,不要只放截圖。

操作順序:先訓練到 collapse → 學生看到只剩兩三群 → 再疊 D 地景 → 讓他們自己發現那些空群是亮的。
-->

---

# GAN 的改進 (1):成因在評估基準,不在架構

<GanFixes />

<div v-click class="mt-2 p-3 border-l-4 border-cyan-400 text-sm">
右邊那些方法沒有一個動了生成器。<b>它們動的全是判別器的地景。</b><br>
<span class="opacity-70">這正是 ② 那張滑桿圖的解法:讓評估基準在分布不重疊時仍然給得出方向。</span>
</div>

<!--
Wasserstein 用「搬土」講:兩座不重疊的土堆,JSD 只會說「完全不同」(log 2,無梯度);
W 距離會說「相距 5 公尺」,距離變近就有獎勵,G 才有路標。

回想 demo:kD=5 時那片又暗又陡的地景,正是這些方法要抹平的對象。
-->

---

# GAN 的改進 (1):四個代表工作

<div class="text-sm">

| 工作 | 年份 | 對付什麼 | 怎麼做 |
|---|---|---|---|
| **WGAN** | 2017 | JSD 飽和、梯度消失 | 換成 Wasserstein 距離;$D$ 變成不設上限的 critic |
| **WGAN-GP** | 2017 | weight clipping 過於粗糙 | 用 gradient penalty $(\|\nabla_{\hat{x}} f\|-1)^2$ 軟性施加 Lipschitz 約束 |
| **Spectral Norm** | 2018 | 同上,更便宜 | 每層權重除以最大奇異值,直接控制 $D$ 的 Lipschitz 常數 |
| **R1 penalty** | 2018 | 收斂性 | 只在真資料上懲罰 $\|\nabla_x D\|^2$;局部收斂有理論保證,StyleGAN 全系列採用 |

</div>

<div v-click class="mt-4 p-3 border-l-4 border-violet-400 text-sm">
四個工作,同一個動作:<b>限制判別器的 Lipschitz 常數</b>。差別只在用什麼手段限制,以及限制得多硬。
</div>

<!--
這頁 90 秒,不要逐列念。

R1 值得多一句:它只在真資料上罰梯度,比 GP 便宜(不用取插值點),
而且 Mescheder et al. (2018) 給了局部收斂的證明。StyleGAN 全系列都用它。
-->

---

# GAN 的改進 (2):架構與工程的十年

<div class="text-sm">

| 工作 | 年份 | 貢獻 |
|---|---|---|
| **cGAN** | 2014 | 條件生成:$G$、$D$ 都吃標籤 $y$,可控生成的起點 |
| **DCGAN** | 2015 | 卷積架構設計準則,GAN 第一次穩定生出像樣的圖 |
| **Progressive GAN** | 2017 | 從 4×4 逐步長到 1024×1024:先學佈局再學細節 |
| **StyleGAN 1/2/3** | 2018–21 | style-based 生成器:$z\to w$ 空間、AdaIN 逐層注入 |
| **BigGAN** | 2018 | 大 batch + 大模型 + truncation trick:類別條件 ImageNet 生成 |

</div>

<div v-click class="mt-4 text-sm">

**GAN 的現代角色**:一步生成、樣本銳利,適合「要快、對感知品質敏感、又不要求全覆蓋」的場合。

<div class="grid grid-cols-2 gap-3 mt-2 text-xs">
<div class="p-2 rounded border border-gray-500">超解析度:SRGAN / <b>ESRGAN</b> 用對抗項補回 MSE 抹掉的高頻紋理</div>
<div class="p-2 rounded border border-gray-500">Image-to-image:pix2pix(成對)、<b>CycleGAN</b>(不成對 + cycle consistency)</div>
<div class="p-2 rounded border border-gray-500">語音 vocoder:MelGAN、<b>HiFi-GAN</b>,即時神經聲碼器的主流</div>
<div class="p-2 rounded border border-gray-500">當 VAE / diffusion 的感知品質補丁(SD 的 VAE 就是對抗式訓練的)</div>
</div>

</div>

<!--
表格 30 秒帶過,重點放在下半:GAN 作為元件而非主角活得非常好。這直接過渡到 ⑥ 的「基準可攜」。

語音組多給 1 分鐘:HiFi-GAN 的任務是 mel-spectrogram → 波形。逐點 loss 對相位幾乎無能為力
(同一個 mel 對應多種合理相位,又是一對多),所以用 multi-period + multi-scale 判別器
加 feature matching loss。一步到位、可即時合成。這正是 GAN「快、銳利、不求密度」的用武之地。
-->

---

# Diffusion / Flow Matching:同一個訓練目標,換一個方向拆

<DecompAxes />

<div v-click class="mt-3 p-3 border-l-4 border-violet-400 text-sm">
還是 forward KL,還是 MLE 路線。差別只在<b>沿什麼軸做鏈鎖分解</b>:AR 沿序列,擴散沿噪聲尺度。<br>
兩者的訓練目標都因此退化成簡單回歸,<b>所以它們一樣穩定。</b>
</div>

<div class="mt-3 text-sm">
<a href="/demos/flow-matching-2d-interactive.html" target="_blank">Demo · Flow Matching 2D</a>
<span class="text-xs opacity-70 ml-2">來源分布不必是高斯;細節留給自修</span>
</div>

<!--
5 分鐘,只建立「同一個訓練目標、不同分解方式」這一個印象。

不要講 SDE、不要講 score matching。想深入的學生給 MIT 6.S184。
-->

---

# 澄清:diffusion 與 flow matching 不是兩個家族

<div class="text-sm">

兩者都在學一個<b>時間相關的向量場</b>,把簡單先驗沿一條機率路徑搬到資料分布。

</div>

<div class="grid grid-cols-3 gap-4 mt-5 text-sm">

<div class="p-3 rounded border border-violet-400">
<b>相同的部分</b><br>
<span class="opacity-75">
機率路徑的框架、回歸型的訓練目標、以及 probability-flow ODE 這個取樣形式。
</span>
</div>

<div class="p-3 rounded border border-amber-400">
<b>真正不同的三件事</b><br>
<span class="opacity-75">
① 選哪一條路徑(VP / VE cosine schedule vs. 線性內插 / OT 耦合)<br>
② 網路預測哪一個量(<katex-elem expr="\varepsilon" /> / <katex-elem expr="x_0" /> / 速度 / score)<br>
③ 用哪一種取樣器與幾步
</span>
</div>

<div class="p-3 rounded border border-cyan-400">
<b>所以</b><br>
<span class="opacity-75">
這三件都是<b>框架內的設計選項</b>,不是家族層級的體質差異。「diffusion 慢、FM 快」是路徑與步數造成的,不是名字造成的。
</span>
</div>

</div>

<div v-click class="mt-5 p-3 border-l-4 border-violet-400 text-sm">
Song et al. (2021) 證明 diffusion 的 SDE 有一條 marginal 相同的 ODE;<br>
Lipman et al. (2023) 證明 diffusion 用的路徑是 conditional flow matching 的<b>特例</b>。
</div>

<!--
這頁是刻意加的更正頁,因為太多教材把兩者並列成兩個家族、還各給一組優缺點。

講法:先問「你覺得它們差在哪?」多數人會說「一個加噪一個學速度場」。
然後指中間那欄:那是同一件事的兩種參數化。

給想深入的:Lipman et al., Flow Matching for Generative Modeling (arXiv:2210.02747) 的 §3 就是
把 diffusion path 寫成 CFM 的特例;Song et al., Score-Based Generative Modeling through SDEs
(arXiv:2011.13456) 的 §4.3 是 probability-flow ODE。

不要在這頁講 SDE 的推導。60 秒,講完就走。
-->

---
layout: section
class: sec-vs
---

# ⑤ VAE vs GAN

## 同一份資料,兩種世界觀

<!--
150–160 分。
-->

---

# 總對照:同一張表,兩次讀法

<div class="text-sm">

| | **VAE** | **GAN** |
|---|---|---|
| 評估基準 | KL(經由 MLE / ELBO),**固定** | 判別器 / JSD,**學出來的** |
| 行為模式 | mode-**covering**:每群都蓋,代價是糊、有橋 | mode-**seeking**:蓋到的銳利,代價是漏群 |
| 訓練 | 單一目標,穩定下降 | 兩人賽局,震盪、可能不收斂 |
| 密度 | 有下界(可比較、可做異常偵測) | 無 |
| 潛在空間 | 有 encoder,天生做表示學習 | 無 encoder,要另外做 inversion |
| loss 可讀性 | 直接讀 ELBO | 不可讀;健康訊號看 $D(\text{real}),D(\text{fake})\to 0.5$ |
| 評估 | test log-likelihood(是下界) | 只能用代理指標:IS、**FID** |

</div>

<div v-click class="mt-3 p-3 border-l-4 border-amber-400 text-sm">
<b>第二次讀法</b>:把左欄遮起來,這整張表就是 ② 那條「forward KL ↔ reverse KL」軸的工程版。<br>
<span class="opacity-70">你們在 B-1 看到的「單峰 q 追雙峰 p」,今天長成了兩個模型家族。</span>
</div>

<!--
「中心偏移高斯混合」是三個 demo 共用的對照組:VAE 搭橋、GAN 漏群、FM 乾淨分群但要多步積分。
同一份資料,三種行為。時間充裕的話現場把兩個 demo 分頁並排。

FID 順帶一提即可(兩組樣本過 Inception 取特徵,擬合高斯後算 Fréchet 距離),
下堂課評估生成模型時會正式用到。
-->

---
layout: section
class: sec-loss
---

# ⑥ 評估基準是可攜的

## 同一個基準,換一個舞台

<!--
160–170 分。

核心觀念:KL 與對抗 loss 都是「分布差異的評估基準」,任何任務只要能表述成
「讓兩個分布接近」或「讓兩個分布不可分」,就能直接搬用。
-->

---

# 兩個基準的遷移地圖

<LossTravel />

<!--
挑兩個細講:

- 知識蒸餾:soft label 攜帶類別之間的相似結構(這張 3 有點像 8),比 one-hot 資訊多;
  溫度 τ 把分布抹軟,放大暗知識。
- RLHF:沒有 KL 項會 reward hacking(語言退化)。那個 β 與 β-VAE 那顆旋鈕是同一個角色。
  學生若走 LLM 方向,這是第一個會親手調的 KL。

DANN 值得一句:min-max 不一定要兩個 optimizer 輪流,梯度反轉層把 max 塞進同一次 backward,
反向時把梯度乘上 −λ。
-->

---

# 為什麼「基準可攜」值得記住

<div class="grid grid-cols-2 gap-5 text-sm mt-4">

<div class="p-4 rounded border border-cyan-400">

### 認出 KL 的句型

「我有一個參考分布,請新分布別離它太遠。」

<div class="mt-3 opacity-80">
一旦你認出這個句型,<b>方向就是可以選的</b>:
寫 <katex-elem expr="\mathrm{KL}(\pi_{\text{ref}}\|\pi_\theta)" /> 得到 covering,
寫 <katex-elem expr="\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})" /> 得到 seeking。<br>
RLHF 用的是後者,這是對齊後多樣性塌陷的直接來源。
</div>

</div>

<div class="p-4 rounded border border-pink-400">

### 認出對抗的句型

「我寫不出『像真的』的公式,那就訓練一個分類器當基準,然後騙過它。」

<div class="mt-3 opacity-80">
一旦你認出這個句型,<b>那個逐點評分的盲點就跟著搬過去了</b>:
判別器、reward model、LLM-as-judge 都不會告訴你「你的輸出分布太窄」。
</div>

</div>

</div>

<div v-click class="mt-4 text-center text-base">

架構每隔幾年換一次,**這兩個基準與它們各自的盲點則會被一併繼承下去**。

</div>

<!--
這頁是 ⑥ 的收束,也是整堂課的實務價值所在。

對做 RLHF 或情感支持對話的組,左欄那句「方向是可以選的」要停 15 秒。
-->

---
layout: section
class: sec-recap
---

# ⑦ 回到失敗現場

## 以及,下堂課要填的空

<!--
170–180 分。
-->

---
layout: center
---

<FailureScenes verdict />

<div v-click class="mt-8 text-center text-lg">

是的,**同一條軸的兩端**。

<div class="text-sm opacity-75 mt-3">
一個被 forward KL 逼著覆蓋全部 support · 一個被 reverse KL 允許只佔住單一模式
</div>

</div>

<!--
回收論題。

再問一次開場那個問題,這次讓他們自己回答。
-->

---

# 今天畫完的:軸的第一列

<div class="mt-6">
  <SpectrumAxis :rows="1" />
</div>

<div class="mt-8 grid grid-cols-3 gap-4 text-sm">

<div class="p-3 border-l-4 border-cyan-400">
<b>今天</b><br>
<span class="opacity-75">訓練時選定一個散度,它決定了模型分布的形狀</span>
</div>

<div class="p-3 border-l-4 border-amber-400">
<b>下堂課 · 第 2 列</b><br>
<span class="opacity-75">訓練結束了,但分布不是你要的 → 推論時滑動</span>
</div>

<div class="p-3 border-l-4 border-pink-400">
<b>下堂課 · 第 3 列</b><br>
<span class="opacity-75">把同樣的滑動烘進權重裡</span>
</div>

</div>

<div v-click class="mt-6 text-center text-base">
下面兩列,是上面這一列的<b>推論時版本</b>與<b>權重版本</b>。
</div>

<!--
明確告訴學生:下面兩列今天留白是刻意的,下堂課才填。

先預告一句:下堂課你會發現 temperature、CFG、DPO 全部可以寫成同一個式子。
-->

---
layout: section
class: sec-recap
---

# 作業

## 兩天後上課,開場第一件事就是收

<!--
作業的兩部分是刻意分開的:A 是把自己的題目形式化,B 是把今天的詞彙用在別人的資料上。
兩件都做完,才算真的聽懂。

強調:A 寫不出來也要交,寫不出來本身就是結論。
-->

---

# 作業 A · 把你自己的題目寫成機率式子

<div class="text-sm">

挑**一個你現在正在做的題目**(不是假想的),完成下面四件事:

</div>

<div class="mt-3 text-sm">

| | 要寫的東西 | 具體要求 |
|---|---|---|
| **A1** | 指認變數 | 哪一個是 $x$?哪一個是 $y$?哪一個是條件 $c$?各用一句話說明它在你的資料裡實際是什麼(不是抽象符號) |
| **A2** | 寫出目標函數 | 你**實際在最小化**的那個式子。如果是現成模型微調,就寫出那個 loss 的數學形式 |
| **A3** | 定位 | 它落在那條軸的**哪一側**?給出理由,理由必須引用 A2 的式子,不能只說「感覺上」 |
| **A4** | 預測失效模式 | 根據 A3 的位置,預測你的系統**應該**出現哪一種失效:過度平滑 / 覆蓋過廣,還是模式塌縮 / 多樣性喪失 |

</div>

<div v-click class="mt-4 p-3 border-l-4 border-amber-400 text-sm">
<b>A4 是這份作業真正的重點。</b>下堂課我們會把你的預測跟你系統的實際行為對照。
預測錯了比預測對了更有價值,那代表軸上還有你沒看到的東西。
</div>

<!--
A2 是最多人會卡的地方。提示他們:如果你在用 HuggingFace 的 Trainer,去看 compute_loss;
如果你在做 prompt engineering 而沒有訓練,那你的 A2 是「我沒有目標函數,我在動 c」,
這本身就是 A3 的答案(在軸上不動,只換條件)。
-->

---

# 作業 B · 三個 demo 的對照觀察

<div class="text-sm">

把三個互動 demo 都切到**「中心偏移混合」**這組資料,各截一張收斂後的圖:

</div>

<div class="grid grid-cols-3 gap-3 mt-3 text-sm">

<div class="p-3 rounded border border-cyan-400">
<b>VAE 2D</b><br>
<span class="text-xs opacity-75"><katex-elem expr="\beta=1" />。注意群與群之間有沒有東西</span>
</div>

<div class="p-3 rounded border border-pink-400">
<b>GAN 2D</b><br>
<span class="text-xs opacity-75">訓到穩定為止。數一下蓋到幾群</span>
</div>

<div class="p-3 rounded border border-violet-400">
<b>Flow Matching 2D</b><br>
<span class="text-xs opacity-75">記下你用了幾步積分</span>
</div>

</div>

<div class="mt-4 text-sm">

然後寫 **300 字以內**,回答:三張圖的差異,分別對應到今天講的哪一個機制?

</div>

<div v-click class="mt-3 p-3 border-l-4 border-cyan-400 text-sm">
必須用到這三個詞:<b>mode-covering</b>、<b>mode-seeking</b>、<b>逐點評分的盲點</b>。<br>
<span class="opacity-70">用不上表示你還沒把 demo 的現象跟機制接起來,回去重看 ② 跟 ④。</span>
</div>

<!--
B 的設計意圖:強迫學生用同一組詞彙描述同一份資料上的三種行為。這是本堂的驗收點。

如果有人問 Flow Matching 為什麼也要看:因為它是下堂課的預告,而且它在三難裡的位置
跟 VAE、GAN 都不同,對照才完整。

注意用詞:demo 叫 Flow Matching,但學生要帶走的是「Diffusion / FM 這一類」的行為,
不是「FM 這個名稱」專屬的行為。
-->

---

# 作業:格式、繳交、評分

<div class="grid grid-cols-2 gap-5 text-sm mt-3">

<div>

### 格式

- **A + B 合併成一份 PDF,兩頁 A4 以內**
- 手寫拍照可以,式子看得清楚就好
- B 的三張截圖算在兩頁內
- 檔名:`姓名_lecture01.pdf`

### 繳交

- **下堂課開始前**(兩天後)上傳到課程資料夾
- 遲交仍要交;開場的對照環節會直接用到你的 A4

</div>

<div>

### 評分重點(不評分數,只回饋)

<div class="p-3 rounded border border-violet-400 mt-2">
① A2 的式子與 A3 的理由<b>是否對得上</b><br>
<span class="text-xs opacity-70">最常見的失分:式子寫 CE,理由卻說 mode-seeking</span>
</div>

<div class="p-3 rounded border border-amber-400 mt-2">
② A4 的預測<b>是否可以被推翻</b><br>
<span class="text-xs opacity-70">「可能會有點問題」不算預測</span>
</div>

<div class="p-3 rounded border border-cyan-400 mt-2">
③ B 有沒有把現象接回機制,而不是只描述畫面
</div>

</div>

</div>

<div v-click class="mt-4 p-3 border-l-4 border-pink-400 text-sm">
A 寫不出來的,通常不是數學不夠,是題目本身還沒定義清楚。<br>
<span class="opacity-70">那也是一個有用的結論。請照實寫下「卡在哪一步、為什麼卡住」,那一段我會單獨回覆。</span>
</div>

<!--
最後這一段要唸出來。每一屆都有人因為寫不出 A2 而乾脆不交,那是最可惜的情況。
寫不出目標函數,通常代表題目的輸入輸出還沒定義清楚,而那正是我最該介入的時候。

下堂課第一件事就是收這個,然後直接把六個題目標到軸上。
-->

---

# 課後資源

<div class="grid grid-cols-2 gap-5 text-sm">

<div>

### 骨架(建議先讀)

- **Stanford CS236** Lecture 1–2 — <span class="text-xs opacity-70">deepgenerativemodels.github.io/notes</span>
- **Bishop & Bishop**《Deep Learning: Foundations and Concepts》機率章
- **Prince**《Understanding Deep Learning》Ch.15–18 — <span class="text-xs opacity-70">GAN / VAE / Flow / Diffusion</span>

### 對應本堂特定段落

- Goodfellow, *NIPS 2016 Tutorial* — ③ 分類樹(arXiv:1701.00160)
- Xiao et al. (ICLR 2022) — ③ 生成三難(arXiv:2112.07804)
- Kingma & Welling (2013) — ④ VAE(arXiv:1312.6114)
- Arjovsky & Bottou (2017) — ④ GAN 那個散度式子的來源
- Kalai et al. (2025) *Why LMs Hallucinate* — 虛假前提段

</div>

<div>

### 三個互動 demo(作業 B 用)

- <a href="/demos/divergence-2d-interactive.html" target="_blank">散度的選擇</a> — 三個散度同時量測
- <a href="/demos/gan-2d-interactive.html" target="_blank">GAN 2D</a> — 「亮著但沒人去」
- <a href="/demos/vae-2d-interactive.html" target="_blank">VAE 2D</a> / <a href="/demos/flow-matching-2d-interactive.html" target="_blank">Flow Matching 2D</a>

### 想深入的

- van den Oord et al., *VQ-VAE* — arXiv:1711.00937
- Vahdat & Kautz, *NVAE* — arXiv:2007.03898
- Karras et al., *StyleGAN2* — arXiv:1912.04958
- Kong et al., *HiFi-GAN* — arXiv:2010.05646
- **MIT 6.S184** — 附三個 Colab lab
- 李宏毅《生成式 AI》系列(中文暖身)

</div>

</div>

<!--
不要念這頁。指一下 demo 連結會放課程頁面,然後結束。

提醒:年份以 arXiv 首發為準,正式發表場次略有出入,引用時依情境標註。
-->

---
layout: center
class: text-center
---

<div class="eyebrow">下堂課 · 兩天後</div>

# 在軸上滑動

<div class="text-lg opacity-80 mt-3">引導生成:改條件、改取樣、改 logits、改聚合、改權重、改表徵</div>

<div class="mt-10 max-w-3xl mx-auto">
  <SpectrumAxis :rows="1" />
</div>

<div class="mt-8 text-sm opacity-70">
帶著你的作業來,第一件事就是把六個題目標到這條軸上。
</div>

<!--
最後一句:今天你們學到的不是「有哪些生成模型」,是「為什麼只能有這些生成模型」。

兩天後見。
-->
