---
theme: seriph
title: 生成模型入門 · 第一堂:選一把尺
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
duration: 120min
mdc: true
---

<div class="eyebrow">生成模型入門 · Lecture 01 / 02</div>

# 選一把尺

## 生成 = 分布逼近,而「逼近」不是一個唯一定義的動作

<div class="pt-6 text-sm opacity-70">
實驗室新生課程 · 120 分鐘<br>
先修:機率、KL 散度、Jensen 不等式、反向傳播
</div>

<div class="abs-br m-6 text-xs opacity-50">
第二堂:在軸上滑動 —— 引導生成
</div>

<!--
這門課不是生成模型通識,是實驗室現有研究的底層說明書。

骨幹放散度,不放模型分類法。分類法只給一張清單,散度給的是可推導的因果。

開場先不要講大綱,直接做 ① 的兩個現場示範。
-->

---
layout: section
class: sec-intro
---

# ① 兩個失敗現場

## 現場示範,不看投影片

<!--
0–10 分。

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

不要立刻否定。說:接下來 35 分鐘我要證明它們是同一件事。
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
  <div class="text-xs opacity-60 mt-2">這條軸今天畫最上面一列,下週補完下面兩列。</div>
</div>

<!--
第 10 分鐘提出論題,第二堂最後一頁回收。

現在這條軸還是空的。今天結束時,最上面一列會被填滿;下週你會看到下面兩列其實是上面那一列的「推論時版本」與「權重版本」。
-->

---

# 你們手上的題目,其實都是機率問題

<div class="text-sm">

| 實驗室題目 | 它其實是什麼機率問題 | 在軸上的位置 |
|---|---|---|
| prompt engineering | 選擇條件變數 $c$,操控 $p(y \mid c)$ | 只換 base,不動係數 |
| memory agent | 條件集合的建構;長 context 下 $p(\text{task} \mid \text{prompt})$ 被稀釋 | 同上 |
| 情感支持對話 | 通用安慰語 vs. 對齊後的多樣性塌陷 | 偏右,且被 $\beta$ 綁住 |
| 虛假前提檢測 | $p(y\mid x)$ 永遠 well-defined,即使 $p(x)\approx 0$ | 左端的結構性後果 |
| confidence vs. accuracy | 校準;predictive entropy vs. semantic entropy | 量測工具本身 |
| LLM-ASR | $p(\text{text}\mid\text{audio})\propto p(\text{audio}\mid\text{text})\,p(\text{text})$ | 對數空間的線性組合 |

</div>

<div v-click class="mt-4 p-3 border-l-4 border-amber-400 text-sm">
右邊那一欄現在看不懂沒關係。這是本課的<b>驗收清單</b>,第二堂最後一頁逐格回填。
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
10–45 分,本堂最重的一段。

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

問題來了:**「逼近」是什麼意思?**

兩個分布之間的差異,得先定義一個評估基準才能量化。

</div>

<div class="mt-4 text-sm opacity-80">
而這樣的基準不只一種,而且<b>不對稱</b>。
</div>

</div>

</div>

<!--
這頁 2 分鐘。重點只有一句:因為建表不可能,所以生成問題被迫變成「分布逼近」問題;而分布逼近問題被迫要選一個散度。

散度的選擇是這門課唯一的主角。
-->

---

# KL 散度:積分的權重決定了它懲罰什麼

$$\mathrm{KL}(p\|q)=\int p(x)\,\log\frac{p(x)}{q(x)}\,dx$$

<div v-click class="mt-4 text-center text-lg">

權重是 <span style="color: var(--violet)">$p$</span>,所以它**只在 $p$ 有質量的地方施加懲罰**。

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

<div class="text-xs opacity-60 mb-1">下面三條曲線是數值最小化各自的散度解出來的,不是示意圖。</div>

<DivergenceFit :w="0.5" :curves="['forward']" annotate="forward" />

<div v-click class="mt-3 p-3 border-l-4 border-cyan-400 text-sm">
forward KL 寧可把大量機率放在<b>資料根本不存在的谷底</b>,也不敢在任何一個峰上放 0。<br>
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
reverse KL 直接放棄左邊那個峰,把自己縮進右峰裡:<b>銳利,但漏掉一半的 support。</b><br>
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
support 不重疊 → $\infty$,而且是沒得救的 $\infty$。

</div>

<div class="p-4 rounded border border-amber-400">

### JSD(換掉分母)

$$\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m)$$
$$m=\tfrac{p+q}{2}$$

分母是混合分布,$m\ge\tfrac12 p$ 且 $m\ge\tfrac12 q$

</div>

</div>

<div v-click class="mt-5 text-center">

於是 **永遠有界**:$\;0\le\mathrm{JSD}(p\|q)\le\log 2$,而且 $\sqrt{\mathrm{JSD}}$ 是真正的距離度量。

</div>

<!--
關鍵在「分母換成 m」這個動作:log(p/m) 最大只能是 log 2,因為 m 至少是 p 的一半。

有界聽起來是優點。下一頁說明它為什麼同時是災難。
-->

---

# JSD:澄清二 · 有界 = 會飽和 = 梯度消失

<JsdSaturate />

<div v-click class="mt-2 p-3 border-l-4 border-amber-400 text-sm">
兩個分布一分開,JSD 立刻貼到 $\log 2$ 就<b>不動了</b>:曲線變平,梯度變 0。<br>
<span class="opacity-70">「你離目標很遠」與「你離目標非常遠」給出同一個 loss,模型就不知道該往哪走。</span>
</div>

<div v-click class="mt-2 text-xs opacity-70">
這正是原始 GAN 的病理,也是 non-saturating loss 與 WGAN 的動機。後面 ④ 會回來算帳。
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

所以 JSD 的意思是:**「一個最好的分類器,能多有把握地分辨這個樣本來自誰。」**

形式上它等於 $I(X;Z)$,其中 $Z\sim\text{Bernoulli}(1/2)$ 是「來自 $p$ 還是 $q$」的標籤。

</div>

<div v-click class="mt-3 text-center text-sm opacity-80">
分不出來 → 互資訊 0 → JSD 0。<b>「像不像」被翻譯成了「猜不猜得到」。</b>
</div>

<!--
這一頁是 ③ 的伏筆:判別器就是 JSD 的操作型定義,不是 GAN 的裝飾品。

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
這頁是本次備課實際算出來的,不是教科書上的圖。w=0.5 時 JSD 解 μ=0 σ=1.66(covering);w=0.3 時直接跳到 μ=1.60 σ=0.58,與 reverse KL 完全相同。

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
用單一高斯擬合 1D 雙峰混合,分別最小化三個散度,數值求解 $\mu,\sigma$。<br>
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
45–70 分。

這段講完,學生對 AR / VAE / GAN / DPM 之間的關係應該就不會再亂了。
-->

---

# 一個散度要能算,你得先拿得到兩樣東西

<div class="grid grid-cols-2 gap-6 mt-6 text-sm">

<div class="p-4 rounded border border-cyan-400">

### 取樣 (sampling)

從某個分布抽出樣本 $x\sim p$

<div class="mt-3 text-xs opacity-75">
$p_{\text{data}}$:有資料集,✓<br>
$p_\theta$:模型自己跑一次,✓
</div>

</div>

<div class="p-4 rounded border border-pink-400">

### 密度 (density)

給定 $x$,算出 $p(x)$ 這個**數值**

<div class="mt-3 text-xs opacity-75">
$p_\theta$:likelihood-based 模型算得出,✓<br>
$p_{\text{data}}$:<b style="color: var(--pink)">永遠拿不到 ✗</b>
</div>

</div>

</div>

<div v-click class="mt-6 text-center text-base">

$\log p_{\text{data}}(x)$ 拿不到。**整個生成模型的分類法,就是從這一個缺口長出來的。**

</div>

<!--
先讓學生確認這四格。特別是右下角:我們有一大堆從 p_data 抽出來的樣本,但我們永遠不知道任何一個樣本「本來應該有多大機率」。

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

講法:先蓋住「代打」與「於是變成」兩欄,只念前兩欄,讓學生自己推。多數人推得出 GAN 那一列。

推不出來也沒關係,重點是他們親身體驗到「這是被逼出來的,不是被發明出來的」。
-->

---
layout: center
---

<div class="px-8">

<div class="eyebrow mb-4">替第二堂鋪路</div>

<div class="text-lg leading-relaxed">

reverse KL 那一列的「**reward 代打**」,<br>
與 GAN 那一列的「**判別器代打**」,<br>
其實是同一件事的兩種說法:

</div>

<div v-click class="mt-6 p-4 border-l-4 border-violet-400 text-lg">
兩者都在補同一個拿不到的東西:$\log p_{\text{data}}$。
</div>

<div v-click class="mt-5 text-sm opacity-70">
第二堂的 capstone(DDO)就站在這句話上:如果 likelihood-based 模型本來就算得出 $\log p_\theta$,那判別器其實可以不用另外蓋一個網路。
</div>

</div>

<!--
30 秒,講完就走,不展開。

這是刻意留的懸念。下週回收。
-->

---
layout: section
class: sec-family
---

# ④ 家族巡禮

## 每個家族的典型失效,都來自它選定的訓練目標

<!--
70–110 分。AR 20 / VAE 7 / GAN 8 / DPM 5。

紀律:每個模型只講「它選了哪個散度」與「那個選擇帶來什麼失效模式」。架構細節一律不講。
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
<b>分類任務</b>:目標是 one-hot,$H(p)=0$,所以 cross-entropy <b>就是</b> forward KL,一絲不差。<br>
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
<div class="p-3 rounded border border-gray-500">② 相對於<b>參考模型</b>正規化 → $\log\frac{p_\theta}{p_{\text{ref}}}$</div>
<div class="p-3 rounded border border-gray-500">③ 用已知 $H(p)$ 的<b>合成資料</b></div>
<div class="p-3 rounded border border-gray-500">④ 繞開 likelihood:MAUVE / MMD / 下游指標</div>
</div>

</div>

<div v-click class="mt-3 text-xs opacity-70">
② 這條路第二堂會變成主角:那個量叫 likelihood ratio,它天生扣掉了「這筆資料本身有多難」。
</div>

<!--
confidence vs. accuracy 那組請特別記 ②:你拿 sequence log-prob 當信心,其實混進了「這句話本身有多罕見」。用 reference model 相除就能扣掉。

這是第二堂最後那個現成 research idea 的起點。
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
第二堂 ⑤ 回收:跨模型比較 perplexity 幾乎都是錯的,除非分母換成 bytes。
</div>

<!--
LLM-ASR 組會直接用到:換 LM 之後,LM weight 這個超參數要能移植,尺度就得統一。

不要在這頁停太久,3 分鐘。
-->

---

# 鏈鎖分解:請盯著期望值的下標

$$\mathrm{KL}(p\|q)=\sum_t \mathbb{E}_{x_{\lt t}\sim \textcolor{#ff6b9d}{p}}\Big[\mathrm{KL}\big(p(\cdot\mid x_{\lt t})\,\big\|\,q(\cdot\mid x_{\lt t})\big)\Big]$$

<div v-click class="mt-6 text-center text-lg">

前綴 $x_{\lt t}$ 取自 <b style="color: var(--pink)">$p$</b>,**不是** $q$。

</div>

<div v-click class="mt-6 text-sm opacity-80 text-center">
這一個下標,決定了 AR 模型的訓練方式與它的招牌 bug。整個第二堂也是從這裡開始。
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
這張圖第二堂 DDO 段會原樣回放。請他們拍照。

問學生:如果你想讓下面那條軌道也被測量,你會怎麼做?先不要回答,下週講。
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

推論時前綴來自 $q$,但訓練目標**從未測量過任何模型自己生成的前綴**。<br>
→ 訓練與推論優化的不是同一個泛函 → **memory agent 的長對話漂移**

</div>

<div class="p-4 rounded border border-violet-400 text-sm">

### 3 · 把下標從 $p$ 換成 $q$,就得到 reverse KL 的分解

scheduled sampling / RL / DPO / DDO 的共同點**不是「用了 RL」**,<br>
而是 **把 loss 搬到了下方那條軌道上。**

</div>

</v-clicks>

<!--
第 2 點對 memory agent 組是直球:長對話漂移不是 context window 的問題,是訓練目標從來沒看過自己生成的長前綴。

第 3 點是整個第二堂的入口,講完停 15 秒。
-->

---

# forward KL 在 token 層級的實作機制

<TokenBars />

<div v-click class="mt-1 text-sm">

語料裡每個前綴通常**只有一個續寫**,所以每個位置的目標是 one-hot,
一個高變異但**無偏**的估計。模型仍能收斂到條件分布,因為 CE 是 **proper scoring rule**。

</div>

<!--
「只有一個續寫卻能學到分布」這件事學生常卡住:因為同一個前綴在整份語料裡會出現很多次,不同次的續寫不同,平均起來就是條件分布。

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
「玉山的第三個火山口叫什麼?」$p(x)\approx 0$,但 $p(y\mid x)$ 照樣算得出來,而且模型從沒被教過在這裡要停。
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
**confidence 估計最常踩的坑。**<br>
第二堂的 semantic entropy 就是為了繞開它。
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
兩個坑的共同點:<b>你以為你在量「模型多有把握」,其實你量到的是「這串 token 有多長、有多罕見」。</b>
</div>

<!--
confidence vs. accuracy 那組現在應該已經知道自己第一步要做什麼了:先把量測工具修好,再談相關性。
-->

---

# VAE:同一個訓練目標,但密度算不動,只好取下界

<div class="grid grid-cols-2 gap-6 mt-4 text-sm">

<div>

$$\log p_\theta(x)\ \ge\ \underbrace{\mathbb{E}_{q}[\log p_\theta(x\mid z)]}_{\text{重建}}-\underbrace{\mathrm{KL}(q\|p(z))}_{\text{正則}}$$

<div class="mt-4">

- 仍然是 **forward KL / MLE 路線**,只是 $p_\theta(x)=\int p(x\mid z)p(z)dz$ 積不出來
- 所以退而求其次,最大化 **ELBO**

</div>

</div>

<div class="p-4 rounded border border-cyan-400">

### 為什麼 VAE 生出來的是糊的?

**兩層原因,都不是「網路不夠大」:**

1. 高斯 likelihood = MSE → 一個輸入對應多個合理輸出時,**最優解是平均**
2. forward KL 路線本身就 mode-covering

</div>

</div>

<div v-click class="mt-4 text-xs opacity-70">
潛在變數 $z$ 與邊際化這組符號請留著。第二堂講 ICL 的隱式貝氏推論與 semantic entropy 會直接再用一次。
</div>

<div class="mt-2 text-sm">
<a href="/demos/vae-2d-interactive.html" target="_blank">Demo · β-VAE 2D</a>:圓環拓撲蓋不乾淨、β 兩端崩壞
</div>

<!--
7 分鐘。VAE 不是今天的重點,但「邊際化 → 積不出來 → 取下界」這個動作要留印象,因為第二堂 ICL 那條式子長得一模一樣。
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
你為了修梯度消失而換掉的那個 loss,<b>順手把 mode-covering 也一起換掉了。</b><br>
<span class="opacity-70">GAN 的兩種 loss,分別坐在兩種失敗模式上。</span>
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
3. min-max 沒有位能函數 → 繞圈 → mode hopping;$D$ 會遺忘,塌縮後沒有回頭的力
4. 一步生成的幾何代價:只蓋單一模式的映射既平滑又便宜,塌縮是低成本解

</div>

</v-clicks>

<div v-click class="mt-5 p-3 border-l-4 border-amber-400 text-sm">
第二堂會看到:RLHF 的 reward model 有<b>一模一樣</b>的盲點。它也是逐點評分器,永遠不會說「你的回應分布太窄了」。LLM-as-judge 同理。
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

# DPM / Flow:同一個訓練目標,換一個方向拆

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
layout: section
class: sec-recap
---

# ⑤ 回到失敗現場

## 以及,下週要填的空

<!--
110–120 分。
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
<b>下週 · 第 2 列</b><br>
<span class="opacity-75">訓練結束了,但分布不是你要的 → 推論時滑動</span>
</div>

<div class="p-3 border-l-4 border-pink-400">
<b>下週 · 第 3 列</b><br>
<span class="opacity-75">把同樣的滑動烘進權重裡</span>
</div>

</div>

<div v-click class="mt-6 text-center text-base">
下面兩列,是上面這一列的<b>推論時版本</b>與<b>權重版本</b>。
</div>

<!--
明確告訴學生:下面兩列今天留白是刻意的,下週才填。

先劇透一句:下週你會發現 temperature、CFG、DPO 全部可以寫成同一個式子。
-->

---
layout: center
---

<div class="px-10">

<div class="eyebrow mb-4">作業 · 下週上課前</div>

<div class="text-lg leading-relaxed">

挑一個**你自己的題目**,寫出它的機率式子:

</div>

<div class="mt-5 p-4 rounded border border-violet-400 text-base">

哪一個是 $x$?哪一個是 $y$?哪一個是條件 $c$?

</div>

<div class="mt-5 text-lg">

然後指出:它落在那條軸的**哪一側**,為什麼?

</div>

<div v-click class="mt-7 p-3 border-l-4 border-amber-400 text-sm">
寫不出來的,通常不是數學不夠,是題目本身還沒定義清楚。<br>
<span class="opacity-70">那也是一個有用的結論,請照實寫。</span>
</div>

</div>

<!--
一頁 A4 以內,不要寫成報告。

下週第一件事就是收這個,然後直接把六個題目標到軸上(第二堂 ⑥)。
-->

---

# 課後資源

<div class="grid grid-cols-2 gap-5 text-sm">

<div>

### 骨架(建議先讀)

- **Stanford CS236** Lecture 1–2 — <span class="text-xs opacity-70">deepgenerativemodels.github.io/notes</span>
- **Bishop & Bishop**《Deep Learning: Foundations and Concepts》機率章 — <span class="text-xs opacity-70">免費線上版</span>
- **Jurafsky & Martin**《SLP》第 3 版 — <span class="text-xs opacity-70">鏈鎖法則、perplexity、噪聲通道</span>

### 對應本堂特定段落

- Arjovsky & Bottou (2017) — ④ GAN 那個散度式子的來源
- Kalai et al. (2025) *Why LMs Hallucinate* — 虛假前提段
- Kim et al. *(QA)²* — 建議實際跑一次

</div>

<div>

### 三個互動 demo

- <a href="/demos/divergence-2d-interactive.html" target="_blank">散度的選擇</a> — 三個散度同時量測
- <a href="/demos/gan-2d-interactive.html" target="_blank">GAN 2D</a> — 「亮著但沒人去」
- <a href="/demos/vae-2d-interactive.html" target="_blank">VAE 2D</a> / <a href="/demos/flow-matching-2d-interactive.html" target="_blank">Flow Matching 2D</a>

### 想深入的

- Tomczak《Deep Generative Modeling》2nd ed.
- **MIT 6.S184** — 附三個 Colab lab
- 李宏毅《生成式 AI》系列(中文暖身)

</div>

</div>

<!--
不要念這頁。指一下 demo 連結會放課程頁面,然後結束。
-->

---
layout: center
class: text-center
---

<div class="eyebrow">下週</div>

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

下週見。
-->
