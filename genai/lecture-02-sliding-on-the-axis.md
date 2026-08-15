---
theme: seriph
title: 生成模型入門 · 第二堂:引導生成
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

<div class="eyebrow">生成模型入門 · Lecture 02 / 02</div>

# 引導生成

## 訓練決定了 $p_\theta$,而部署時需要的是另一個分布

<div class="pt-6 text-sm opacity-70">
實驗室新生課程 · 180 分鐘<br>
先修:第一堂(選一把尺)· forward / reverse KL 的不對稱
</div>

<div class="abs-br m-6 text-xs opacity-50">
解碼 · 對比解碼 · RLHF / DPO · DDO · 不確定性量測
</div>

<!--
時間切法:① 0–12 / ② 12–42 / ③ 42–84 / 休息 84–94 / ④ 94–142 / ⑤ 142–168 / ⑥ 168–180。

開場先收上週的作業,五分鐘內把六個題目標到 mode-covering–mode-seeking 的位置上,再翻頁。

第一堂只推導了 forward KL 這一端。RLHF 為什麼導致多樣性下降,上週講了三次但沒寫過目標函數,
今天 ④ 補上。
-->

---
layout: section
class: sec-intro
---

# ① 從訓練分布到部署分布

## 引導生成的問題設定

<!--
0–12 分。
-->

---

# 第一堂建立的結果

<div class="mt-4">
  <SpectrumAxis :rows="1" />
</div>

<div class="mt-6 grid grid-cols-2 gap-5 text-sm">

<div class="p-3 border-l-4 border-cyan-400">
<b>forward KL</b><br>
<span class="opacity-75">積分權重為 p_data,零機率處的懲罰發散,因此 q 必須覆蓋整個 support。代價是質量被配置到資料不存在的區域。</span>
</div>

<div class="p-3 border-l-4 border-pink-400">
<b>reverse KL</b><br>
<span class="opacity-75">積分權重換成 q,未被覆蓋的模式不受懲罰。代價是 support 覆蓋不足與多樣性下降。</span>
</div>

</div>

<div v-click class="mt-5 text-center text-base">

散度的選擇決定了模型的失效模式。訓練結束後,這個選擇已經固化在 $\theta$ 裡。

</div>

<!--
複習 90 秒,不要重講第一堂。

要喚起的只有一件事:兩種失效模式來自同一個選擇的兩個方向。
第一堂的圖(單一高斯擬合雙峰)可以口頭提一次。

上週懸而未決的是右半邊:RLHF 的目標函數長什麼樣、KL 寫在哪一邊。④ 會處理。
-->

---

# 部署時需要的分布,通常不是 $p_\theta$

<div class="grid grid-cols-2 gap-6 mt-5">

<div>

四種常見需求:

<div class="mt-3 text-sm space-y-2">
<div class="p-2 rounded border border-cyan-400">提高樣本品質,降低離題與錯誤陳述</div>
<div class="p-2 rounded border border-amber-400">符合某個條件:這一題、這個使用者、這份文件</div>
<div class="p-2 rounded border border-pink-400">拒絕特定輸出</div>
<div class="p-2 rounded border border-violet-400">提高多樣性:重複取樣得到不同回應</div>
</div>

</div>

<div v-click>

四者都可以表述成同一個問題:

<div class="my-4 p-4 rounded border border-violet-400 text-sm">
給定固定的 <katex-elem expr="p_\theta" />,構造一個變換,得到目標分布 <katex-elem expr="p_{\text{guided}}" />
</div>

差別只在變換施加的位置:條件、logits、取樣、樣本集合、或權重本身。

</div>

</div>

<div v-click class="mt-5 text-center text-base">

第四項與前三項的方向相反,而它們共用同一個控制參數。

</div>

<!--
第四項刻意放在最後。前三項每天在做,第四項多數人沒想過那也是一個可以主動要求的目標。

最後那句是 ④ 的預告:β 同時控制安全性與多樣性。情感支持組會在這裡有反應。
-->

---

# 三種介入時機

<div class="mt-4">
  <SpectrumAxis :rows="3" />
</div>

<div v-click class="mt-6 text-sm">

三列對應同一組性質,差別在施加的時間點:

<div class="mt-3 grid grid-cols-3 gap-4">
<div class="p-3 rounded border border-cyan-400">訓練時選定散度,決定 <katex-elem expr="p_\theta" /> 的形狀</div>
<div class="p-3 rounded border border-amber-400">推論時調整解碼,不動任何參數</div>
<div class="p-3 rounded border border-pink-400">訓練後微調權重,把調整固化進 <katex-elem expr="\theta" /></div>
</div>

</div>

<!--
上週這張圖只有第一列。第 2、3 列今天填上。

指著第 2 列:③ 講這一列。第 3 列:④ 講這一列,並證明它與第 1 列是同一組性質。
-->

---

# 六個介入點

<GuidanceLadder />

<div v-click class="mt-3 p-3 border-l-4 border-amber-400 text-sm">
編號依<b>介入成本</b>排序。第 1–4 層只改變推論流程,隨時可撤回;第 5–6 層改動模型本身,撤回需要重新訓練或保留舊 checkpoint。
</div>

<!--
逐層點名對應段落:1–4 在 ③(42–84 分),5–6 在 ④(94–142 分)。

有人一定會問編號與執行順序的關係:取樣(2)在計算 logits(3)之後才發生。先講掉。
-->

---
layout: section
class: sec-ruler
---

# ② 引導的對數線性形式

## 解碼方法與對齊目標的共同表述

<!--
12–42 分。這一段最該慢。

紀律:表格的每一行都要在講完後 30 秒內指出它對應到誰的日常工作。
做不到的那一行就該刪掉。
-->

---

# 一般形式

$$\log p_{\text{guided}}(x) \;=\; \underbrace{\log p_{\text{base}}(x)}_{\text{起點}} \;+\; w\,\underbrace{\big(\log p_A(x) - \log p_B(x)\big)}_{\text{方向}} \;-\; \log Z$$

<div class="mt-4 text-center text-sm opacity-80">
最後一項是重新歸一化的常數
</div>

<v-clicks>

<div class="grid grid-cols-3 gap-4 mt-6 text-sm">

<div class="p-3 rounded border border-cyan-400">
<b>起點</b><br>
<span class="opacity-75">變換施加於哪個分布。條件變數的改動只作用在這一項。</span>
</div>

<div class="p-3 rounded border border-violet-400">
<b>方向</b><br>
<span class="opacity-75">兩個分布的對數之差。決定質量往哪個方向移動。</span>
</div>

<div class="p-3 rounded border border-pink-400">
<b>係數</b><br>
<span class="opacity-75">移動的幅度。一個純量,控制銳化的程度。</span>
</div>

</div>

</v-clicks>

<!--
三個位置先講清楚,後面每出現一個新方法,學生要能自動問「它動的是哪一格」。

對數空間的線性組合等價於機率空間的乘冪與相乘。這句要說出來,
④ 最後那個 p_ref^(1−1/β)·p_data^(1/β) 才會看起來理所當然。
-->

---

# 六種方法的對應項

<div class="text-sm">

| 方法 | base | 方向 $\log p_A - \log p_B$ | 係數 |
|---|---|---|---|
| temperature | $\log p$ | — | 整體乘上 $1/T$ |
| CFG(影像擴散) | $\log p(x\mid c)$ | $\log p(x\mid c)-\log p(x)$ | $w$ |
| contrastive decoding / DoLa | $\log p_{\text{strong}}$ | $\log p_{\text{strong}}-\log p_{\text{weak}}$ | $\lambda$ |
| Autoguidance | $\log p_\theta$ | $\log p_\theta-\log p_\phi$ | $w$ |
| RLHF 的最優解 | $\log \pi_{\text{ref}}$ | $r(y)$ | $1/\beta$ |
| DDO 的最優解 | $\log p_{\text{ref}}$ | $\log p_{\text{data}}-\log p_{\text{ref}}$ | $1/\beta$ |

</div>

<div v-click class="mt-4 p-3 border-l-4 border-violet-400 text-sm">
前四列是推論時的方法,後兩列是訓練目標的閉式最優解,推導見 §④。
</div>

<!--
逐行落地,每行 30 秒:

- temperature:每個人都調過。
- CFG:做影像的同學每天在用的那個 guidance scale。
- contrastive decoding:第 3 層,③ 會講,不需要訓練。
- Autoguidance:同一個模型的早期 checkpoint 當作 weak,這篇把統一形式講得最乾淨。

第 3 到第 6 列的方向項形式相同,只有 A 與 B 換了。讓學生自己看出來。
-->

---

# 係數對分布的作用

<GuidanceShift />

<div v-click class="mt-3 p-3 border-l-4 border-pink-400 text-sm">
三條曲線出自同一組 <katex-elem expr="p(x\mid c)" /> 與 <katex-elem expr="p(x)" />,只有 <katex-elem expr="w" /> 不同。<br>
<span class="opacity-70">w 由 0 增至 4,次要模式的質量被移走。第一堂 reverse KL 的解具有相同的行為,而這裡沒有動過任何一個參數。</span>
</div>

<!--
曲線是數值算出來的:p_guided ∝ p(x|c)·(p(x|c)/p(x))^w,離散化到 481 個格點再歸一化。

同樣的品質與覆蓋取捨,訓練時要付重新訓練的代價,推論時只要改一個純量。

問學生:w 取負值會怎樣?(答:分布被抹平,與 T > 1 的效果同向。)
-->

---

# 截斷式取樣

<TempTopP />

<div v-click class="mt-3 p-3 border-l-4 border-amber-400 text-sm">
temperature 連續地重新分配質量;top-p 把尾部截斷後重新歸一化。兩者都降低分布的熵,差別在可微性與截斷位置是否隨 context 變動。
</div>

<!--
第四個面板那幾個 × 是重點:top-p 讓那些 token 的機率成為嚴格的 0,
而模型自己不會輸出 0(第一堂 forward KL 那頁,懲罰是 +∞)。

所以 top-p 執行的操作,是模型在訓練時被結構性禁止的操作。④ 的第二個反面結果會回到這裡。
-->

---

# 係數的方向性

<div class="grid grid-cols-2 gap-6 mt-6">

<div>

<div class="p-4 rounded border border-cyan-400 text-sm">

### 係數增大

$w \uparrow$、$\lambda \uparrow$、$1/\beta \uparrow$、$T \downarrow$、$p \downarrow$

→ **mode-seeking**

<div class="mt-2 opacity-75">樣本品質與一致性上升,support 覆蓋與多樣性下降</div>

</div>

</div>

<div>

<div class="p-4 rounded border border-amber-400 text-sm">

### 係數減小或取負值

→ **mode-covering**

<div class="mt-2 opacity-75">覆蓋範圍與變化度上升,樣本品質與相關性下降</div>

</div>

</div>

</div>

<div v-click class="mt-6 text-center text-base">

表上六種方法各有名稱,受控的是同一個一維量。

</div>

<!--
講得快沒關係,重點在下兩頁。

實務補充:同時調 T、top-p 與 CFG 等於在同一個維度上疊三次,無法歸因。
先固定兩個再動一個。
-->

---

# 推論時施加與固化進權重

<div class="grid grid-cols-2 gap-5 mt-5 text-sm">

<div class="p-4 rounded border border-cyan-400">

### 推論時施加

每產生一個 token 重算一次方向項

<div class="mt-3 opacity-80">
CFG 需要有條件與無條件兩次前向;contrastive decoding 需要同時載入兩個模型。權重不動,推論成本上升。
</div>

</div>

<div class="p-4 rounded border border-pink-400">

### 固化進權重

以該形式的最優解作為訓練目標

<div class="mt-3 opacity-80">
訓練完成後 <katex-elem expr="p_\theta" /> 本身即為調整過的分布。推論成本與原模型相同,代價付在訓練階段。
</div>

</div>

</div>

<div v-click class="mt-5 p-3 border-l-4 border-violet-400 text-sm">
由此可以預測:把 guidance 固化進權重之後,無條件那次前向不再需要,<b>推論成本減半</b>。§④ 的實驗數據符合這個預測。
</div>

<!--
「推論成本減半」的原因就是省掉無條件前向。學生若能自己說出這句,② 就算聽懂了。
-->

---

# prompt 的作用範圍

<div class="mt-5 text-center">

$$\log p_{\text{guided}} = \underbrace{\log p_{\text{base}}}_{\color{#5edfff}\text{prompt 作用於此}} + \underbrace{w}_{\color{#ff6b9d}\text{prompt 不作用於此}}\big(\log p_A - \log p_B\big)$$

</div>

<div v-click class="mt-8 p-4 border-l-4 border-pink-400">

prompt 改變起點分布,不改變係數。**分布的銳化程度無法由 prompt 控制。**

<div class="mt-2 text-sm opacity-75">
因此多樣性不足與過度銳化這兩類問題,無法用 prompt 修正。
</div>

</div>

<div v-click class="mt-5 text-center text-sm opacity-80">
在 prompt 裡加上 <span style="font-family: var(--mono)">be creative and diverse</span> 改動的是起點;偶爾有效,是因為新的起點恰好落在分布的另一個區域。
</div>

<!--
留 3 分鐘。這是這門課對日常工作最直接的一項結論,不要趕。

具體場景:情感支持對話的回應單調,常見反應是在 prompt 裡要求多元。
它不穩定、不可複製,換一個模型就失效,因為銳化程度沒有被改動。

可用的修法在第 2 層(調 T)、第 4 層(取 n 個再挑)、第 5 層(調 β)。③④ 各講一種。

問學生:上週作業寫 prompt engineering 的那幾位,A2 寫了什麼?
答案通常是「沒有目標函數」,正好對上這一頁。
-->

---
layout: section
class: sec-compute
---

# ③ 推論時的介入

## 第 1–4 層

<!--
42–84 分,四層各約 12 / 8 / 12 / 10 分。

全段不斷提示:這些方法都不需要訓練。⑥ 的結論由此而來。
-->

---

# 第 1 層 · 條件變數

<div class="mt-3">

把 prompt 視為條件變數 $c$,in-context learning 可以寫成對任務的隱式貝氏推論:

$$p(y\mid \text{prompt})=\int p(y\mid \text{task})\;p(\text{task}\mid \text{prompt})\;d\,\text{task}$$

</div>

<div v-click class="mt-5 grid grid-cols-2 gap-5 text-sm">

<div class="p-3 rounded border border-cyan-400">
<b>示例的作用</b><br>
<span class="opacity-75">把後驗 p(task | prompt) 推向某一個既有任務,而非為模型加入新能力。</span>
</div>

<div class="p-3 rounded border border-violet-400">
<b>對記憶系統的意義</b><br>
<span class="opacity-75">記憶的功能是選擇哪些證據進入這個後驗。存入的內容若不改變後驗,對輸出沒有影響。</span>
</div>

</div>

<!--
Xie et al. (ICLR 2022)。memory agent 組必讀。

這個框架的價值在於它讓「記憶要存什麼」成為可推導的問題。

不要在這裡爭論這個貝氏解釋是否為 LLM 的真實機制。它是一個有預測力的模型,
下一頁那個現象就是它預測出來的。
-->

---

# 無關 context 對後驗的稀釋

<IclBayes />

<div v-click class="mt-3 p-3 border-l-4 border-pink-400 text-sm">
右側面板是 lost-in-the-middle 與 position bias 的機率表述:無關的 context 未將後驗推向任何任務,只是把它重新攤平。<br>
<span class="opacity-70">即使 context window 容量充足,任務準確率仍隨無關內容的比例上升而下降。</span>
</div>

<!--
memory agent 組的核心頁。停 1 分鐘。

實務推論:檢索回來的 20 段裡若有 15 段無關,那 15 段是有害的。
memory agent 該優化的指標是進入後驗的證據信噪比,而召回率不能反映這件事。

這可以直接變成實驗設計:固定相關證據不變,逐步注入無關 context,
畫出任務準確率的衰減曲線。⑥ 的第四個題目。
-->

---

# RAG 與微調:條件的兩種攤提方式

<div class="grid grid-cols-2 gap-5 mt-5 text-sm">

<div class="p-4 rounded border border-cyan-400">

### RAG · 顯式條件

每次推論將證據放進 $c$

<div class="mt-3 opacity-80">
可更新、可追溯來源。代價是每次都要付 context 長度的成本,並受上一頁的稀釋效應影響。
</div>

</div>

<div class="p-4 rounded border border-pink-400">

### 微調 · 攤提進權重

把條件在訓練分布上平均後寫入 $\theta$

<div class="mt-3 opacity-80">
推論時零額外成本。代價是無法逐次更新、無法追溯來源,且模型取得的是訓練分布的平均條件。
</div>

</div>

</div>

<div v-click class="mt-5 text-center text-base">

失效模式因此可以預測:RAG 受限於證據品質,微調受限於條件與當下情境的匹配度。

</div>

<!--
「這題該用 RAG 還是微調」應該換成「這個條件每次都不同,還是每次都一樣?」
每次都不同就不能攤提。

順帶處理一個常見誤解:微調改變的是 p_θ 這個分布,不是往模型裡「加入知識」。
④ 會回到 SFT 在 forward KL 下的位置。
-->

---

# 第 2 層 · 取樣

<div class="grid grid-cols-2 gap-6 mt-4 text-sm">

<div>

### temperature

$$p_T(x_t) = \frac{\exp(z_t/T)}{\sum_j \exp(z_j/T)}$$

直接調整分布的熵。$T\to 0$ 收斂到 argmax,$T\to\infty$ 收斂到均勻分布。

</div>

<div>

### top-p(nucleus)

累積機率達到 $p$ 為止,其餘設為 0 後重新歸一化。

截斷位置隨 context 變動:分布尖銳時截去較少,平坦時截去較多。

</div>

</div>

<div v-click class="mt-5 p-3 border-l-4 border-cyan-400 text-sm">
兩者都不改變 token 的排序,只改變質量分配。因此 greedy 解碼的輸出與 T、top-p 無關。
</div>

<!--
最後那句常被誤解。有人回報「調了 temperature 但輸出沒變」,原因是在跑 greedy 或 beam。

min-p 值得一提:以最高機率 token 的某個比例當門檻,分布尖時比 top-p 保守、
分布平時比 top-p 寬鬆,實務上比 top-p 穩定。
-->

---

# 取樣設定的選擇:情感支持系統

<div class="grid grid-cols-3 gap-4 mt-6 text-sm">

<div class="p-4 rounded border border-cyan-400">
<b>低 T · 緊 top-p</b><br>
<span class="opacity-75">輸出可控且可複製。重複取樣的回應高度相似,使用者數次互動後即可辨識出模板。</span>
</div>

<div class="p-4 rounded border border-amber-400">
<b>高 T · 寬 top-p</b><br>
<span class="opacity-75">回應變化度上升。不當回應的機率同時上升,而這一題的不當回應成本很高。</span>
</div>

<div class="p-4 rounded border border-violet-400">
<b>分層</b><br>
<span class="opacity-75">高 T 取 n 個候選,再以安全性判準篩選。介入點移到第 4 層。</span>
</div>

</div>

<div v-click class="mt-6 text-center text-base">

前兩欄的取捨在取樣層內無解;第三欄把它移到另一層處理。

</div>

<!--
情感支持組在這裡要停。他們現在多半用預設值(T=1 或 0.7),那不是選擇,是沒選。

不要給建議數值。讓他們自己做校準實驗,那本身就是一個題目。
-->

---

# 第 3 層 · 約束解碼

<div class="mt-4 text-sm">

每一步把機率重新分配到合法的 token 子集上:

$$p_{\text{constrained}}(x_t \mid x_{<t}) = \frac{p(x_t\mid x_{<t})\cdot \mathbb{1}[x_t \in \mathcal{V}_{\text{legal}}]}{\sum_{v \in \mathcal{V}_{\text{legal}}} p(v\mid x_{<t})}$$

合法集合由一個文法或 schema 在解碼過程中即時決定。

</div>

<div v-click class="mt-5 p-4 border-l-4 border-cyan-400 text-sm">

**輸出合法性由解碼器保證。** 在 prompt 裡要求模型只輸出 JSON,提供的是統計上的傾向;約束解碼把非法 token 的機率設為 0,提供的是硬性保證。

</div>

<div v-click class="mt-4 text-xs opacity-70">
代價:合法集合若排除了模型原本的高機率路徑,質量會被擠到品質較低的合法路徑上。
</div>

<!--
最後那個 caveat 要講。常見現象:強制 JSON 之後內容品質下降,因為模型「想」先寫推理再給答案,
而 schema 不允許。修法是把推理欄位寫進 schema。

工具:outlines、guidance、llama.cpp 的 GBNF、OpenAI 的 structured outputs 都在這一層。
-->

---

# 第 3 層 · 對比解碼

<div class="mt-3 text-sm">

$$\log p_{\text{final}} = \log p_{\text{strong}} + \lambda\big(\log p_{\text{strong}} - \log p_{\text{weak}}\big)$$

</div>

<div class="grid grid-cols-3 gap-4 mt-4 text-sm">

<div class="p-3 rounded border border-cyan-400">
<b>contrastive decoding</b><br>
<span class="opacity-75">weak 為同族的小模型。放大大模型相對於小模型的增益部分。</span>
</div>

<div class="p-3 rounded border border-violet-400">
<b>DoLa</b><br>
<span class="opacity-75">weak 為同一模型的較早層。只需一次前向。</span>
</div>

<div class="p-3 rounded border border-pink-400">
<b>CFG for LLM</b><br>
<span class="opacity-75">weak 為移除 prompt 後的無條件分布。放大該 prompt 造成的差異。</span>
</div>

</div>

<div v-click class="mt-5 p-3 border-l-4 border-amber-400 text-sm">
三者共用 §② 的同一個式子,差別只在 <katex-elem expr="p_{\text{weak}}" /> 的來源。影像擴散的 CFG 與 LLM 的對比解碼是同一個操作。
</div>

<!--
這頁兌現 ②。學生若在這裡露出「原來如此」的表情,② 那 30 分鐘就值回票價。

實務提醒:λ 太大會造成語法退化,因為被放大的差異裡也包含雜訊。
通常要搭配一個 plausibility 門檻,只在 p_strong 已經夠高的 token 上施加對比。
那個門檻本身又是一次截斷。

成本:contrastive decoding 要載入兩個模型;DoLa 只要一次前向,對這個實驗室最實際。
-->

---

# 第 4 層 · 樣本聚合

<div class="grid grid-cols-3 gap-4 mt-5 text-sm">

<div class="p-3 rounded border border-cyan-400">
<b>best-of-n</b><br>
<span class="opacity-75">取 n 個樣本,以評分器選出一個。評分器決定分布往哪個方向移動。</span>
</div>

<div class="p-3 rounded border border-violet-400">
<b>MBR 解碼</b><br>
<span class="opacity-75">選出與其他樣本平均相似度最高者,而非機率最高者。</span>
</div>

<div class="p-3 rounded border border-pink-400">
<b>reranking</b><br>
<span class="opacity-75">以另一個模型的分數重排候選,該分數通常是一個似然比。</span>
</div>

</div>

<div v-click class="mt-5 p-3 border-l-4 border-amber-400 text-sm">
MBR 選擇的是樣本間的共識,因此對長度偏誤與 tokenization 的敏感度低於序列 log-prob。<br>
<span class="opacity-70">序列 log-prob 最高的候選,經常是最短、最保守的那個。</span>
</div>

<div v-click class="mt-3 text-center text-sm opacity-80">
這一層的成本是 <katex-elem expr="n" /> 倍的前向計算,不需要梯度。
</div>

<!--
best-of-n 的評分器若是 reward model,就繼承 reward model 的盲點(④);
若是規則(長度、格式、安全性),盲點換成規則本身的盲點。挑評分器等於挑失效模式。
-->

---

# LLM-ASR:噪聲通道模型

<div class="mt-4 text-center">

$$p(\text{text}\mid\text{audio}) \;\propto\; p(\text{audio}\mid\text{text})\;p(\text{text})$$

$$\log p(\text{text}\mid\text{audio}) = \log p_{\text{acoustic}} + \log p_{\text{LM}} + \text{const}$$

</div>

<div v-click class="mt-6 p-4 border-l-4 border-violet-400 text-sm">

第二行與 §② 的一般形式同構,兩項都是完整的對數機率。

<div class="mt-2 opacity-75">
傳統 ASR 的解碼式寫成 <katex-elem expr="\log p_{\text{acoustic}} + \alpha \log p_{\text{LM}} + \beta \cdot |\text{words}|" />,
其中 <katex-elem expr="\alpha" />(LM weight)扮演 §② 的係數,<katex-elem expr="\beta" />(insertion penalty)補償長度偏誤。
</div>

</div>

<!--
LLM-ASR 組專段。訊息:他們每天在調的那兩個超參數,是 ② 那個形式裡的係數,
1990 年代的 ASR 解碼器就有了。

insertion penalty 存在的原因是長度偏誤:每多一個詞就多乘一個小於 1 的機率,
不補償的話解碼器偏好短句。⑤ 會再遇到一次同樣的問題。
-->

---

# LLM-ASR 的三種接法

<div class="text-sm">

| 做法 | 層 | 動到的項 |
|---|---|---|
| speech encoder 接進 LLM | 1 · 條件 | 聲學表徵作為條件 $c$,起點分布改變 |
| n-best rescoring | 4 · 聚合 | 以 $\log p_{\text{LM}}$ 重排既有候選 |
| LLM 後編輯糾錯 | 4 · 聚合 | 以另一個分布重估輸出 |
| 調 LM weight | 2 / 3 | 直接改動係數 |

</div>

<div v-click class="mt-5 p-3 border-l-4 border-cyan-400 text-sm">
把 LM 分數改用 bits per byte 尺度,超參數在更換 LM 後仍大致可移植。<br>
<span class="opacity-70">BPB 的分母是位元組數,與 tokenizer 無關,因此不同 LM 的分數落在可比較的尺度上。</span>
</div>

<!--
第一堂 ④ 的 BPB 在這裡第一次用上,⑤ 會再用一次。

實務上換 LM 之後 LM weight 要重新掃,原因就是各 tokenizer 的 log-prob 尺度不同。
改成 BPB 至少緩解一半。⑥ 的第五個題目,可以直接跑。
-->

---
layout: section
class: sec-loss
---

# 休息 10 分鐘

## 前半場:不動參數 · 後半場:改動權重

<!--
84–94 分。

回來的第一件事是 RLHF 的目標函數。
-->

---
layout: section
class: sec-family
---

# ④ 權重層的介入

## 第 5–6 層

<!--
94–142 分。SFT + RLHF 18 / DDO 22 / 收束 8。

兩個重點:RLHF 的 reverse KL 推導,以及 DDO 損失的兩個期望值。
-->

---

# SFT 的位置

<div class="mt-4">

在新資料上做監督微調,目標函數為

$$\max_\theta \;\mathbb{E}_{(x,y)\sim \mathcal{D}_{\text{SFT}}}\big[\log \pi_\theta(y\mid x)\big]$$

</div>

<div v-click class="mt-5 p-4 border-l-4 border-cyan-400 text-sm">

這是 MLE,等價於 forward KL,只是換了資料集。

<div class="mt-2 opacity-75">
forward KL 的所有性質原封不動地繼承:zero-avoiding、teacher forcing、exposure bias,以及不會把任何 plausible token 的機率壓到 0。
</div>

</div>

<div v-click class="mt-5 text-center text-base">

SFT 改變模型的行為與輸出格式,但 mode-covering 的性質保持不變。

</div>

<!--
這頁是後面的對照組。學生要能區分:
「SFT 之後模型會照格式回答」是起點分布換了,銳化程度沒換。

第一堂那條軸的第 3 列把 SFT 放在左端,就是這個意思。
-->

---

# RLHF 的目標函數

<div class="mt-4 text-center">

$$\max_{\pi}\; \mathbb{E}_{y\sim\pi}\big[r(y)\big] \;-\; \beta\,\mathrm{KL}\big(\pi \,\|\, \pi_{\text{ref}}\big)$$

</div>

<div class="mt-5 grid grid-cols-2 gap-5 text-sm">

<div class="p-3 rounded border border-amber-400">
<b>第一項</b><br>
<span class="opacity-75">最大化 reward。reward model 由人類偏好資料訓練而得。</span>
</div>

<div class="p-3 rounded border border-violet-400">
<b>第二項</b><br>
<span class="opacity-75">限制與參考模型的距離。移除此項會導致 reward hacking 與語言退化。</span>
</div>

</div>

<div v-click class="mt-5 p-3 border-l-4 border-pink-400 text-sm">
注意 KL 的寫法:左側為 <katex-elem expr="\pi" />(訓練中的模型),右側為 <katex-elem expr="\pi_{\text{ref}}" />。第一堂已建立,左側的分布決定積分的權重,也就決定了 mode-covering 或 mode-seeking。
</div>

<!--
學生要親眼看到 π 寫在左邊。

第一堂的口訣可以複誦:forward KL 問「資料出現的地方有沒有給機率」;
reverse KL 問「放了機率的地方資料是否存在」。這裡的「資料」換成了參考模型。
-->

---

# 最優解與優化方向

<div class="text-sm mt-3">

該目標函數有閉式解:

$$\pi^*(y) \;=\; \frac{1}{Z}\,\pi_{\text{ref}}(y)\,\exp\!\big(r(y)/\beta\big)
\qquad\Longleftrightarrow\qquad
\log \pi^* = \log\pi_{\text{ref}} + \tfrac{1}{\beta}\,r(y) - \log Z$$

</div>

<div v-click class="mt-4 p-3 border-l-4 border-violet-400 text-sm">
右式即 §② 表格的第五列:base 為 <katex-elem expr="\log\pi_{\text{ref}}" />,方向為 <katex-elem expr="r(y)" />,係數為 <katex-elem expr="1/\beta" />。
</div>

<div v-click class="mt-4 text-sm">

把目標函數改寫,可以看出優化過程的方向:

$$\mathbb{E}_{y\sim\pi}[r(y)]-\beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}}) \;=\; -\beta\,\mathrm{KL}\big(\pi \,\|\, \pi^*\big) + \beta\log Z$$

</div>

<div v-click class="mt-3 p-4 border-l-4 border-pink-400">

$\log Z$ 與 $\pi$ 無關,因此最大化左式等價於最小化 $\mathrm{KL}(\pi\|\pi^*)$。

<div class="mt-2 text-sm opacity-80">
<katex-elem expr="\pi" /> 位於 KL 的左側,這是 <b>reverse KL</b>,mode-seeking。
</div>

</div>

<!--
兩個式子各停 30 秒。

第二個式子的推導只是把 exp(r/β) 代回 KL 的定義再整理,一行,可以請學生課後自己補。

強調:這不是某個實作的選擇,是目標函數寫成那樣之後的必然。
對齊是一個 mode-seeking 的操作,與工程細節無關。
-->

---

# $\beta$ 控制的取捨

<div class="mt-4">
  <SpectrumAxis :rows="3" />
</div>

<div v-click class="mt-5 p-4 border-l-4 border-pink-400 text-sm">

$\beta$ 減小 → KL 約束放鬆 → 更用力最大化 reward → 更 mode-seeking;$\beta$ 增大則趨近參考模型。

<div class="mt-2 opacity-75">
對齊後模型的多樣性下降,是目標函數的數學後果,而非實作缺陷。
</div>

</div>

<div v-click class="mt-4 text-center text-base">

對情感支持系統:**安全性與回應多樣性由同一個 $\beta$ 控制。**

</div>

<!--
情感支持組要點名:回應千篇一律不是模型不夠好,是對齊目標函數的後果。
要多樣性就得付安全性的代價,或者換一層介入(第 3、4 層)。

上週作業 A4 預測「多樣性喪失」的那幾位,現在可以告訴他們預測對了,理由在這裡。
-->

---

# 逐點評分器的共同限制

<div class="grid grid-cols-2 gap-5 mt-5 text-sm">

<div class="p-4 rounded border border-pink-400">

### GAN 的生成器目標

$$\mathbb{E}_{x\sim p_g}\big[\log D(x)\big]$$

期望值取在模型自身的樣本上,$D$ 為逐點評分。

<div class="mt-2 opacity-75">
覆蓋度是分布層級的性質,判別器的輸入空間裡沒有這個量。
</div>

</div>

<div class="p-4 rounded border border-violet-400">

### RLHF 的策略目標

$$\mathbb{E}_{y\sim\pi}\big[r(y)\big]$$

期望值取在模型自身的樣本上,$r$ 為逐點評分。

<div class="mt-2 opacity-75">
reward model 無法表達「回應分布過窄」這個判斷。
</div>

</div>

</div>

<div v-click class="mt-5 p-3 border-l-4 border-amber-400 text-sm">
唯一約束塌縮的是 <katex-elem expr="\beta" /> 這個錨,而它約束的量是與參考模型的距離,不是多樣性本身。<br>
<span class="opacity-70">LLM-as-judge 具有相同的限制:逐則評分無法度量一批回應之間的差異程度。</span>
</div>

<!--
第一堂 ⑥「認出對抗的句型」在這裡兌現。

實務推論:要量多樣性必須另外設計分布層級的指標(distinct-n、self-BLEU、語意聚類數),
而它不可能從 reward model 得到。

對用 LLM-as-judge 做評估的人是直接的警告:judge 分數上升不代表系統沒有塌縮。
-->

---

# 第 6 層 · 表徵層介入

<div class="mt-4 text-sm">

推論時將一個概念方向加到隱藏狀態上:

$$h' \;=\; h \;+\; \alpha\,v_{\text{concept}}$$

其中 $v_{\text{concept}}$ 通常取自兩組對比提示的平均啟動之差。

</div>

<div v-click class="mt-5 p-3 border-l-4 border-violet-400 text-sm">
形式與 §② 相同:一個起點、一個差、一個係數。相減發生在表徵空間而非對數機率空間。
</div>

<div v-click class="mt-4 text-xs opacity-70">
本課只介紹到這裡。它的實作門檻高於第 3、4 層,穩定性目前也不如它們。
</div>

<!--
1 分鐘帶過。

想深入的看 representation engineering 那一系列。對這個實驗室,
第 3、4 層的投報率明顯更高,不建議現在投入。
-->

---

# 訓練軌道與推論軌道

<TwoTracks />

<div v-click class="mt-3 text-center text-base">

訓練目標的期望值取自 $p_{\text{data}}$ 的前綴。**模型自己生成的前綴,要如何進入目標函數?**

</div>

<!--
142 分附近。DDO 段開始,約 22 分。

先讓學生回想第一堂的結論:訓練 loss 的期望值下標取自 p,不是 q,
所以模型自己生成的前綴從來沒有被目標函數看過。

停 30 秒讓他們自己想答案,再翻頁。
-->

---

# 三種把模型樣本納入目標的方法

<div class="grid grid-cols-3 gap-4 mt-6 text-sm">

<div class="p-4 rounded border border-cyan-400">
<b>scheduled sampling</b><br>
<span class="opacity-75">訓練時餵入模型自己的前綴。最直接,但破壞目標函數的統計性質。</span>
</div>

<div class="p-4 rounded border border-violet-400">
<b>RL / DPO</b><br>
<span class="opacity-75">在模型自身樣本上取期望值,訊號來自 reward 或成對偏好。</span>
</div>

<div class="p-4 rounded border border-pink-400">
<b>DDO</b><br>
<span class="opacity-75">在模型自身樣本上取期望值,訊號來自似然比。</span>
</div>

</div>

<div v-click class="mt-6 p-4 border-l-4 border-amber-400">

三者的共同結構:**負向訊號取自模型自己生成的樣本。**

<div class="mt-2 text-sm opacity-75">
MLE 只有正向訊號,在資料點上提高機率,沒有任何項去降低模型在無資料區域配置的質量。
</div>

</div>

<!--
最後那句是這一段的軸心,梯度那頁會用數學再講一次。

很多人把 DPO 歸類成「RL 的簡化版」。真正的分類依據是期望值的下標取自哪個分布,
不是有沒有用到 policy gradient。
-->

---

# DDO 的判別器參數化

<div class="text-sm mt-3">

GAN 的最佳判別器可以寫成似然比的 sigmoid:

$$d^*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_{\theta_{\text{ref}}}(x)}=\sigma\!\left(\log\frac{p_{\text{data}}(x)}{p_{\theta_{\text{ref}}}(x)}\right)$$

</div>

<div v-click class="text-sm mt-4">

likelihood-based 模型可以直接計算 $\log p_\theta$,因此令

$$d_\theta(x) := \sigma\!\left(\log\frac{p_\theta(x)}{p_{\theta_{\text{ref}}}(x)}\right)$$

再以標準 GAN 判別器的 BCE loss 訓練。最優解落在 $p_\theta = p_{\text{data}}$。

</div>

<div v-click class="mt-4 p-3 border-l-4 border-cyan-400 text-sm">
不需要額外的判別器網路,不需要交替訓練,<b>不需要對取樣過程反向傳播</b>。最後一項對迭代取樣的擴散模型與 AR 模型是決定性的。
</div>

<!--
這兩行是本堂最漂亮的地方,慢慢講。

第一行是第一堂 ④ GAN 段講過的,學生應該認得。關鍵是把它寫成 σ(log ratio),
因為那個 log ratio 正好是 likelihood-based 模型算得出來的東西。

第二行的動作只有一個:把 p_data 換成 p_θ。
最優解的驗證只要把 p_θ = p_data 代回 BCE 的一階條件。
-->

---

# DDO 的訓練流程

<DdoMechanism />

<div v-click class="mt-3 p-3 border-l-4 border-violet-400 text-sm">
與 GAN 的差別集中在中間那個框:判別器由 <katex-elem expr="p_\theta" /> 與 <katex-elem expr="p_{\text{ref}}" /> 的似然比直接參數化,沒有獨立的參數。
</div>

<!--
指著虛線講 self-play:每一輪結束後,參考模型換成本輪的最佳模型,再訓練下一輪。
論文每輪的 fine-tune 量不到預訓練 epoch 的 1%。

這與 SPIN、iterative DPO 是同一個 self-play 骨架,差別在訊號的定義。
-->

---

# 梯度形式

<div class="mt-4 text-center">

$$\nabla_\theta L=\int \big(1-d_\theta(x)\big)\big(p_\theta(x)-p_{\text{data}}(x)\big)\,\nabla_\theta \log p_\theta(x)\,dx$$

</div>

<div class="grid grid-cols-2 gap-5 mt-5 text-sm">

<div class="p-4 rounded border border-cyan-400">
<b>p_θ 小於 p_data 之處</b><br>
<span class="opacity-75">括號為負,沿 <katex-elem expr="\nabla_\theta \log p_\theta" /> 提高機率。與 MLE 同向。</span>
</div>

<div class="p-4 rounded border border-pink-400">
<b>p_θ 大於 p_data 之處</b><br>
<span class="opacity-75">括號為正,降低機率。MLE 的梯度沒有這一項。</span>
</div>

</div>

<div v-click class="mt-5 p-4 border-l-4 border-amber-400">

MLE 只在資料點上提高機率。**模型在無資料區域配置的質量,不受任何梯度約束。**

<div class="mt-2 text-sm opacity-75">
幻覺與離群樣本有一部分來自這些區域。
</div>

</div>

<!--
第一堂 forward KL 的性質在這裡全部找到對照。

回到第一堂 GAN 段那個「亮著但沒人去」的畫面:判別器地景上有高分區域但生成器沒去。
DDO 的第二項處理相反的情況:生成器去了但資料不在。
-->

---

# DDO 損失的兩個期望值

<div class="mt-5 text-center text-sm">

$$L_{\text{DDO}} \;=\; -\,\mathbb{E}_{x\sim p_{\text{data}}}\big[\log d_\theta(x)\big]\;-\;\mathbb{E}_{x\sim p_{\text{ref}}}\big[\log\big(1-d_\theta(x)\big)\big]$$

</div>

<div class="grid grid-cols-2 gap-5 mt-5 text-sm">

<div class="p-4 rounded border border-cyan-400">
<b>左項 · 期望值取在 p_data 上</b><br>
<span class="opacity-75">要求覆蓋資料的 support,與 forward KL 同向。</span>
</div>

<div class="p-4 rounded border border-pink-400">
<b>右項 · 期望值取在模型側樣本上</b><br>
<span class="opacity-75">降低無資料區域的質量,與 reverse KL 同向。</span>
</div>

</div>

<div v-click class="mt-5 p-4 border-l-4 border-violet-400">

損失同時包含兩個方向的梯度。**這是本課介紹的方法中唯一不需要在覆蓋與銳度之間預先選邊的。**

<div class="mt-2 text-sm opacity-75">
下一頁的 <katex-elem expr="\beta<1" /> 情形是例外,取捨在那裡重新出現。
</div>

</div>

<!--
不要誇大:兩個方向的力並存,不代表取捨消失了。caveat 二會誠實交代 β<1 時
最優解其實是 p_data 的銳化版。這裡說的是 loss 的結構,不是最終效果。
-->

---

# 與 DPO 的關係

<div class="text-sm mt-3">

| | DPO | DDO |
|---|---|---|
| 隱式參數化 | reward $=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ | discriminator $=\sigma\big(\beta\log\frac{p_\theta}{p_{\text{ref}}}\big)$ |
| 目標 | 偏好學習 | 分布對齊 |
| 資料 | 成對的人工標註 | 原始訓練資料與模型樣本,不需配對 |
| 優化對象 | winner 與 loser 的機率差距 | 整個分布向 $p_{\text{data}}$ 對齊 |

</div>

<div v-click class="mt-5 p-3 border-l-4 border-cyan-400 text-sm">
兩者的隱式參數化是同一個似然比,分別代入 Bradley–Terry 與 BCE 兩種損失。
</div>

<div v-click class="mt-3 text-sm opacity-80 text-center">
第三列是實務上的主要差別:DDO 不需要成對標註,資料成本低一個數量級。
</div>

<!--
第三列對這個實驗室最重要。DPO 要收成對偏好資料,那是最貴的部分;
DDO 只需要一批真實資料加一批模型樣本。

但注意:DDO 對齊的是 p_data,不是人類偏好。目標若是偏好而非分布,DPO 仍是對的工具。
-->

---

# 與推論時 guidance 的對應

<div class="mt-4 text-sm">

實作上需要引入 $\alpha,\beta$ 兩個係數,因為 $\log p_\theta$ 的量級可達 $10^3$,直接代入 sigmoid 會飽和。加入之後,最優解為

$$p_\theta^*(x) \;\propto\; p_{\text{ref}}(x)^{\,1-1/\beta}\;p_{\text{data}}(x)^{\,1/\beta}$$

</div>

<div v-click class="mt-4 p-3 border-l-4 border-violet-400 text-sm">
取對數:<katex-elem expr="\log p_\theta^* = \log p_{\text{ref}} + \tfrac{1}{\beta}\big(\log p_{\text{data}} - \log p_{\text{ref}}\big)" />,即 §② 表格的最後一列。
</div>

<div v-click class="mt-5 p-4 border-l-4 border-amber-400 text-center text-base">

guidance 在推論時執行這個銳化,DDO 把同一個最優解寫進權重。

</div>

<!--
② 的第二個結論在這裡兌現。② 講的時候是預告,現在是推導結果。

β<1 時指數 1/β > 1,p_data 被升冪,即銳化;β=1 時最優解正好是 p_data。
下一頁的 overshoot 就是這個。
-->

---

# 實驗結果:影像生成

<div class="text-sm mt-3">

| 模型 · 資料集 | guidance-free FID(前 → 後) |
|---|---|
| EDM2-L · ImageNet 512 | 1.96 → **1.26** |
| EDM2-S · ImageNet-64 | 1.60 → **0.97** |
| EDM · CIFAR-10 | 1.97 → 1.38 / 1.85 → **1.30** |
| VAR-d30 · guidance-free | 4.74 → **1.79** |

</div>

<div v-click class="mt-4 p-3 border-l-4 border-cyan-400 text-sm">
最後一列的對照是原始 VAR 搭配 CFG 的 1.90。DDO 在不使用 guidance 的情況下取得更低的 FID,且推論成本減半,因為省去了無條件那次前向。
</div>

<div v-click class="mt-3 text-xs opacity-70">
訓練代價:每一輪 fine-tune 不到預訓練 epoch 的 1%。
</div>

<!--
Zheng et al., ICML 2025, arXiv 2503.01103。

推論成本減半在 ② 已經推導過,這裡是實證。學生應該覺得理所當然;
如果覺得驚訝,表示 ② 沒聽進去。
-->

---

# 適用範圍

<div class="grid grid-cols-2 gap-5 mt-5 text-sm">

<div class="p-4 rounded border border-pink-400">

### 尚未驗證的部分

DDO 的實驗全部在影像生成上,文字生成沒有同等規模的驗證。

<div class="mt-2 opacity-75">
FID 本身偏好銳化的分布,對這個方法是有利的評估指標。
</div>

</div>

<div class="p-4 rounded border border-cyan-400">

### 形式相近的工作

**SPIN** 與 **iterative DPO** 是文字側的 self-play 微調。

<div class="mt-2 opacity-75">
共用 self-play 結構與似然比參數化,但推導與目標不同,不能作為 DDO 在文字上的等價結果引用。
</div>

</div>

</div>

<div v-click class="mt-5 p-3 border-l-4 border-amber-400 text-sm">
由此得到一個可執行的題目:<b>把 DDO 的判別式目標移植到文字生成,並找出一個不偏好銳化的評估指標。</b><br>
<span class="opacity-70">實作不是難點,後半句才是。</span>
</div>

<!--
把影像結果改述成「LLM 上也如此」會是超出證據的斷言,學生若照著寫進論文會被審稿人抓。

後半句值得展開:FID 偏好銳化,文字側常用的指標(perplexity、勝率)也各有偏向。
要證明 DDO 在文字上有效,得先解決「怎麼同時量品質與多樣性」,那正好是 ⑤ 的內容。
-->

---

# 兩個反面結果

<div class="mt-4 space-y-4 text-sm">

<div class="p-4 rounded border border-violet-400">

### 一 · 繼續以原本的 MLE loss 訓練,結果退化

<div class="mt-2 opacity-80">
排除了超參數設定的解釋,指向 forward KL 這個目標本身已達上限。<br>
更換目標函數與調整超參數在此有可分辨的效果差異。
</div>

</div>

<div class="p-4 rounded border border-amber-400">

### 二 · 原始 VAR 的數字依賴 top-k / top-p 取樣

<div class="mt-2 opacity-80">
這些啟發式實際上在降低有效溫度,拉開訓練與推論的分布差距。<br>
以 decoding 參數截窄分布,遮蓋了模型分布本身的缺陷。
</div>

</div>

</div>

<!--
第二點回到 ③ TempTopP 那頁:top-p 讓某些 token 的機率成為嚴格的 0,
而那是模型在訓練時被結構性禁止的操作。用它來補救,等於在推論時換掉訓練目標。

方法論意義:報告數字時 decoding 設定必須一起報,
否則無法判斷改善來自模型還是來自截斷。
-->

---

# 三個限制

<div class="mt-4 space-y-3 text-sm">

<div class="p-3 rounded border border-cyan-400">
<b>一 · <katex-elem expr="\alpha,\beta" /> 需要 grid search</b><br>
<span class="opacity-75">論文每輪掃約 20 個節點:<katex-elem expr="\alpha\in[0.5,50]" />、<katex-elem expr="\beta\in[0.01,0.1]" />。</span>
</div>

<div class="p-3 rounded border border-pink-400">
<b>二 · <katex-elem expr="\beta<1" /> 時最優解是 p_data 的銳化版</b><br>
<span class="opacity-75">銳度與多樣性的取捨仍然存在,發生位置由推論時移到訓練時。FID 偏好銳化,因此這個 overshoot 在影像上呈現為改善。對情感支持這類題目,同一個係數的最佳方向未必相同。</span>
</div>

<div class="p-3 rounded border border-violet-400">
<b>三 · divergence bound 要求似然比有界</b><br>
<span class="opacity-75">須 <katex-elem expr="\log(p_{\text{ref}}/p_{\text{data}})" /> 與 <katex-elem expr="\log(p_\theta/p_{\text{ref}})" /> 皆有界,因此只在短時間微調的設定下成立,與 DPO 中 <katex-elem expr="\beta" /> 的角色一致。</span>
</div>

</div>

<!--
限制二最重要,它防止學生把 DDO 當成免費解決取捨的方法。

「發生位置移動」要說清楚:以前是推論時調 CFG,現在是訓練時調 β。
取捨沒有消失,只是變成一次性的決定,而且改起來更貴。

限制一與三合起來看:這是一個微調方法,不是預訓練方法。
-->

---

# 似然比在三個層次的重複出現

<div class="text-sm mt-3">

$\log\dfrac{p_\theta}{p_{\text{ref}}}$ 在六層中出現三次:

| 層 | 名稱 | 用途 |
|---|---|---|
| 3 · logits | contrastive decoding、DoLa、CFG for LLM | 推論時即時相減 |
| 4 · 聚合 | likelihood-ratio reranking、n-best rescoring | 重排候選 |
| 5 · 權重 | DPO 的隱式 reward、DDO 的隱式判別器 | 固化進參數 |

</div>

<div v-click class="mt-4 p-4 border-l-4 border-amber-400 text-sm">

第一堂用於消去 $H(p)$ 的「相對於參考模型正規化」,就是這個量。

<div class="mt-2 opacity-75">
它扣除了樣本本身的難度:難的句子兩個模型都給低機率,相減後這部分抵銷。
</div>

</div>

<!--
第一堂講 H(p) 消不掉時給了四條實務路徑,第二條是「相對於參考模型正規化」。
當時沒說那是什麼,今天它出現了三次。

⑥ 的第一個題目從這裡長出來:似然比既然扣掉了難度,作為 confidence 訊號
應該比原始 sequence log-prob 更好。這是可以下週就開始跑的實驗。
-->

---
layout: section
class: sec-vs
---

# ⑤ 分布變動的量測

## 序列分數、不確定性與校準

<!--
142–168 分。

confidence vs. accuracy 那組在這段是主角。
-->

---

# 從 token 機率到序列分數

<div class="mt-4 text-sm space-y-3">

<div class="p-3 rounded border border-cyan-400">
<b>token 機率</b> —— 單一位置的條件分布。受 tokenization 影響:同一個詞被切成兩片,機率就成為兩個數的乘積。
</div>

<div class="p-3 rounded border border-violet-400">
<b>序列 log-prob</b> <katex-elem expr="\sum_t \log p(x_t\mid x_{<t})" /> —— 具有長度偏誤,每個額外 token 都加上一個負數。
</div>

<div class="p-3 rounded border border-pink-400">
<b>長度正規化</b> —— 除以 token 數會偏好長而平淡的句子;除以位元組數(BPB)才跨 tokenizer 可比。
</div>

</div>

<div v-click class="mt-4 p-3 border-l-4 border-amber-400 text-sm">
三者度量不同的量。分數的選擇應由待測的性質決定,選錯會使整組實驗的結論失效。
</div>

<!--
長度偏誤在 ③ 的 insertion penalty 已經遇過一次。同一個問題在 ASR 解碼與 LLM 評估各出現一次,
兩邊發明了不同的補救。

實務提醒:HuggingFace 的 generate 有 length_penalty,作用在 beam search 上,
與這裡的評估用正規化不是同一件事,不要混用。
-->

---

# predictive entropy 的失效

<SemanticEntropy />

<div v-click class="mt-3 p-3 border-l-4 border-pink-400 text-sm">
「巴黎」與「法國巴黎」是兩個 token 序列、一個語意。把它們視為兩個結果去算熵,度量到的是表達形式的變異,而非知識的不確定性。
</div>

<!--
confidence 組的核心頁。

semantic entropy 的做法:對同一問題取 n 個樣本,用 NLI 模型判斷雙向蘊涵,
互相蘊涵者歸為同一類,再對類別分布算熵。

一個被低估的優點:聚類發生在語意層,換 tokenizer 或換模型都不影響類別劃分,
因此跨模型比較時特別有用。

Kuhn, Gal & Farquhar (ICLR 2023);Farquhar et al., Nature (2024)。
-->

---

# self-consistency 作為邊際化

<div class="mt-5 text-center">

$$p(a\mid q) \;=\; \sum_{r} p(a\mid r,q)\,p(r\mid q) \;\approx\; \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}\big[a_i = a\big]$$

</div>

<div v-click class="mt-6 p-4 border-l-4 border-violet-400 text-sm">

推理鏈 $r$ 是潛在變數,重複取樣後投票即為對它做蒙地卡羅邊際化。

</div>

<div v-click class="mt-5 text-center text-sm opacity-80">
這個表述回答了兩個實務問題:<katex-elem expr="n" /> 的取值由蒙地卡羅估計的變異數決定;<br>
取樣溫度不能太低,否則 <katex-elem expr="n" /> 個樣本高度相關,估計量沒有變異。
</div>

<!--
最後那句是實用價值。常見錯誤:用 T=0.2 跑 self-consistency,20 個樣本幾乎相同,
投票等於沒投。溫度要夠高才有獨立性,但太高會讓每條推理鏈品質下降。

第一堂 VAE 段的潛在變數與邊際化符號在這裡用上了。

Wang et al. (2023)。
-->

---

# verbalized confidence 的偏誤

<div class="grid grid-cols-2 gap-5 mt-5 text-sm">

<div class="p-4 rounded border border-pink-400">

### 直接詢問模型的信心

取得的是經過對齊訓練塑形的輸出行為。

<div class="mt-2 opacity-75">
對齊獎勵聽起來有把握且有幫助的回答,自報信心因此被系統性推高。
</div>

</div>

<div class="p-4 rounded border border-cyan-400">

### 內部機率

logprob、entropy、似然比都可以直接計算。

<div class="mt-2 opacity-75">
它們與答案正確性之間需要一個校準函數,而該函數不是恆等映射。
</div>

</div>

</div>

<div v-click class="mt-5 p-3 border-l-4 border-amber-400 text-sm">
兩者度量不同的量:前者是被對齊目標塑形過的輸出,後者是分布的性質。兩者背離是預期中的結果。
</div>

<!--
這是 ④ RLHF 那段的直接後果,可以指回去。

不要一概而論:Kadavath et al. (2022) 顯示模型在某些設定下確實知道自己知道什麼,
所以 verbalized confidence 需要校準,而且跨任務不可移植。

對 confidence 組:這裡有兩條可比較的路線,第三條在下一頁。
-->

---

# 校準:reliability diagram 與 ECE

<Calibration />

<div v-click class="mt-3 p-3 border-l-4 border-cyan-400 text-sm">
溫度縮放不改變 logits 的排序,因此 accuracy 不變。它調整的是信心的刻度,計算成本也因此極低。
</div>

<!--
confidence vs. accuracy 那組的直接工具。今天看靜態版,互動版在待做清單上。

三件事:
1. ECE 是圖上垂直落差的加權平均,不是獨立指標。
2. 曲線在對角線下方即過度自信,RLHF 之後幾乎是慣例。
3. 溫度縮放只擬合一個純量,在驗證集上做,不動模型權重。

ECE 有自己的問題(分箱數敏感、對少數高信心樣本不敏感),要報就連分箱設定一起報。
-->

---

# 跨模型比較與多選題評估

<div class="grid grid-cols-2 gap-5 mt-5 text-sm">

<div class="p-4 rounded border border-violet-400">

### 跨模型:以位元組為分母

$$\mathrm{BPB}=\frac{T}{N_{\text{bytes}}}\cdot\log_2 \mathrm{PPL}_{\text{token}}$$

<div class="mt-2 opacity-75">
第一堂的三個 caveat 仍然適用:tokenizer 須無損、計算的是 canonical 切法(為真實字串機率的上界)、chunking 與 BOS 處理須固定。
</div>

</div>

<div class="p-4 rounded border border-cyan-400">

### 多選題:PMI 正規化

$$\log p(a\mid q) - \log p(a)$$

<div class="mt-2 opacity-75">
扣除選項本身的先驗頻率,避免常見詞組成的選項因為容易生成而勝出。
</div>

</div>

</div>

<div v-click class="mt-5 text-center text-base">

PMI 正規化同樣是一個似然比,參考分布取無條件的 $p(a)$。

</div>

<!--
最後那句是刻意的:它與 ④ 收束那頁的似然比是同一個量,只換了參考分布。
這是本堂第四次出現同一個形式,學生應該開始自己認出來。

BPB 在 ③ 的 LLM-ASR 段用過一次,這是第二次。
-->

---

# 三個尚未實作的互動工具

<div class="grid grid-cols-3 gap-4 mt-6 text-sm">

<div class="p-4 rounded border border-cyan-400">
<b>校準散點圖</b><br>
<span class="opacity-75">reliability diagram、ECE 與 temperature scaling 滑桿。優先序第一,產出的圖可以直接用於論文。</span>
</div>

<div class="p-4 rounded border border-violet-400">
<b>Token 機率瀏覽器</b><br>
<span class="opacity-75">逐 token 顯示 top-k 與 entropy,拖動 T 與 top-p 觀察分布的變化。</span>
</div>

<div class="p-4 rounded border border-pink-400">
<b>虛假前提 / 語意熵</b><br>
<span class="opacity-75">並排比較正常前提與虛假前提的 prompt BPB;對同一問題取 n 個樣本、語意聚類、顯示 semantic entropy。</span>
</div>

</div>

<div v-click class="mt-6 text-center text-sm opacity-80">
三者沿用現有 demo 的樣式與 API 呼叫方式。有意接手的請在課後告知。
</div>

<!--
第一個如果有人接,它不只是教具,是可以直接產出論文圖的工具。
第二個做起來最快,對每一屆新生都有用。

不要花時間解釋做法,讓他們知道這裡有空位即可。
-->

---
layout: section
class: sec-recap
---

# ⑥ 對應到實驗室題目

## 六個題目的機率表述與介入層

<!--
168–180 分。全部是回收,講得輕鬆一點。
-->

---

# 六個題目的機率表述

<div class="text-sm">

| 實驗室題目 | 對應的機率問題 | 性質 | 對應段落 |
|---|---|---|---|
| prompt engineering | 選擇條件變數 $c$,操控 $p(y\mid c)$ | 只改起點,不動係數 | §② |
| memory agent | 條件集合的建構;長 context 下後驗被稀釋 | 同上 | §③ 第 1 層 |
| 情感支持對話 | 通用安慰語與對齊後多樣性下降 | mode-seeking,受 $\beta$ 控制 | §④ |
| 虛假前提檢測 | $p(y\mid x)$ 恆為 well-defined,即使 $p(x)\approx 0$ | mode-covering 的結構性後果 | 第一堂 §④ |
| confidence vs. accuracy | 校準;predictive entropy 與 semantic entropy | 量測工具本身 | §⑤ |
| LLM-ASR | 噪聲通道 = 對數空間的線性組合 | 係數 $w$ 的手動版本 | §③ 第 4 層 |

</div>

<!--
逐列點名對應的組,問一句「這樣講得通嗎」。

虛假前提那一列是唯一不在本堂的,因為機制第一堂就講完了:
forward KL 訓練的模型結構上沒有拒絕回答這個選項,除非後訓練另外教它。
今天補的是後半句 —— 後訓練怎麼教,就是 ④。
-->

---
layout: center
---

<FailureScenes verdict />

<div v-click class="mt-8 text-center text-lg">

兩種失效模式的推導都完成了。

<div class="text-sm opacity-75 mt-3">
左:forward KL 的積分權重迫使 support 被完整覆蓋 · 右:reverse KL 的目標函數容許只佔住單一模式
</div>

</div>

<!--
上週這一頁放的時候右邊還是斷言,今天有了 ④ 的推導。

時間不夠的話這頁停 15 秒即可,學生記得的是那個對照。
-->

---

# 第 3、4 層的空間

<div class="mt-8">
  <GuidanceLadder :active="[3, 4]" />
</div>

<div v-click class="mt-4 p-4 border-l-4 border-amber-400 text-center text-base">

以 prompt engineering 為主的實驗室,可用的空間集中在第 3、4 層。

<div class="mt-2 text-sm opacity-75">
這兩層的成本是額外的前向計算,不需要訓練資源。
</div>

</div>

<!--
理由攤開說:第 1 層已經做到頂了;第 5、6 層需要算力與資料,短期做不了;
第 2 層只有一顆旋鈕,調完就沒了。第 3、4 層有大量現成方法沒被試過。

具體一點:contrastive decoding、DoLa、約束解碼、MBR、n-best rescoring
——這五個裡面,這個實驗室目前一個都沒在用。
-->

---

# 五個可執行的題目

<div class="text-sm space-y-2 mt-4">

<div class="p-3 rounded border border-cyan-400">
<b>一 · 似然比校準</b> — 以一個小的 reference model 計算 <katex-elem expr="\log(p_\theta/p_{\text{ref}})" />,檢驗它與 accuracy 的相關度是否高於原始 sequence log-prob。<span class="opacity-70">(confidence 組 · §④)</span>
</div>

<div class="p-3 rounded border border-violet-400">
<b>二 · 語意熵用於虛假前提</b> — 檢驗虛假前提的問題,其語意熵是否系統性高於正常前提。<span class="opacity-70">(虛假前提組 · §⑤)</span>
</div>

<div class="p-3 rounded border border-pink-400">
<b>三 · 多樣性控制移出 prompt 層</b> — 情感支持系統改以第 3、4 層(對比解碼 + MBR)控制多樣性,與調整 prompt 的基線比較。<span class="opacity-70">(情感支持組 · §②③)</span>
</div>

<div class="p-3 rounded border border-amber-400">
<b>四 · 證據稀釋曲線</b> — 固定相關證據,逐步注入無關 context,量測任務準確率的衰減。<span class="opacity-70">(memory agent 組 · §③ 第 1 層)</span>
</div>

<div class="p-3 rounded border border-cyan-400">
<b>五 · BPB 尺度的 LM weight</b> — 將 ASR 解碼的 LM 分數改為 BPB 尺度,量測超參數在更換 LM 後的可移植性。<span class="opacity-70">(LLM-ASR 組 · §③ 第 4 層)</span>
</div>

</div>

<!--
每個題目對應一組,都可以在兩週內看到第一個結果。
五個都不需要訓練大模型,四個不需要 GPU。

想認領的課後找我,我會給起手的 baseline 與資料。
不認領也沒關係,但下次組會請至少能說出自己的題目落在第幾層。
-->

---

# 課後資源

<div class="grid grid-cols-2 gap-5 text-sm">

<div>

### 引導與解碼(§②③)

- Ho & Salimans (2022) — *Classifier-Free Diffusion Guidance*
- Sanchez et al. (2023) — *Stay on topic with CFG*(LLM 側)
- Karras et al. (2024) — *Autoguidance*(arXiv:2406.02507)
- Li et al. (2023) — *Contrastive Decoding*;Chuang et al. (2024) — *DoLa*
- Holtzman et al. (2020) — *The Curious Case of Neural Text Degeneration*
- Xie et al. (ICLR 2022) — *ICL as Implicit Bayesian Inference*
- Liu et al. (2024) — *Lost in the Middle*

### 對齊與多樣性(§④)

- Ouyang et al. (2022) InstructGPT;Rafailov et al. (2023) DPO
- Kirk et al. (2024) — *Effects of RLHF on Generalisation and Diversity*

</div>

<div>

### DDO 與相關工作(§④)

- **Zheng et al. (ICML 2025)** — *Direct Discriminative Optimization*(arXiv:2503.01103)
- <span class="text-xs opacity-70">research.nvidia.com/labs/dir/ddo</span>
- Chen et al. (2024) *SPIN*;Xu et al. (2023) Iterative DPO
- <span class="text-xs opacity-70">與 DPO、CFG 兩篇合讀</span>

### 量測(§⑤)

- Kuhn, Gal & Farquhar (ICLR 2023) — *Semantic Uncertainty*
- Farquhar et al., Nature (2024)
- Kadavath et al. (2022) — *LMs (Mostly) Know What They Know*
- Lin, Hilton & Evans (2022) — *Teaching Models to Express Uncertainty in Words*
- Guo et al. (2017) — *On Calibration*(temperature scaling 原始出處)
- Wang et al. (2023) — *Self-Consistency*

</div>

</div>

<!--
不要念這頁。

只讀三篇的話:Autoguidance(②)、DDO(④)、Semantic Uncertainty(⑤)。

年份以 arXiv 首發為準,正式發表場次略有出入,引用時依情境標註。
-->

---
layout: center
class: text-center end
---

<div class="eyebrow">兩堂課 · 完</div>

# 三種介入時機

<div class="text-lg opacity-80 mt-3">訓練目標 · 解碼設定 · 權重微調</div>

<div class="mt-10 max-w-3xl mx-auto">
  <SpectrumAxis :rows="3" />
</div>

<div class="mt-8 text-sm opacity-70">
下次組會:請說出自己的題目落在第幾層
</div>

<!--
最後一頁停著。第一堂結束時這張圖只有第一列,現在三列都滿了。
-->
