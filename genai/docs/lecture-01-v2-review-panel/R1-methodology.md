# R1 · 方法論審查意見

審查對象:`/media/8tsp/projects/orientation-2026/genai/lecture-01-choose-a-ruler-v2.md`(1910 行)
與 `/media/8tsp/projects/orientation-2026/genai/components/*.vue`、`components/divergence-math.js`

審查範圍:數學正確性、數值與圖的一致性、前提是否說出口、絕對化措辭。
**不涉及**教學設計、時間配置、文獻歸屬。

環境:`node v22.18.0`,工作目錄 `/media/8tsp/projects/orientation-2026/genai`。
本審查未修改任何檔案;所有驗證腳本都寫在 `/tmp/`。

---

## 總體評價

這份簡報的數學骨架是可靠的。我逐條重推了 24 條數學宣稱,**19 條完全正確**(含 KL 的積分權重論證、JSD 的 log 2 上界與 √JSD 的度量性質、最佳判別器 $D^*$、$V(D^*,G)=2\,\mathrm{JSD}-2\log 2$、JSD $=I(X;Z)$、CE $=H(p)+\mathrm{KL}$、ELBO 的 Jensen 方向與等式形式、重參數化、BPB 的量綱、序列 KL 的鏈鎖分解)。符號 $p/q/p_{\text{data}}/p_\theta/p_g$ 在全篇是一致的,沒有前後翻面。這在一份 180 分鐘、橫跨四個家族的簡報裡不常見。

問題集中在三處:

1. **一個數值假象被當成教學重點**(JSD 飽和頁)。那張圖上的 KL 曲線在 $d\gtrsim 4$ 之後是平的,原因是 `divergence-math.js` 的 `EPS=1e-12` 把積分尾巴截掉了。該頁要教的正是「KL 還在漲、JSD 已經停住」,而圖與滑桿讀數都不支持這句話。這是唯一的 CRITICAL。
2. **兩個宣稱與同一份投影片上的數字/圖直接矛盾**:「forward KL 從不倒向任一側」(圖例自己寫著 $\mu=0.64$)、「判別器的介面裡沒有覆蓋度這個欄位」(六頁後的 demo 頁說判別器地景上那些 mode 分數偏高)。
3. **上界/下界講反一處**(BPB 頁的 canonical 切法),以及**一處下標只換了一半**(reverse KL 的鏈鎖分解)。

最重要的一個結構性問題不在任何單一行:**整堂課的因果鏈(散度決定失效模式)只在「模型族無法表示 $p_{\text{data}}$」時成立**,而簡報從頭到尾沒說這句話,反而在第 861–876 行用 proper scoring rule 論證「模型仍能收斂到條件分布」——那是可實現情形,兩者不能同時為真。詳見〈未說出口的前提〉P1。

統計:CRITICAL 1、MAJOR 8、MINOR 12。

---

## 已驗證正確的推導

| # | 項目 | 行號 | 結論 |
|---|---|---|---|
| 1 | $\mathrm{KL}(p\|q)=\int p\log(p/q)$,權重是 $p$ | 202, 206 | ✅ 正確 |
| 2 | forward KL:$p>0,q\to0\Rightarrow+\infty$,故 $q$ 必須覆蓋 $p$ 的 support | 216–222 | ✅ 正確(零測度集除外,見 MINOR-10) |
| 3 | Jeffreys 在 support 不重疊時為 $\infty$ | 300–304 | ✅ 正確 |
| 4 | $m=\tfrac{p+q}{2}\Rightarrow m\ge\tfrac12 p,\ m\ge\tfrac12 q$ | 314, 327 | ✅ 正確 |
| 5 | $0\le\mathrm{JSD}\le\log 2$ | 322 | ✅ 正確(自然對數;數值驗到 0.693147) |
| 6 | $\sqrt{\mathrm{JSD}}$ 滿足度量公理(含三角不等式) | 322 | ✅ 正確 |
| 7 | $D^*(x)=\dfrac{p(x)}{p(x)+q(x)}$ | 361 | ✅ 正確;由 $a\log y+b\log(1-y)$ 求極值,$y^*=a/(a+b)$ |
| 8 | $V(D^*,G)=2\,\mathrm{JSD}(p\|q)-2\log 2$ | 365, 1211 | ✅ 正確($-\log4=-2\log2$) |
| 9 | $\mathrm{JSD}=I(X;Z),\ Z\sim\mathrm{Bern}(1/2)$ | 373 | ✅ 正確;$I=H(m)-\tfrac12 H(p)-\tfrac12 H(q)$ |
| 10 | GAN value function 的寫法 | 1191 | ✅ 正確 |
| 11 | non-saturating:$G$ 改為最大化 $\log D(G(z))$ | 1202 | ✅ 正確 |
| 12 | NS loss 的第一項 $\mathrm{KL}(p_g\|p_{\text{data}})$ 是 reverse KL | 1222 | ✅ 與第 202/228 行的 $p/q$ 約定一致 |
| 13 | $H(p,q)=H(p)+\mathrm{KL}(p\|q)$ | 688 | ✅ 正確 |
| 14 | one-hot 目標 $\Rightarrow H(p)=0\Rightarrow$ CE $\equiv$ forward KL | 693 | ✅ 正確 |
| 15 | $H(p)=-\mathbb{E}_{x\sim p}[\log p]$ 無法用樣本做 MC | 707–711 | ✅ 正確(MC 需要 $\log p$ 的數值) |
| 16 | BPB $=\frac{T}{N_{\text{bytes}}}\log_2\mathrm{PPL}_{\text{token}}$ | 751 | ✅ 量綱正確:bits/token × tokens/byte |
| 17 | $\mathrm{KL}(p\|q)=\sum_t\mathbb{E}_{x_{<t}\sim p}[\mathrm{KL}(p(\cdot\|x_{<t})\|q(\cdot\|x_{<t}))]$ | 783 | ✅ 離散反例驗證,誤差 0(見〈驗證 V5〉) |
| 18 | ELBO 的 Jensen 方向($\log$ 凹 $\Rightarrow \mathbb{E}\log\le\log\mathbb{E}$) | 1026–1027, 1043 | ✅ 正確,是下界 |
| 19 | ELBO $=\mathbb{E}_q[\log p_\theta(x\|z)]-\mathrm{KL}(q_\phi\|p(z))$ | 1028 | ✅ 正確 |
| 20 | $\log p_\theta(x)=\mathrm{ELBO}+\mathrm{KL}(q_\phi(z\|x)\|p_\theta(z\|x))$ | 1036 | ✅ 正確;間隙 $\ge0$,故 ELBO 恆為下界 |
| 21 | $z=\mu_\phi(x)+\sigma_\phi(x)\odot\varepsilon$ | 1076 | ✅ 正確;高斯對高斯的 KL 有閉式解亦正確 |
| 22 | $\mathcal{L}_\beta$ 的形式與 $\beta$ 三段行為 | 1093–1101 | ✅ 正確 |
| 23 | $W_1(p,q)=\sup_{\|f\|_L\le1}\big(\mathbb{E}_p f-\mathbb{E}_q f\big)$ | GanFixes.vue:66 | ✅ Kantorovich–Rubinstein 對偶,正確 |
| 24 | WGAN-GP 的 $(\|\nabla_{\hat x}f\|-1)^2$、R1 的 $\|\nabla_x D\|^2$(只在真資料上) | 1352, 1354 | ✅ 公式正確 |
| 25 | B-1 的三組擬合參數(w=0.5 與 w=0.3) | 262, 287, 409–410 | ✅ 與 `divergence-math.check.mjs` 完全相符,且對 `EPS` 不敏感 |
| 26 | 「forward 解在谷底配置可觀質量」 | 257 | ✅ 實算:$p(0)=0.0105$,$q_{\text{fwd}}(0)=0.2347$,約 22 倍 |
| 27 | RLHF 用的是 $\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$(mode-seeking 方向) | 1575–1577 | ✅ 方向正確 |
| 28 | Trilemma 三個模型坐的邊與放棄的頂點 | Trilemma.vue:52–58 | ✅ 與 Xiao et al. 一致 |

---

## 問題清單

### [CRITICAL] JSD 飽和頁的 KL 曲線是數值假象,該頁的教學對比在圖上不成立

**位置**:`lecture-01-choose-a-ruler-v2.md` 第 336、339、348 行;
`components/divergence-math.js` 第 5、19 行;`components/JsdSaturate.vue` 第 8、28、106 行。

**問題**:
`divergence-math.js` 的 `kl()` 寫成

```js
s += a[i] * Math.log((a[i] + EPS) / (b[i] + EPS))   // EPS = 1e-12
```

這個 `EPS` 是為了避免 $\log 0$ 而加的,但它同時把「$a$ 尚有質量、$b$ 已經極小」的那段尾巴的貢獻鎖死在 $a\log(a/\text{EPS})$。兩個等變異數高斯($\sigma=0.5$)的 KL 有解析解 $d^2/2\sigma^2=2d^2$,實測結果:

| $d$ | 圖上/滑桿顯示的 KL | 解析 KL | 比值 |
|---|---|---|---|
| 2 | 8.00 | 8 | 1.00 |
| 3 | 17.68 | 18 | 0.98 |
| 4 | 25.66 | 32 | 0.80 |
| 5 | 26.89 | 50 | 0.54 |
| 6 | 26.91 | 72 | 0.37 |

也就是說,**在滑桿的後半段($d\in[4.5,6]$),KL 曲線也是平的**(26.89 → 26.91,變動 0.08%),而且 $y$ 軸上限是 30、KL 停在 26.9,並沒有被削頂——學生看到的就是兩條都躺平的線。這頁整頁的論證是「JSD 停住不動、KL 還在漲」,而現場拖滑桿會得到相反的印象。第 348 行講者備註「KL 一路飆到 27」把這個假象寫成了教學台詞;真值是 72。

我確認過**這只影響 `separationCurve`,不影響 B-1 的擬合結果**:把 `EPS` 改成 `1e-300` 重跑 `fitGaussian`,六組 $(\mu,\sigma)$ 一字不差。所以這是個孤立的、可以安全修掉的 bug。

**驗證**:

```bash
cd /media/8tsp/projects/orientation-2026/genai
cat > /tmp/r1kl.mjs <<'EOF'
const SQRT2PI=Math.sqrt(2*Math.PI)
const gauss=(x,m,s)=>Math.exp(-0.5*((x-m)/s)**2)/(s*SQRT2PI)
function klNum(d,sig,{xmax=6,dx=0.025,eps=1e-12}={}){
  const n=Math.round(2*xmax/dx)+1; let s=0
  for(let i=0;i<n;i++){const x=-xmax+i*dx
    const a=gauss(x,-d/2,sig),b=gauss(x,d/2,sig)
    s+=a*Math.log((a+eps)/(b+eps))}
  return s*dx }
for(const d of [2,3,4,5,6])
  console.log(d, klNum(d,0.5).toFixed(2), klNum(d,0.5,{eps:0}).toFixed(2), (2*d*d).toFixed(2))
EOF
node /tmp/r1kl.mjs
```

輸出第二欄是目前設定,第三欄是 `eps=0`,第四欄是解析值——第三與第四欄完全相同,證明唯一肇因就是 `EPS`。

**建議**:

1. `divergence-math.js:5` 改 `const EPS = 1e-300`(這個網格上密度最小值約 $10^{-179}$,雙精度下不會下溢,不需要放寬積分區間)。
2. `JsdSaturate.vue:32` 的 `VMAX` 從 30 改成 100,讓 $d=6$ 的 KL=72 落在刻度內。
3. `JsdSaturate.vue:28` 的註解改成:
   > 為什麼要對數:KL 從 2e-2 一路到 72(= $d^2/2\sigma^2$,無上界),JSD 從 5e-3 到 0.693 就停住。
4. 第 348 行講者備註改成:
   > 現場拖那個滑桿。從 d=0 拖到 d=6:KL 從 0 一路漲到 72(而且是 $d^2$ 的速度,永遠不會停),JSD 停在 0.693 動也不動。

修完之後這頁才真的是「一張圖說完一件事」;修之前它說的是反話。

---

### [MAJOR] 「forward KL 從不倒向任一側」與同一頁圖例上的 μ=0.64 直接矛盾

**位置**:第 399 行(另見 `components/divergence-math.js` 第 54 行的同義註解)。

**問題**:第 395 行畫的是 $w=0.3$ 的三條擬合曲線,`DivergenceFit.vue` 的圖例會逐條印出 $\mu,\sigma$。實測 forward KL 在 $w=0.3$ 的解是 $\mu=0.64,\ \sigma=1.58$——**它確實倒向了大峰**。掃描更多權重:

| $w$(左峰權重) | forward | JSD | reverse |
|---|---|---|---|
| 0.5 | (0.00, 1.70) | (0.00, 1.66) | (1.60, 0.58) |
| 0.4 | (0.32, 1.66) | (0.32, 1.66) | (1.60, 0.58) |
| 0.3 | (0.64, 1.58) | (1.60, 0.58) | (1.60, 0.58) |
| 0.2 | (0.96, 1.38) | (1.60, 0.58) | (1.60, 0.54) |
| 0.1 | (1.28, 1.10) | (1.60, 0.54) | (1.60, 0.54) |

forward KL 的 $\mu$ 隨 $w$ **連續、單調地**朝大峰移動。它與 JSD/reverse 的真正差別不是「倒不倒」,而是「怎麼倒」:forward 的 $\sigma$ 始終維持在 1.1–1.7(峰位在 $\pm1.6$,所以它一直罩得住兩個峰),是**連續平移**;JSD 在 $w$ 跨過某個門檻時**不連續地跳**到單峰解。學生只要看一眼圖例就會發現這句話跟數字對不上,而這一頁的賣點正是「這是算出來的,不是示意圖」。

**驗證**:

```bash
cd /media/8tsp/projects/orientation-2026/genai
node -e "
import('./components/divergence-math.js').then(({bimodal,fitGaussian})=>{
  for(const w of [0.5,0.4,0.3,0.2,0.1])
    console.log(w, JSON.stringify(['forward','jsd','reverse'].map(k=>fitGaussian(k,bimodal(w)))))
})"
```

**建議**:第 398–399 行整段改為

> 兩峰的權重一旦拉開,**JSD 的解會不連續地跳到大峰上**:它放棄小峰,跟 reverse KL 收斂到同一個地方。
> forward KL 也會朝大峰平移($w=0.3$ 時 $\mu=0.64$),但 $\sigma$ 仍維持在 1.58,兩個峰都還罩得住;
> reverse KL 從頭到尾就只坐在單一峰上。**差別在「平移」與「跳邊」,不在「倒不倒」。**

並把第 404 行的收束句改成:

> 「在中間」不等於「取平均」,而是**「權重一旦夠不對稱,就整個跳到其中一側」**。

`divergence-math.js:54` 的註解同樣要改:「forward KL 永遠不翻」→「forward KL 只平移不翻邊」。

---

### [MAJOR] 「兩峰一旦不對稱,JSD 就倒向大峰」有反例;真正的門檻同時取決於峰距

**位置**:第 398 行、第 409–412 行講者備註。

**問題**:$w=0.4$ 已經不對稱(60/40),但 JSD 的解是 $(0.32,1.66)$——仍然是 covering,而且與 forward KL 的解完全相同。翻邊發生在 $w$ 介於 0.35 與 0.3 之間。更要緊的是,這個門檻**不是 JSD 的性質,而是這組 $p$ 的性質**:

| 峰位 | $w=0.5$ | $w=0.45$ | $w=0.4$ | $w=0.35$ | $w=0.3$ |
|---|---|---|---|---|---|
| $\pm1.0$ | (0, 1.14) | (0.08, 1.10) | (0.20, 1.10) | (0.28, 1.10) | (0.40, 1.06) ← 到 0.3 都沒翻 |
| $\pm1.6$(投影片用的) | (0, 1.66) | (0.16, 1.66) | (0.32, 1.66) | (0.48, 1.62) | **(1.60, 0.58)** |
| $\pm2.2$ | (0, 2.38) | **(2.20, 0.54)** | (2.20, 0.54) | (2.20, 0.54) | (2.20, 0.54) |

峰距 $\pm1.0$ 時 JSD 到 $w=0.3$ 都還在 covering;峰距 $\pm2.2$ 時 $w=0.45$ 就翻了。簡報選的 $\pm1.6$ 恰好讓門檻落在 0.3 與 0.35 之間。這不影響「JSD 會翻邊」這個定性結論,但「一旦不對稱」這個量詞不成立,而且學生若照第 464 行的課後練習自己換一組參數,很可能重現不出投影片的現象。

**驗證**:見 `/tmp/r1sep.mjs`(下方〈驗證指令彙整〉V4)。

**建議**:第 398 行第一句改為

> 兩峰的權重**拉開到一定程度之後**(這組資料是左峰降到約 30% 時),<b>JSD 的解會整個跳到大峰</b>

第 409–412 行的講者備註補一句:

> 補充:翻邊的門檻同時取決於峰距。峰位 ±1.0 時 $w=0.3$ 都還沒翻,峰位 ±2.2 時 $w=0.45$ 就翻了。要講的是「存在一個翻邊點」,不是「不對稱就翻」。若有人課後自己換參數重跑,這點要先講。

---

### [MAJOR] BPB 頁:「上界」的方向講反了

**位置**:第 762 行。

**問題**:原文是

> **上界** | 算的是 canonical 切法,是真實字串機率的**上界**;不同 tokenizer 鬆緊不同

字串 $s$ 的真實機率是**所有能解碼成 $s$ 的切法之和**:

$$p(s)=\sum_{t:\ \mathrm{decode}(t)=s} p(t)\ \ge\ p(t_{\text{canonical}})$$

所以 canonical 切法的機率是真實字串機率的 **下界**,不是上界。取負對數之後方向再翻一次:$-\log p(t_{\text{canonical}}) \ge -\log p(s)$,因此**算出來的 bits、PPL、BPB 都是真值的上界**。表格的欄名(「上界」)其實是對的,錯的是後面那句解釋——它把「BPB 的上界」寫成了「字串機率的上界」。

這是實務上會咬人的錯:學生若照字面理解,會以為自己的 BPB 是樂觀值,於是往反方向做調整。

**驗證**:純推導,不需執行。一行反例:tokenizer 有兩種切法都能解碼成 `"ab"`,各給 0.3 與 0.2,則 $p(\text{"ab"})=0.5 > 0.3 = p(\text{canonical})$。

**建議**:該列改為

> **上界** | 算的只是 canonical 切法的機率,它 $\le$ 真實字串機率(真值要對所有切法求和),因此**算出來的 BPB 是真值的上界**;不同 tokenizer 的鬆緊不同,上界的鬆緊也不同

---

### [MAJOR] BPB 頁:「字串的 log-likelihood 與 token 切法無關」不成立,且與下一段自相矛盾

**位置**:第 747 行。

**問題**:原文是

> 鏈鎖法則會 **telescope**:字串的 log-likelihood 與 token 切法無關,會變的只有分母。

telescope 成立的範圍是**固定切法之內**:$\sum_t\log p(x_t\mid x_{<t})=\log p(t_{\text{canonical}})$,得到的是**那一條 token 序列**的機率,不是字串的機率。換一個 tokenizer,得到的是另一條序列的機率——既不是同一個數,也不是同一個模型。這正是上一條(第 762 行)所說「不同 tokenizer 鬆緊不同」的原因;把兩句話擺在同一頁上,前一句說「與切法無關」、後一句說「不同 tokenizer 鬆緊不同」,學生兩句都會記不住。

BPB 之所以有用,不是因為分子與切法無關,而是因為**分母換成 bytes 之後,兩個上界至少落在同一個尺度上、可以比較**。

**建議**:第 747 行改為

> 鏈鎖法則在**固定的切法之內** telescope:$\sum_t\log p(x_t\mid x_{<t})$ 剛好等於那條 token 序列的 log-likelihood。
> 換一個 tokenizer 就換了一條序列,分子本來就會變——BPB 的作用是把分母統一成 bytes,讓兩個(各自鬆緊不同的)上界至少可以並排看。

---

### [MAJOR] 「把下標從 p 換成 q,就得到 reverse KL 的分解」——下標只換了一半

**位置**:第 843 行。

**問題**:forward 的分解是

$$\mathrm{KL}(p\|q)=\sum_t\mathbb{E}_{x_{<t}\sim p}\big[\mathrm{KL}\big(p(\cdot\mid x_{<t})\,\big\|\,q(\cdot\mid x_{<t})\big)\big]$$

reverse KL 的分解是

$$\mathrm{KL}(q\|p)=\sum_t\mathbb{E}_{x_{<t}\sim \mathbf{q}}\big[\mathrm{KL}\big(\mathbf{q}(\cdot\mid x_{<t})\,\big\|\,\mathbf{p}(\cdot\mid x_{<t})\big)\big]$$

**外層期望與內層 KL 的方向必須一起換**。只換外層下標得到的是第三個泛函

$$\sum_t\mathbb{E}_{x_{<t}\sim q}\big[\mathrm{KL}\big(p(\cdot\mid x_{<t})\,\big\|\,q(\cdot\mid x_{<t})\big)\big]$$

它既不是 forward KL 也不是 reverse KL。這不是吹毛求疵:上面那個「只換一半」的泛函,恰好就是 **scheduled sampling** 在做的事(前綴取自模型、目標仍是資料的條件分布),而第 845 行把 scheduled sampling 與 RL / DPO / DDO 列在同一類。這頁把兩個不同的東西合併了,而下堂課要在這個基礎上蓋 DDO。

**驗證**(隨機離散兩步序列,三個泛函逐一算):

```bash
cat > /tmp/r1chain.mjs <<'EOF'
const rnd=()=>Math.random()+0.05, norm=a=>{const s=a.reduce((x,y)=>x+y,0);return a.map(v=>v/s)}
const mk=()=>norm([rnd(),rnd(),rnd(),rnd()])
const KL=(a,b)=>a.reduce((s,v,i)=>s+(v>0?v*Math.log(v/b[i]):0),0)
const m1=j=>[j[0]+j[1],j[2]+j[3]], c2=j=>[norm([j[0],j[1]]),norm([j[2],j[3]])]
for(let t=0;t<3;t++){const p=mk(),q=mk(),pm=m1(p),qm=m1(q),pc=c2(p),qc=c2(q)
 console.log('KL(p||q)=',KL(p,q).toFixed(6),
  ' 鏈鎖(both p)=',(KL(pm,qm)+pm.reduce((s,w,i)=>s+w*KL(pc[i],qc[i]),0)).toFixed(6),
  '| KL(q||p)=',KL(q,p).toFixed(6),
  ' 鏈鎖(both q)=',(KL(qm,pm)+qm.reduce((s,w,i)=>s+w*KL(qc[i],pc[i]),0)).toFixed(6),
  '| 只換下標=',(KL(pm,qm)+qm.reduce((s,w,i)=>s+w*KL(pc[i],qc[i]),0)).toFixed(6))}
EOF
node /tmp/r1chain.mjs
```

輸出:兩個鏈鎖式各自與對應的 KL 完全相等(驗證了第 783 行的式子正確),而「只換下標」是第三個數。

**建議**:第 843–846 行改為

> ### 3 · 把**外層下標與內層 KL 的方向一起**換成 $q$,才得到 reverse KL 的分解
>
> $\mathrm{KL}(q\|p)=\sum_t\mathbb{E}_{x_{<t}\sim q}\big[\mathrm{KL}(q(\cdot\mid x_{<t})\,\|\,p(\cdot\mid x_{<t}))\big]$
>
> 只換外層下標、內層仍以資料的條件分布為目標,得到的是第三個泛函——那就是 scheduled sampling。
> RL / DPO / DDO 才是兩者都換。三者的共同點**不是「用了 RL」**,而是 **把 loss 搬到了下方那條軌道上**。

---

### [MAJOR] mode collapse 第一層原因的論證有誤,且與六頁後的 demo 頁自相矛盾

**位置**:第 1252–1255 行;`components/AdversarialLoop.vue` 第 55–57 行;矛盾對象是第 1307–1308 行。

**問題**:第 1254–1255 行寫

> $D(x)$ 是**逐點**判斷:「覆蓋度」是分布層級的性質,**判別器的介面裡沒有這個欄位。**

但在最佳判別器下 $D^*(x)=\dfrac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}$——這個逐點的數值**就是**覆蓋度資訊:$p_g$ 在哪裡不足,$D^*$ 在那裡就高。第 1307–1308 行自己說得很清楚:

> 未被覆蓋的那幾個 mode,在判別器地景上分數**偏高**,但生成分布不會移向該處。

兩頁不能同時成立。**真正的機制不在判別器的介面,而在生成器 loss 的期望值取在哪裡**:$\mathbb{E}_{z\sim p(z)}[\cdot]$ 只在 $G$ **目前產出的樣本上**取值,所以梯度只在 $G$ 已經去過的地方存在。判別器把「那裡有洞」寫在地景上了,但生成器從來不在那裡取樣,那份資訊沒有任何一條路徑可以送達。

這個更正反而讓第 1290–1308 行的 demo 更有力:學生看到的正是「訊息在那裡,但收不到」。

**驗證**:代入 $D^*$ 即可;無需執行。可用第 1314 行的 GAN 2D demo 現場印證:未覆蓋區 $D$ 分數高,而 $G$ 的樣本從不落在那裡。

**建議**:第 1252–1256 行整塊改為

> ### 1 · 生成器只在自己已經去過的地方收得到梯度
>
> $\mathbb{E}_{p_{\text{data}}}[\log D(x)]$ 這一項**不含 $G$**;含 $G$ 的那一項是 $\mathbb{E}_{z\sim p(z)}[\cdot]$,
> 期望值只取在**生成器目前產出的樣本上**。
> 最佳判別器 $D^*=\frac{p_{\text{data}}}{p_{\text{data}}+p_g}$ 其實已經把「這裡有洞」標在地景上了——
> 但生成器從不在那裡取樣,**那份資訊沒有路徑可以送回來**。

`AdversarialLoop.vue:56` 那句「這條迴路上沒有任何地方能傳遞它」同步改成:
「這條迴路只在 $G$ 已經產出的樣本上傳梯度;$G$ 沒去過的地方,地景再高也傳不回來。」

第 1272 行下堂課的伏筆也要跟著修(見 MINOR-20)。

---

### [MAJOR] 「換掉 non-saturating loss 同時也換掉了 mode-covering」——原本那個目標本來就不是 mode-covering

**位置**:第 1234 行。

**問題**:原文

> 為了修梯度消失而換掉的那個 loss,<b>同時也換掉了 mode-covering。</b>

被換掉的那個 loss(minimax / saturating)在最佳判別器下等於 $2\,\mathrm{JSD}-2\log2$,而這份簡報第 398 行剛剛用自己算的圖證明過:**JSD 在權重不對稱時會跳到大峰,跟 reverse KL 收斂到同一個解**。所以原始 GAN 從來就沒有 forward-KL 那種 mode-covering,談不上「換掉」。

正確的說法是程度問題:JSD 在對稱情形下仍是 covering 解($w=0.5$ 時 $\mu=0,\sigma=1.66$),換成含 $+\mathrm{KL}(p_g\|p_{\text{data}})$ 的 NS loss 之後,連這點殘存的 covering 傾向也一起沒了。

**驗證**:見上面第二條 MAJOR 的表格($w=0.5$ 時 JSD 解為 covering,$w=0.3$ 時為 seeking)。

**建議**:第 1234–1235 行改為

> 為了修梯度消失而換掉的那個 loss,**把僅存的一點 covering 傾向也一起換掉了**。<br>
> JSD 在資料對稱時還會給出覆蓋解(② 那張 $w=0.5$ 的圖),NS loss 的第一項直接是 reverse KL,連這點都不剩。<br>
> GAN 的兩種 loss,分別坐在兩種失效模式上。

---

### [MAJOR] ELBO 間隙「不取決於 decoder 的規模」不成立

**位置**:第 1056 行。

**問題**:間隙是 $\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)$,而真後驗 $p_\theta(z\mid x)\propto p_\theta(x\mid z)p(z)$ **完全由 decoder 決定**。同一個對角高斯 $q$ 家族,配上不同的 decoder,間隙可以差很多:decoder 越強、真後驗越複雜,對角高斯就越追不上(這也正是 posterior collapse 的另一面——當 decoder 強到可以忽略 $z$,間隙反而以退化的方式縮小)。

間隙取決於**「$q$ 家族」與「真後驗」的落差**,而真後驗兩端都動。說成「不取決於 decoder」會讓學生得到一個錯誤的實務結論:「我的下界很鬆,那跟我把 decoder 加大無關」。

**建議**:第 1056 行改為

> 我們優化的是 ELBO,不是 $\log p_\theta(x)$。而<b>間隙 = $q_\phi$ 與**真後驗**的 KL</b>:
> 它同時取決於 $q$ 家族夠不夠靈活,以及 decoder 把真後驗弄得多複雜——這兩端都動,所以加大 decoder 不保證下界會變緊。

`ElboGap.vue` 第 28 行「$\log p_\theta(x)$ · 與 q 無關,固定在這條線上」是**對的**($\theta$ 固定時),但那張圖的三根柱子共用一條天花板,容易被讀成「$\theta$ 怎麼變都固定」。建議把該行文字改成「$\log p_\theta(x)$ ·($\theta$ 固定時)與 q 的選擇無關」。

---

### [MAJOR] non-saturating 的散度分解同樣需要「D 已達最優」,但投影片的對比句暗示它不需要

**位置**:第 1211–1217 行。

**問題**:第 1211 行先說

> $V=2\,\mathrm{JSD}-2\log 2$ **只在最佳判別器下成立**,而那正是梯度消失的地方

接著第 1213–1217 行給出

> 改用 non-saturating loss 之後,生成器實際在最小化的是 $\mathrm{KL}(p_g\|p_{\text{data}})-2\,\mathrm{JSD}(p_g\|p_{\text{data}})$

兩個問題:

1. **前提沒有變**。Arjovsky & Bottou 的那條結果同樣假設 $D=D^*$——它說的是「NS loss 的梯度」等於「該泛函的梯度」,前提就是最佳判別器。整段的行文讓人以為前一頁的前提被繞開了,其實只是換了個泛函。
2. **那是梯度的等式,不是目標函數的等式**。原結果是
   $$\nabla_\theta\,\mathbb{E}_{z}[-\log D^*(g_\theta(z))]\;=\;\nabla_\theta\big[\mathrm{KL}(p_{g_\theta}\|p_{\text{data}})-2\,\mathrm{JSD}(p_{g_\theta}\|p_{\text{data}})\big]$$
   兩邊差一個常數,「生成器實際在最小化的是」這個講法會讓學生以為可以把它當 loss 監控——實際上算不出來(兩邊的密度都拿不到,這是本課自己的論點)。

**建議**:第 1213–1217 行改為

> 改用 non-saturating loss 之後,**在同樣的「$D$ 已達最優」前提下**,生成器的梯度等於下面這個泛函的梯度:
>
> $$\nabla_\theta\Big[\mathrm{KL}(p_g\|p_{\text{data}})-2\,\mathrm{JSD}(p_g\|p_{\text{data}})\Big]$$
>
> (是梯度相等,不是 loss 相等;這個量本身算不出來,只能拿來讀方向。)

並在第 1211 行後補一句,把兩個前提分開:

> 注意這裡其實是兩個前提疊在一起:$D$ 訓到最優,**而且**兩個分布的 support 幾乎不重疊。梯度消失是後者造成的,不是前者。

---

### [MINOR] zero-forcing 這個標籤與所給的解釋不對應

**位置**:第 230–234 行。

**問題**:框內的解釋是「權重換成 $q$,$q=0$ 的地方整項歸零 → $p$ 的其他模式被忽略也不受懲罰」——這解釋的是 **mode-seeking**。而 **zero-forcing** 指的是另一個方向的機制:$p=0$(或極小)的地方,$\log(q/p)\to+\infty$,所以 $q$ **被迫**在那裡歸零。兩個機制互補,但投影片只給了一個解釋卻掛了兩個標籤。

**建議**:框內第二句後補一行:

> 反過來,$p\approx0$ 的地方 $\log\frac{q}{p}\to+\infty$,所以 $q$ **被迫**在那裡歸零 —— 這才是 zero-forcing 這個名字的來源。

---

### [MINOR] $p(y\mid x)$ 應寫成模型的 $q_\theta(y\mid x)$

**位置**:第 121 行、第 897 行、第 900 行。

**問題**:第 121 行寫「$p(y\mid x)$ **永遠** well-defined,即使 $p(x)\approx0$」。若指的是**真分布**的條件,那麼在 $p(x)=0$ 處條件分布根本沒有定義(只在零測度集意義下任意)。這一頁真正要說的是:**模型**無論輸入什麼都會算出一個 softmax。這正是本課最在意的 $p$ / $q$ 分辨。

**建議**:第 121 行改成

> | 虛假前提檢測 | $q_\theta(y\mid x)$ 對任何 $x$ 都算得出來,哪怕 $p_{\text{data}}(x)\approx0$ | 左端的結構性後果 |

第 900 行同步改為:「$p_{\text{data}}(x)\approx0$,但**模型的** $q_\theta(y\mid x)$ 照樣算得出來,而且模型從沒被教過在這裡要停。」

---

### [MINOR] ComputeMap 的 reverse KL 那一列少了一項需求

**位置**:`components/ComputeMap.vue` 第 15–22 行(投影片第 542 行)。

**問題**:$\mathrm{KL}(p_\theta\|p_{\text{data}})=\mathbb{E}_{p_\theta}[\log p_\theta]-\mathbb{E}_{p_\theta}[\log p_{\text{data}}]$ 需要**三件事**:從 $p_\theta$ 取樣、$p_\theta$ 的密度(熵那一項)、$p_{\text{data}}$ 的密度。表上只列了前一與後一。缺的那一項不是細節——它正是「VI 為什麼要求 $q$ 的密度可算」與「RLHF 為什麼要算 $\log\pi_\theta$」的原因,而這一頁的賣點就是「把散度往四格上一放,家族自己掉出來」。

**建議**:`ROWS[1].needs` 補一個 chip:

```js
needs: [
  { t: '從 p_θ 取樣', ok: true },
  { t: '算 p_θ 密度', ok: true },
  { t: '算 p_data 密度', ok: false },
],
```

講者備註可加一句:「reverse KL 要三樣東西,不是兩樣。$p_\theta$ 的密度就是熵項,這是 VI 與 RLHF 都要算 $\log\pi_\theta$ 的原因。」

---

### [MINOR] DecompAxes 的鏈鎖式掉了期望值下標

**位置**:`components/DecompAxes.vue` 第 29 行(投影片第 1410 行)。

**問題**:圖上寫 `Σₜ KL(p(·|x<ₜ) ‖ q(·|x<ₜ))`,少了 $\mathbb{E}_{x_{<t}\sim p}$。第 781–799 行整整一頁的教學重點就是「請盯著期望值的下標」,結果 30 頁後同一個式子把下標拿掉了。

**建議**:改成 `Σₜ E_{x<ₜ~p}[ KL(p(·|x<ₜ) ‖ q(·|x<ₜ)) ]`。若寬度不夠,拆成兩行。

---

### [MINOR] 「兩者的訓練目標都退化成簡單回歸」——AR 是分類不是回歸

**位置**:第 1414 行。

**問題**:AR 每一步是 token 上的 cross-entropy(分類),擴散每一步才是回歸。`DecompAxes.vue` 第 44–45 行寫得是對的(只說擴散退化成回歸),投影片正文擴大成了「兩者」。

**建議**:第 1413–1414 行改為

> 還是 forward KL,還是 MLE 路線。差別只在<b>沿什麼軸做鏈鎖分解</b>:AR 沿序列,擴散沿噪聲尺度。<br>
> 兩者的每一步都因此退化成一個**簡單的監督式問題**(AR 是分類、擴散是回歸),<b>所以它們一樣穩定。</b>

---

### [MINOR] 「CE 的絕對數值沒有意義」過強

**位置**:第 713 行。

**問題**:CE 的絕對值有明確意義——它就是用這個模型編碼這批資料的期望碼長(換底之後就是 bits)。**不可辨識的是「離真分布多遠」那一部分**,因為 $H(p)$ 未知。原句會讓學生連「CE = 壓縮率」這個正確且有用的解讀也一併丟掉,而下一頁 BPB 恰恰就是靠這個解讀。

**建議**:改為

> 所以 CE 的絕對值只能讀成「編碼長度」,**不能讀成「離真分布多遠」**——後者差一個未知的 $H(p)$,只有差值才消得掉它。

---

### [MINOR] 「同一個前綴在整份語料裡會出現很多次」對 LLM 的長前綴不成立

**位置**:第 873–874 行(講者備註)。

**問題**:這個解釋對短 n-gram 前綴成立,對 LLM 幾乎必然不成立——長度上百 token 的前綴在整份語料裡通常只出現一次。模型之所以仍能學到條件分布,靠的是**參數共享帶來的泛化**(相似前綴共用表示),不是同一前綴的重複平均。式子上,$\mathbb{E}_{x_{<t}\sim p}$ 的期望是對**前綴分布**取的,單一樣本就是無偏估計,不需要「同一個前綴重複出現」。

**建議**:改為

> 「只有一個續寫卻能學到分布」學生常卡住。正確的說法是:期望值是對**前綴分布**取的,每個前綴出現一次就已經是無偏樣本;
> 模型能在單次出現的長前綴上學到分布,靠的是參數共享讓相似前綴互相借力,不是同一個前綴被重複平均。

---

### [MINOR] 「四個工作,同一個動作:限制判別器的 Lipschitz 常數」——R1 不是

**位置**:第 1359 行。

**問題**:WGAN(weight clipping)、WGAN-GP、Spectral Norm 都在(近似地)施加**全域** Lipschitz 約束。R1 只在**真資料點上**罰 $\|\nabla_x D\|^2$,那是局部的梯度懲罰,Mescheder et al. 的動機是 GAN 動力學的**局部收斂性**,不是 Lipschitz 控制。第 1366 行的講者備註其實說對了(「局部收斂的證明」),正文的收束句卻把四個併成一句。

**建議**:第 1359 行改為

> 四個工作,同一個方向:**壓制判別器梯度的大小**。WGAN / GP / SN 是(近似)全域的 Lipschitz 約束,R1 是真資料附近的局部梯度懲罰——手段與硬度都不同,但都在把地景弄平緩。

---

### [MINOR] w=0.5 的 reverse KL 有兩個等價解,目前選到右峰是浮點噪音決定的

**位置**:第 274 行;`components/divergence-math.js` 第 61 行;`components/DivergenceFit.vue` 第 64–69 行。

**問題**:$w=0.5$ 時 $p$ 對稱,所以 $(\mu,\sigma)=(-1.6,0.58)$ 與 $(+1.6,0.58)$ 的 reverse KL 完全相同。實測網格上的兩個值是

```
mu = -1.59999999999999920   reverseKL = 0.687821164223678339
mu =  1.60000000000000253   reverseKL = 0.687821164223678228
```

差 $1.1\times10^{-16}$——最後一個 bit。`FITS` 記下的 $+1.6$ 是浮點累加誤差選出來的;換個網格步長或求和順序就會翻到左峰。而第 274 行說「reverse KL 的解**直接放棄左峰**」,`DivergenceFit.vue:67` 的標註「整個峰未被覆蓋,不受懲罰」也固定畫在左峰上。學生若照第 464 行的課後練習自己重跑,有相當機會得到鏡像的圖,然後以為自己算錯了。

**建議**:第 274 行改為

> reverse KL 的解直接放棄其中一個峰,收縮到另一個峰內部:<b>樣本銳利度高,但只覆蓋一半的 support。</b><br>
> (兩峰等重時左右兩個解一樣好,圖上是哪一邊由數值求解的細節決定——重點是「只挑一個」,不是「挑右邊」。)

第 287 行講者備註補一句:「$w=0.5$ 的 reverse 解是二重簡併,$\mu=\pm1.6$ 等價;有人自己重跑得到鏡像圖是正常的。」

---

### [MINOR] 「直接用於排序 → 永遠選出最短的候選」

**位置**:第 939 行。

**問題**:是強烈偏誤,不是必然。一個長而高機率的候選仍可能勝過短而低機率的候選(例如 5 個 token 每個 0.9 vs 1 個 token 0.3:$5\log0.9=-0.53 > \log0.3=-1.20$)。

**建議**:改為「直接用於排序 → **系統性偏好較短的候選**。」

---

### [MINOR] 「reward model 永遠不會指出回應分布過窄」——RLHF 目標裡的 β·KL 正是分布層級的項

**位置**:第 1272 行。

**問題**:reward model 本身是逐點評分器,這點成立。但把它推成「RLHF 對分布過窄無感」就不成立了:標準 RLHF 目標裡的 $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$ **就是**一個分布層級的項,它會直接懲罰塌縮(只不過是相對於 $\pi_{\text{ref}}$,不是相對於 $p_{\text{data}}$)。這份簡報第 1554 行與第 1113 行自己都強調過那個 $\beta$。

**建議**:第 1272 行改為

> 下堂課會看到:RLHF 的 reward model 有<b>一模一樣</b>的盲點——它也是逐點評分器,不會告訴你「回應分布過窄」。
> 唯一擋著這件事的是目標裡那個 $\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})$,而它拉住的是「別離參考模型太遠」,不是「別把分布收窄」。LLM-as-judge 連這個都沒有。

---

### [MINOR] 「間隙本身無法直接量測」

**位置**:`components/ElboGap.vue` 第 50 行。

**問題**:間隙算不出**精確值**,但可以夾:IWAE 給越來越緊的下界,AIS / BDMC 可以做上下夾擠。「無法直接量測」會讓學生以為這條路是封死的,而第 1148 行的 IWAE 那一列恰恰就是在做這件事。

**建議**:改為「間隙 = KL(q_φ(z|x) ‖ p_θ(z|x)) ≥ 0,所以 ELBO 恆為下界。間隙算不出精確值,只能用 IWAE / AIS 這類方法夾出來。」

---

### [MINOR] FamilyMatrix:Normalizing Flow 的「取樣速度」應與 Diffusion 同樣畫成區間

**位置**:`components/FamilyMatrix.vue` 第 12 行。

**問題**:表上 Flow 的取樣速度給了定值 2(最高)。但 flow 的取樣速度是**框架內的設計選擇**:MAF 類密度快、取樣慢(逐維自迴歸),IAF 類反過來,coupling flow(RealNVP/Glow)兩邊都快。這正是第 586–587 行「虛線格代表該欄的高低是框架內的設計選擇」要處理的情形,標準應該一致套用。

**建議**:`{ n: 'Normalizing Flow', v: [2, 1, [0, 2], 2, 2], note: '取樣快慢看耦合方向' }`。

---

### [MINOR] 「高斯 likelihood = MSE」需要固定變異數

**位置**:`components/VaeDefects.vue` 第 15 行;第 1127 行講者備註。

**問題**:$-\log\mathcal{N}(x;\hat x,\sigma^2 I)=\frac{\|x-\hat x\|^2}{2\sigma^2}+\text{const}$ 只有在 $\sigma$ 固定時才與 MSE 等價(差一個常數與正比係數)。若 decoder 也輸出 $\sigma(z)$,就變成加權 MSE 加上 $\log\sigma$ 項,行為明顯不同(這也是很多 VAE 實作把 $\sigma$ 設成超參數的原因)。

**建議**:改為「高斯 likelihood(**固定變異數**)= MSE。一個 z 對應多個合理輸出時,最優解是平均。」

---

## 未說出口的前提

以下每一條都是「結論其實依賴它、但簡報沒說」的前提。按會造成的外推傷害排序。

**P1 · 模型族無法表示 $p_{\text{data}}$(misspecification)。這是全課的隱形地基。**
如果 $p_\theta$ 的表達力足以精確表示 $p_{\text{data}}$,那麼 forward KL、reverse KL、JSD 的最佳解**全都是同一個** $p_\theta=p_{\text{data}}$,散度的選擇不影響任何東西。整堂課的因果鏈(散度 → 失效模式)只在「$q$ 罩不住 $p$」時才啟動——B-1 那張圖之所以會分岔,正是因為刻意把 $q$ 限制成單一高斯。

這個前提在第 861–876 行被自己踩到了:那一頁用 proper scoring rule 論證「模型仍能收斂到真實條件分布」——**那是可實現情形的結論**。同一堂課先說「forward KL 讓你的 LLM 過度平滑」,再說「CE 會讓你的 LLM 收斂到真條件分布」,兩句話都對,但只有加上「在模型族罩不住真分布的前提下 / 在模型族罩得住的極限下」才不矛盾。

*建議*:在第 250 行 B-1 那頁的開頭加一句,並在第 700 行、第 867 行各回扣一次:

> 注意這張圖的前提:$q$ 被限制成**單一高斯**。如果 $q$ 的表達力足夠罩住 $p$,三個散度的最佳解會是同一個,今天整堂課就不必上了。
> **散度的選擇之所以決定失效模式,是因為模型永遠不夠大。**

**P2 · JSD 飽和需要「兩個分布的 support 幾乎不重疊」。**
第 334–351 行整頁靠這個前提,而它在真實影像 GAN 上之所以成立,是因為 $p_{\text{data}}$ 與 $p_g$ 都落在低維流形上(Arjovsky & Bottou 的 Thm 2.1–2.2)。第 1211 行把「最佳判別器」與「梯度消失」直接畫上等號,其實中間少了這一步:**$D$ 最優 + support 不重疊**才會梯度消失;support 重疊時最佳判別器給的梯度是好的。
*建議*:第 344 行後補一句「前提是兩個分布的 support 幾乎不重疊——高維資料落在低維流形上,這件事幾乎必然發生。」

**P3 · $V=2\,\mathrm{JSD}-2\log2$ 與 NS loss 的分解都要求 $D=D^*$。**
第 365 行完全沒說,第 1211 行說了但只針對前者。詳見上面的 MAJOR 條目。

**P4 · 「只剩下訓一個分類器」的前提是「你已經決定要用 JSD」。**
第 546 行的「只剩下……這條路」在 JSD 的框架內是對的(算 JSD 需要密度比,而估計密度比的標準做法就是訓分類器)。但作為第 545 行那句「判別器是計算上的必然」的支撐,它漏掉了:**還有一整類只需要樣本、完全不需要密度也不需要判別器的散度**——MMD / kernel two-sample test(第 725 行自己列了 MMD),以及各種 IPM。真正被逼出來的是「必須改用只靠樣本的評估基準」;「用判別器」是其中一支。
*建議*:第 546 行後補一句:「嚴格說,被逼出來的是『改用只靠樣本的基準』。判別器是其中一條路(對應 JSD),MMD 是另一條(對應 IPM)——GAN 選了前者。」

**P5 · $\log$ 的底數。**
第 322 行的 $\log 2$、第 339 行的 $\log2$、第 348 行的 0.693 都是自然對數;第 751 行的 BPB 用的是 $\log_2$。兩者在同一份簡報裡並存,沒有任何一處說明。學生自己實作時最常見的錯誤就是把 JSD 的上界寫成 1(bits)還是 0.693(nats)搞混。
*建議*:第 322 行改成「$0\le\mathrm{JSD}\le\log2\approx0.693$(自然對數;若用 bits 則上界為 1)」。

**P6 · 最大化 ELBO $\ne$ 最小化 forward KL。**
$\mathbb{E}_{p_{\text{data}}}[\mathrm{ELBO}]=-\mathrm{KL}(p_{\text{data}}\|p_\theta)+\text{const}-\mathbb{E}_{p_{\text{data}}}\big[\mathrm{KL}(q_\phi(z|x)\|p_\theta(z|x))\big]$。
多出來的那一項**跟 $\theta$ 有關**,所以 VAE 的最佳解不是 forward KL 的最佳解——它會被推向「後驗容易被 $q$ 近似」的 $\theta$。第 962–964 行「同一個 forward KL,只是退而求其次取下界」的說法,對「下界」是對的,對「同一個最佳解」不對。這也意味著第 1119–1133 行「VAE 的模糊來自 forward KL」少了一個成因:攤銷/變分間隙本身會把 $\theta$ 帶偏。
*建議*:第 1038 行後補一句:「注意這一項也含 $\theta$,所以最大化 ELBO 會把 decoder 推向『後驗好近似』的那一側 —— VAE 的最佳解並不等於 forward KL 的最佳解。」

**P7 · 所有 forward KL 的討論實際上是對經驗分布 $\hat p_{\text{data}}$。**
第 216 行起講的是 $\mathrm{KL}(p_{\text{data}}\|p_\theta)$,實作上最小化的是 $\mathrm{KL}(\hat p_{\text{data}}\|p_\theta)$。第 863 行 TokenBars 的「語料裡出現過一次就不能給 0」講的正是經驗分布的性質——真分布沒有這種硬性 one-hot 結構。這個落差就是 overfitting 與 label smoothing 存在的原因(第 695 行提到了 label smoothing 卻沒接上)。
*建議*:第 869 行後補半句:「而且這裡的 $p$ 是**經驗**分布——真分布不會在單一 token 上放 1。label smoothing 就是在承認這件事。」

**P8 · B-1 / 澄清四的所有結論綁在一組具體參數上。**
峰位 $\pm1.6$、峰寬 $0.55$、$q$ 是單一高斯。翻邊門檻對峰距極度敏感(見上面的表:$\pm1.0$ 時到 $w=0.3$ 都不翻,$\pm2.2$ 時 $w=0.45$ 就翻)。第 252 行強調「這是解出來的,不是示意圖」,那就同時有義務說明這是**哪一組**參數解出來的。
*建議*:第 252 行改為「下面的曲線是在『雙峰位於 ±1.6、峰寬 0.55、$q$ 限定為單一高斯』這組設定下,數值最小化各自的散度解出來的,不是示意圖。」

**P9 · Spectral Norm「直接控制 $D$ 的 Lipschitz 常數」需要激活函數是 1-Lipschitz。**
第 1353 行。逐層除以最大奇異值只有在 ReLU / LeakyReLU(斜率 ≤ 1)這類激活下才把整個網路的 Lipschitz 常數乘起來 ≤ 1。這是實務上會踩的坑,一句話可以帶過。

---

## 絕對化措辭審查

| 行號 | 措辭 | 是否成立 | 說明 / 建議 |
|---|---|---|---|
| 121 | $p(y\mid x)$ **永遠** well-defined | ❌ 不成立 | 真分布的條件在 $p(x)=0$ 處無定義;要說的是模型的 $q_\theta$。見 MINOR |
| 167 | 這張表**不可能存在** | ✅ 成立 | $256^{786432}\approx10^{1.9\times10^6}$,無爭議 |
| 220 | $q$ **必須**覆蓋 $p$ 的**全部** support | ✅ 成立 | 正測度集上 $q=0$ 即 $\mathrm{KL}=\infty$;零測度集不影響 |
| 322 | **恆**為有界 | ✅ 成立 | $0\le\mathrm{JSD}\le\log2$,已數值驗證 |
| 327 | 最大**只能**是 log 2 | ✅ 成立 | $m\ge p/2\Rightarrow p/m\le2$ |
| 339 | 貼上 log 2 **不再變動** | ⚠️ 需條件 | 需 support 幾乎不重疊(P2);且該頁的 KL 曲線因 bug 也不再變動(CRITICAL) |
| 399 | forward KL **從不**倒向任一側 | ❌ 不成立 | 同頁圖例 $\mu=0.64$。見 MAJOR |
| 399 | reverse KL **一律**倒向單側 | ✅ 成立(本設定下) | $w=0.5$ 時左右簡併,「倒向哪一側」是任意的。見 MINOR |
| 398 | **一旦**不對稱就倒向大峰 | ❌ 不成立 | $w=0.4$ 反例;門檻還取決於峰距。見 MAJOR |
| 410 | 與 reverse KL **完全相同** | ✅ 成立 | 兩者都是 (1.60, 0.58),已驗證 |
| 518 / 533 | $p_{\text{data}}$ 密度**永遠**拿不到 | ✅ 成立 | 本課的核心前提,無異議 |
| 545 | 判別器是計算上的**必然** | ⚠️ 需條件 | 前提是「已選定 JSD」;只需樣本的 MMD 是另一條路。見 P4 |
| 546 | **只剩下**訓一個分類器 | ⚠️ 需條件 | 同上 |
| 606 | 短板**一定**存在 | ⚠️ 需條件 | Trilemma 是經驗性歸納,不是定理;第 611 行自己說位置可移動 |
| 693 | 兩者**恆等** | ✅ 成立 | one-hot 下 $H(p)=0$ |
| 700 | **全部原封不動**適用於你手上的 LLM | ❌ 不成立 | 需要 P1(模型族罩不住)與 P7(經驗分布);且真實 LLM 都經過後訓練。建議改為「② 段的 mode-covering 性質,在『模型罩不住真分布』的前提下原封不動適用於預訓練階段的 LLM」 |
| 711 | **無法**蒙地卡羅估計 | ✅ 成立 | MC 需要 $\log p$ 的數值 |
| 713 | 絕對數值**沒有意義** | ❌ 過強 | 絕對值 = 編碼長度,有意義;沒意義的是「離真分布多遠」。見 MINOR |
| 828 | 因此**只能**輸入真實前綴 | ✅ 成立 | forward KL 的期望就取在 $p$ 的前綴上 |
| 876 | 最小化期望 CE 的**唯一**解就是真實條件機率 | ⚠️ 需條件 | 在所有分布上取極值時成立(proper scoring rule);參數族內要加可實現性。見 P1 |
| 939 | **永遠**選出最短的候選 | ❌ 不成立 | 是偏誤不是必然。見 MINOR |
| 1057 | **一律**是保守估計 | ✅ 成立 | ELBO $\le\log p$,間隙 $\ge0$ |
| 1234 | 同時也換掉了 mode-covering | ❌ 不成立 | 原目標(JSD)本來就不是。見 MAJOR |
| 1255 | 判別器的介面裡**沒有**這個欄位 | ❌ 不成立 | $D^*$ 就編碼了覆蓋度;問題在梯度的取樣位置。見 MAJOR |
| 1272 | **永遠不會**指出「回應分布過窄」 | ⚠️ 需條件 | reward model 本身成立;整個 RLHF 目標有 $\beta\cdot$KL。見 MINOR |
| 1278 | 在 RLHF 裡**原封不動**地重演 | ⚠️ 需條件 | 同上 |
| 1359 | 四個工作,**同一個**動作 | ⚠️ 需條件 | R1 是局部梯度懲罰,不是 Lipschitz 約束。見 MINOR |
| 1414 | 所以它們**一樣**穩定 | ⚠️ 需條件 | 「每步退化成簡單監督問題」是穩定的必要條件之一,不是充分條件;且 AR 是分類不是回歸 |
| 1505 | mode-covering:**覆蓋所有** mode | ⚠️ 需條件 | VAE 優化的是下界,posterior collapse / 容量不足仍可能漏 mode。建議「傾向覆蓋所有 mode」 |
| 1599 | **會被一併繼承**下去 | ✅ 可接受 | 論述性宣稱,不是數學宣稱 |
| 1633 | 被 forward KL **逼著**覆蓋**全部** support | ⚠️ 需條件 | 同 P1;在模型罩得住時不成立 |
| ElboGap.vue:50 | 間隙**無法**直接量測 | ❌ 過強 | IWAE / AIS 可夾。見 MINOR |

---

## 驗證指令彙整

所有指令在 `/media/8tsp/projects/orientation-2026/genai` 下執行,`node v22.18.0`。

**V1 · 作者提供的檢查(確認 B-1 的六組參數)**

```bash
node components/divergence-math.check.mjs
```

輸出與投影片第 262、287、409–410 行完全相符。

**V2 · CRITICAL:KL 的數值 vs 解析值**

```bash
node /tmp/r1kl.mjs     # 腳本內容見 CRITICAL 條目
```

**V3 · MAJOR:forward KL 的 μ 隨 w 移動**

```bash
node -e "import('./components/divergence-math.js').then(({bimodal,fitGaussian})=>{
  for(const w of [0.5,0.4,0.3,0.2,0.1])
    console.log(w, JSON.stringify(['forward','jsd','reverse'].map(k=>fitGaussian(k,bimodal(w)))))})"
```

**V4 · MAJOR:JSD 翻邊門檻對峰距的敏感度**

```bash
node /tmp/r1sep.mjs    # 掃峰位 ±1.0 / ±1.6 / ±2.2 × w ∈ [0.3, 0.5]
```

**V5 · MAJOR:鏈鎖分解與「只換下標」的差異**

```bash
node /tmp/r1chain.mjs  # 腳本內容見該 MAJOR 條目
```

**V6 · MINOR:w=0.5 的 reverse KL 簡併**

```bash
node /tmp/r1tie.mjs    # 印出 μ=±1.6 兩點的 reverse KL,差 1.1e-16
```

**V7 · 確認 EPS 修正不會動到 B-1 的參數**

```bash
node /tmp/r1fit.mjs    # 以 EPS=1e-12 與 1e-300 各跑一次 fitGaussian,六組參數一字不差
```

---

## 給作者的優先順序建議

若時間有限,按這個順序處理即可覆蓋 90% 的風險:

1. **CRITICAL(改三行)**:`divergence-math.js:5` 的 `EPS`、`JsdSaturate.vue:32` 的 `VMAX`、第 348 行的講者備註。這是唯一會在課堂上當場被學生看穿的問題。
2. **第 399、398 行**:把「倒不倒」改成「平移 vs 跳邊」,並加上翻邊門檻的條件。同一頁的圖例正在反駁正文。
3. **第 762、747 行**:BPB 的上界方向與 telescope 範圍。這兩句學生會直接拿去用。
4. **第 843 行**:reverse KL 分解的下標要換兩個地方。這是下堂課 DDO 的地基。
5. **第 1252–1255 行**:mode collapse 的機制改成「梯度只在 $G$ 已產出的樣本上存在」。改完之後與第 1307 行的 demo 頁反而互相加強。
6. **P1(加三句話)**:在 B-1 那頁、第 700 行、第 867 行各補一次「模型族罩不住真分布」這個前提。這是整份簡報唯一的結構性缺口。
