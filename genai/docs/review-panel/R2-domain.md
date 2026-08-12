# R2 · 領域審查意見

審查件:`lecture-01-choose-a-ruler-v2.md`(1910 行)+ `components/*.vue`
審查範圍:文獻歸屬、術語、遺漏對照、分類法、領域共識。**未**審查教學設計、時間配置、數學推導。
查證方式:全部經 arXiv 摘要頁 / ar5iv 全文 / ACL Anthology 實查,未憑記憶。

---

## 總體評價

這份簡報的骨幹論證(「散度的選擇決定失效模式」)在領域上是站得住的,而且比多數同類教材更誠實——它主動處理了 JSD 不是對稱化 KL、non-saturating loss 不是 JSD、diffusion 與 flow matching 不是兩個家族這三件常被講錯的事。**引用的年份大體正確,32 筆查證中 24 筆完全無誤。** 這是好簡報。

但它有一個系統性的問題:**簡報最有力的幾句話,都是「必然性」句型**——「不是巧思,是計算上的必然」「同一個動作」「這條迴路上沒有任何地方能傳遞它」「唯一的分類依據」。這種句型在教學上非常有效,在領域上卻是最容易被單一反例擊倒的。而本次查證找到的三條 CRITICAL,全部落在這個句型上,其中兩條的反例**就寫在簡報自己引用的那份文獻裡**(Goodfellow 2016 Tutorial 的 Figure 9 把 GMMN 和 GAN 放在同一個葉節點;Mescheder et al. 的 R1 論證從頭到尾沒有提 Lipschitz)。

三條 CRITICAL 的修法都不是刪掉論點,而是**把「唯一」換成「主流,以及它為什麼是主流」**——修完之後論點反而更強,因為反例本身就是論點的證據(例如 PacGAN 之所以要改判別器的介面,正好證明了問題出在介面)。

第二個系統性問題是**年份標註規則被自己違反**。規則寫在 1883 行的講者備註:「年份以 arXiv 首發為準,正式發表場次略有出入,引用時依情境標註。」查證 23 筆有 arXiv 版本的引用,**5 筆違反**,而且違反與遵守的兩種標法出現在同一張表裡(1141–1148 行:IWAE 標 2015 是 arXiv 年,VAE-GAN 標 2016、VampPrior 標 2018 是會議年)。

第三,**Goodfellow et al. 2014 從頭到尾沒有被引用一次**,但簡報用了它的 Proposition 1(最佳判別器)、Theorem 1(V = 2JSD − 2log2)與 non-saturating loss 三個結果。

---

## 文獻查證表

「簡報標註」欄的行號指主檔;`X.vue:n` 指元件。

### A. 查證正確,不用動

| 工作 | 簡報標註 | 查證結果 | 判定 |
|---|---|---|---|
| Kingma & Welling, VAE | 1853:`(2013)` arXiv:1312.6114 | v1 2013-12-20 | ✓ |
| Arjovsky & Bottou | 1239, 1854:`(2017)` | arXiv:1701.04862 v1 2017-01-17 | ✓ 年份對(主張見 M3) |
| Goodfellow, NIPS 2016 Tutorial | 575, 1851: arXiv:1701.00160 | v1 2016-12-31, Ian Goodfellow | ✓ 編號/標題對(內容見 M2) |
| Xiao, Kreis & Vahdat, Trilemma | 615–616, 1852: ICLR 2022 / arXiv:2112.07804 | 三作者、會議、編號全對;`Trilemma.vue:59-66` 的三頂點(品質/速度/覆蓋)與各家族落位與原圖 Fig.1 一致 | ✓ |
| Higgins et al., β-VAE | 1110: ICLR 2017 | 正確;此文無 arXiv 預印本,以會議年標註是合理例外 | ✓(建議註明例外,見 m9) |
| WGAN | 1351: 2017 | arXiv:1701.07875 v1 2017-01-26 | ✓ |
| WGAN-GP | 1352: 2017 | arXiv:1704.00028 v1 2017-03-31 | ✓ |
| Spectral Norm | 1353: 2018 | arXiv:1802.05957 v1 2018-02-16 | ✓ |
| Mescheder et al., R1 | 1354, 1366: 2018 | arXiv:1801.04406 v1 2018-01-13,標題《Which Training Methods for GANs do actually Converge?》 | ✓ 年份/作者/標題對(主張見 C2) |
| cGAN | 1377: 2014 | arXiv:1411.1784 v1 2014-11-06 | ✓ |
| DCGAN | 1378: 2015 | arXiv:1511.06434 v1 2015-11-19 | ✓ |
| Progressive GAN | 1379: 2017 | arXiv:1710.10196 v1 2017-10-27 | ✓ |
| StyleGAN 1/2/3 | 1380: 2018–21 | 1812.04948(2018-12)/ 1912.04958(2019-12)/ 2106.12423(2021-06) | ✓ 年份區間對(AdaIN 敘述見 m1) |
| BigGAN | 1381: 2018 | arXiv:1809.11096 v1 2018-09-28;大 batch + truncation trick 描述正確 | ✓ |
| VQ-VAE | 1143, 1869: 2017 / arXiv:1711.00937 | v1 2017-11-02, van den Oord et al. | ✓ |
| VQ-VAE-2 | 1143: 2019 | arXiv:1906.00446 v1 2019-06-02 | ✓ |
| NVAE | 1144, 1870: 2020 / arXiv:2007.03898 | v1 2020-07-08, Vahdat & Kautz | ✓ |
| IWAE | 1148: 2015 | arXiv:1509.00519 v1 2015-09-01, Burda/Grosse/Salakhutdinov | ✓ |
| VampPrior 的內容描述 | 1146:「pseudo-input 的 posterior 混合」 | 原文:"a mixture distribution with components given by variational posteriors conditioned on learnable pseudo-inputs" | ✓ 轉述精確 |
| Song et al. §4.3 = probability-flow ODE | 1477–1478 | 實查 ar5iv:§4.3「Probability flow and connection to neural ODEs」 | ✓ 章節號正確 |
| Kalai et al. | 910, 1855: (2025) | arXiv:2509.04664 v1 2025-09-04;作者 Kalai, Nachum, Vempala, Zhang | ✓ 年份/作者對(主張見 m3) |
| (QA)² benchmark | 910 | Kim, Htut, Bowman, Petty;ACL 2023;arXiv:2212.10003 | ✓ 名稱正確(建議補編號) |
| HiFi-GAN | 1872: Kong et al. / arXiv:2010.05646 | v1 2020-10-12, Kong, Kim, Bae | ✓ |
| StyleGAN2 | 1871: Karras et al. / arXiv:1912.04958 | ✓ | ✓ |
| Stable Diffusion 第一層 | 1153, 1165 | 正確:LDM autoencoder 為 KL 正則 + patch 判別器 + LPIPS | ✓ |
| V(D\*,G) = 2·JSD − 2log2 | 365 | Goodfellow et al. 2014 Thm 1:C(G) = −log4 + 2·JSD;−log4 = −2log2 | ✓ 式子對(出處見 C3) |
| √JSD 滿足度量公理 | 322 | 正確(Endres & Schindelin 2003;Fuglede & Topsøe 2004) | ✓ |
| JSD = I(X;Z), Z~Bern(1/2) | 373 | 正確 | ✓ |
| CE = forward KL(one-hot 時) | 693 | 正確 | ✓ |

### B. 需要修正

| 工作 | 簡報標註 | 查證結果 | 需修正 |
|---|---|---|---|
| VDVAE | 1144:**2021** | arXiv:2011.10650 v1 **2020-11-20**(Rewon Child;ICLR 2021) | ✗ → 2020 |
| VAE-GAN (Larsen et al.) | 1145:**2016** | arXiv:1512.09300 v1 **2015-12-31**(ICML 2016) | ✗ → 2015 |
| VampPrior | 1146:**2018** | arXiv:1705.07120 v1 **2017-05-19**(AISTATS 2018) | ✗ → 2017 |
| Song et al., SDE | 1466, 1477:**(2021)** | arXiv:2011.13456 v1 **2020-11-26**(ICLR 2021) | ✗ → 2020 |
| Lipman et al., Flow Matching | 1467, 1476:**(2023)** | arXiv:2210.02747 v1 **2022-10-06**(ICLR 2023) | ✗ → 2022 |
| Lipman et al. 章節指引 | 1476:「§3 就是把 diffusion path 寫成 CFM 的特例」 | 實查 ar5iv:§3 是「Flow Matching」;diffusion 為特例在 **§4.1 "Special instances of Gaussian conditional probability paths", Example I: Diffusion conditional VFs** | ✗ → §4.1 |
| Goodfellow et al. 2014 | **完全未出現**(全檔 grep 無 1406.2661、無「Goodfellow, 2014」) | D\* 推導、V(D\*,G)、non-saturating loss 三者皆源自此文 | ✗ 補引用 |
| R1 的機制歸屬 | 1354, 1359, `GanFixes.vue:4,54` | Mescheder et al. §4.1 + Thm 4.1 是**局部收斂**論證,全文未以 Lipschitz 框架 R1 | ✗ 見 C2 |
| GAN 判別器「必然性」 | 545–546, `ComputeMap.vue:31-40` | Goodfellow 2016 Tutorial Fig.9 的 implicit/direct 葉節點同時列 GAN **與 kernel moment matching** | ✗ 見 C1 |
| 逐點介面「無法傳遞覆蓋」 | 1254–1256, `AdversarialLoop.vue:56` | PacGAN (Lin et al., NeurIPS 2018, arXiv:1712.04086) 與 minibatch discrimination (Salimans et al. 2016) 正是把批次餵給 D 以偵測塌縮 | ✗ 見 C3 |
| Diffusion「還是 MLE 路線」 | 1413 | DDPM 摘要自述為 "weighted variational bound";Kingma & Gao 2023 證明只有**均勻加權**才等於 ELBO | ✗ 見 M4 |
| VI 需要「reward / energy 代理」 | `ComputeMap.vue:15-22` | VI 的 log p̃(z,x) 是已知的(只差歸一化常數),正是 VI 可行的原因 | ✗ 見 M6 |
| 跨 tokenizer log-likelihood 不變 | 747 | 與同檔 762、923 直接矛盾 | ✗ 見 M11 |

---

## 問題清單

### [CRITICAL] C1 · 「判別器是計算上的必然」被簡報自己引用的那份文獻否證

**位置**:544–546 行;`components/ComputeMap.vue:31-40`(JSD 列 → 判別器代理 → GAN);連帶 552 行講者備註「這是被逼出來的,不是被發明出來的」

**問題**:
> 「JSD 兩邊的密度都拿不到,**只剩下**「訓一個分類器去逼近那個比值」這條路。」

這句是全課「最該被記住的一頁」(550 行自述)的結論句,但它是錯的。在「只有樣本、沒有密度」的條件下,至少還有一整條**核方法**路線:GMMN(Li et al., ICML 2015;Dziugaite et al., UAI 2015)與 MMD-GAN(Li et al., NeurIPS 2017)用 kernel MMD 這個**雙樣本統計量**直接量兩個分布的差距,**不需要密度,也不需要學任何分類器**。MMD 在 support 完全不重疊時仍良好定義且有梯度——這正是簡報在 ② 段花兩頁描述的 JSD 病理,而 MMD 天生沒有。

更關鍵的是:簡報在 575 行把這棵樹的出處指給 Goodfellow 2016 Tutorial,而**該文 Figure 9 的 implicit/direct 葉節點就同時寫著 "GANs" 與 "kernel moment matching"**。簡報引用了那張圖,卻刪掉了那張圖上唯一的反例,然後用刪過的版本論證「必然性」。這是本次審查最嚴重的一條。

**依據**:
- Goodfellow, *NIPS 2016 Tutorial: Generative Adversarial Networks*, arXiv:1701.00160, Fig. 9(ar5iv 全文查證:implicit / direct 葉節點含 GAN 與 kernel moment matching)
- Li, Swersky & Zemel, *Generative Moment Matching Networks*, ICML 2015;Dziugaite, Roy & Ghahramani, UAI 2015(兩篇同年獨立提出)
- 查證摘要原文:「Unlike f-divergences, MMD is well defined even for distributions that do not have overlapping support」

**建議**:
把 544–546 行的斷言句改成**主流性**句 + **交代反例**,論點反而更完整。具體改法:

> GAN 的判別器不是設計上的巧思,是**計算條件逼出來的少數幾條路之一**。
> JSD 兩邊的密度都拿不到,可走的只有兩種:(a) 訓一個分類器去逼近那個密度比值 → GAN;(b) 用純樣本的核統計量 → MMD / GMMN。
> (b) 不需要學基準,但需要人選 kernel,而在高維影像上固定 kernel 的偵測力不足;於是實務上「學出來的基準」贏了。**這場勝負本身就是本課的主題:基準要嘛你選,要嘛你學,沒有第三種。**

`ComputeMap.vue` 的 JSD 列「由誰代理」欄位建議改成「判別器代理(或核統計量)」,並在 `family` 欄加一行小字 `GAN(主流)· GMMN / MMD-GAN`。

---

### [CRITICAL] C2 · R1 penalty 被說成 Lipschitz 約束,原論文從頭到尾不是這麼論證的

**位置**:1354 行(表格列)、**1359 行**(結論句)、`components/GanFixes.vue:4` 與 `:54`(右半圖標題「WGAN / GP / SN / R1 · 約束 D 的 Lipschitz 常數」)

**問題**:
> 「四個工作,同一個動作:**限制判別器的 Lipschitz 常數**。差別只在用什麼手段限制,以及限制得多硬。」

前三個(WGAN weight clipping、WGAN-GP、Spectral Norm)確實是 Lipschitz 約束。**R1 不是。** Mescheder et al. (2018) 的 R1 是**零中心梯度懲罰,只施加在真資料分布上**,論證框架是**訓練動力學在均衡點附近的局部收斂**,不是函數類的 Lipschitz 連續性。兩者的差別不是措辭:

- WGAN-GP 罰的是 `(‖∇f‖ − 1)²`,**雙邊**,施加在真假樣本的**插值點**上,目的是逼近 ‖∇f‖ = 1 的 Lipschitz 邊界;
- R1 罰的是 `‖∇_x D‖²`,**單邊歸零**,**只在真資料上**,目的是讓判別器在資料流形上不能生出正交於流形的非零梯度,從而讓 GAN 動力系統的雅可比在均衡點附近沒有純虛數特徵值。

把 R1 併進「同一個動作」會直接誤導學生:一個常見的後果是有人以為 R1 也該用插值點、或以為把 R1 的係數調大等於「限制得更硬的 Lipschitz」。

**依據**:ar5iv 全文查證 arXiv:1801.04406
- §4.1 原文:"The simplest way to achieve this is to penalize the gradient on real data alone: when the generator distribution produces the true data distribution and the discriminator is equal to 0 on the data manifold, the gradient penalty ensures that the discriminator cannot create a non-zero gradient orthogonal to the data manifold without suffering a loss in the GAN game."
- Theorem 4.1:"For small enough learning rates, simultaneous and alternating gradient descent … are both convergent to M_G × M_D in a neighborhood of (θ\*, ψ\*). Moreover, the rate of convergence is at least linear."
- 全文未以 Lipschitz 常數框架 R1。

**建議**:
1359 行改成:
> 四個工作,同一個目標:**不要讓判別器的地景變成一道懸崖**。前三個用 Lipschitz 約束(硬性限制斜率上界),R1 走的是另一條:只在真資料上把梯度罰向 0,理由來自**局部收斂分析**而不是 Lipschitz 連續性——這也是為什麼 R1 便宜(不用取插值點)卻能穩定 StyleGAN 全系列。

`GanFixes.vue:54` 的圖標題建議改為「WGAN / GP / SN(Lipschitz 約束)· R1(零中心梯度懲罰)· 共同效果:地景變平緩」,`:4` 的註解同步。
1354 行「對付什麼」欄的「收斂性」是對的,不用動;「局部收斂有理論保證」建議補上「(Thm 4.1)」。

---

### [CRITICAL] C3 · 「這條迴路上沒有任何地方能傳遞覆蓋度」是絕對化陳述,而反例是一整條研究線

**位置**:`components/AdversarialLoop.vue:56`;1254–1256 行;1276–1278 行講者備註(「答案是沒有」);1590 行

**問題**:
> `AdversarialLoop.vue:56`:「模式覆蓋不足」是分布層級的性質,**這條迴路上沒有任何地方能傳遞它**。
> 1276–1278:「判別器有沒有辦法告訴生成器『有一個 mode 沒被覆蓋』?」→「答案是沒有,因為它一次只看一個點。」

「一次只看一個點」是**標準判別器介面的性質**,不是 GAN 的性質,而且領域裡有明確的反例:
- **minibatch discrimination**(Salimans et al., *Improved Techniques for Training GANs*, NeurIPS 2016):在 D 裡加一層,讓每個樣本看得到同批次其他樣本的距離統計;
- **PacGAN**(Lin, Khetan, Fanti & Oh, NeurIPS 2018, arXiv:1712.04086):把 m 個樣本 concat 成一個「packed」輸入交給 D 聯合判定。作者用 Blackwell 的二元假設檢定結果**證明** packing 會自然懲罰塌縮的生成器。

這條的嚴重性不在於論點錯——論點方向是對的——而在於**這是全課四張投影片共用的支點**(GAN mode collapse、GAN demo、RLHF reward model 盲點、⑥ 的「基準可攜」),一個學生只要讀過 PacGAN 就會認為整條論證線垮了。而其實它沒垮,只是句子寫太滿。

**依據**:Lin et al., PacGAN, NeurIPS 2018(查證原文:"pass m 'packed' or concatenated samples to the discriminator, which are jointly classified"、"borrow analysis tools from binary hypothesis testing—in particular the seminal result of Blackwell—to prove a fundamental connection between packing and mode collapse");Salimans et al. 2016 §3.2。

**建議**:把「沒有任何地方能傳遞」改成「**標準介面裡沒有這個欄位;要傳遞它必須改介面**」,並在 1261–1265 行那份「其餘三層」清單裡加一條第 5 點:

> 5. 反證:改介面就能傳。minibatch discrimination(2016)、PacGAN(2018)把「一批樣本」而不是「一個樣本」餵給 D,塌縮就變得可偵測、也可被證明會被懲罰。
> **這正好確認了第 1 層的診斷:問題出在判別器的輸入介面,不是出在對抗訓練本身。**

`AdversarialLoop.vue:56` 改為:「模式覆蓋不足」是分布層級的性質,而 D 的輸入介面一次只裝得下一個樣本 → 這條迴路傳不了它。(改介面才傳得了:PacGAN)

1272 行對 RLHF reward model 的類比同步加一句限定:「除非把評分單位從『一則回應』換成『一批回應』」——這對做 RLHF 的學生是可執行的研究線索,比一句「一模一樣的盲點」有用得多。

---

### [MAJOR] M1 · 年份標註規則(1883 行)被違反 5 處,且違反與遵守混在同一張表裡

**位置**:規則在 **1883 行**講者備註:「年份以 arXiv 首發為準,正式發表場次略有出入,引用時依情境標註。」
違反處:**1144**(VDVAE 2021→2020)、**1145**(VAE-GAN 2016→2015)、**1146**(VampPrior 2018→2017)、**1466 / 1477**(Song et al. 2021→2020)、**1467 / 1476**(Lipman et al. 2023→2022)

**問題**:1141–1148 那張表最刺眼——同一張表六列,IWAE 用 arXiv 年(2015,正確),VAE-GAN 與 VampPrior 用會議年,VDVAE 也用會議年。學生無法從表面判斷哪個是哪個,而這張表的教學功能是建立時間線(「每一項都對準一個缺陷」的演進順序),年份標錯會直接把 VampPrior(2017)排到 NVAE(2020)之後,把 VAE-GAN(2015)排到 VQ-VAE(2017)之後——**兩處順序都反了**。

**依據**:全部經 arXiv 摘要頁實查(見上方查證表 B 區)。

**建議**:
- 1144:`**NVAE** (2020)、**VDVAE** (2020)`
- 1145:`**VAE-GAN** (2015)`
- 1146:`**VampPrior** (2017)`
- 1466–1467:`Song et al. (2020)` / `Lipman et al. (2022)`
- 1477–1478 的備註同步。
- 建議在 1883 行的規則後面加一句可執行的判準:「例外:僅在 OpenReview 發表、無 arXiv 預印本者(如 β-VAE),標會議年並註明。」這樣 1110 行的 `Higgins et al., ICLR 2017` 就是合規的例外而不是漏網。
- 1852 行 `Xiao et al. (ICLR 2022)` 與 615 行同時給了會議與 arXiv 編號,可保留,但建議統一寫成 `Xiao et al. (2021, ICLR 2022)`。

---

### [MAJOR] M2 · 分類樹掛著 Goodfellow 的名字,但根節點被換掉、兩個葉節點被刪、一個新家族被加進去

**位置**:**564–565 行**、**575 行**、`components/FamilyTree.vue:2, 10, 21, 64`

**問題**:
> 564:「**Goodfellow 那棵樹**之所以是這個形狀,不是因為有人想把模型分成六類,而是因為『p_θ(x) 寫不寫得出來』這一問只有三種答案」
> `FamilyTree.vue:64`:「**唯一的**分類依據:p_θ(x) 這個數值寫不寫得出來」

實查 Goodfellow 2016 Tutorial Fig. 9:

| | Goodfellow 原圖 | 簡報的樹(`FamilyTree.vue`) |
|---|---|---|
| 根節點 | **Maximum likelihood** | `分布逼近 / min D(p_data‖p_θ)`(`:10`) |
| explicit/tractable | FVBN、Nonlinear ICA | Autoregressive、Normalizing Flow |
| explicit/approx | Variational (VAE)、**Markov Chain (Boltzmann machine)** | VAE、**Diffusion / Flow Matching**(`:21`) |
| implicit | **GSN(Markov chain)**、GAN + **kernel moment matching**(direct) | GAN |

也就是:根節點換了(MLE → 一般化的分布逼近)、刪了 Boltzmann machine 與 GSN、刪了 kernel moment matching(見 C1)、加了 2016 年還不存在於該圖的 Diffusion / Flow Matching。**這棵樹已經是一棵新的樹**,只是形狀像。

同時 564 行的「六類」是 Goodfellow 原圖的葉數(FVBN / Nonlinear ICA / VAE / Boltzmann / GSN / GAN = 6),而簡報的樹只有 5 個葉節點——這句話在指自己看不到的東西。

另外 Goodfellow 原文有一條簡報沒有轉述的重要保留:該樹**只涵蓋以最大概似為原則的模型**,作者明言 "many of these models are often used with principles other than maximum likelihood"。簡報把根節點改成更廣的「分布逼近」之後,「唯一的分類依據是 p_θ 寫不寫得出來」這句就更不成立了——因為在最大概似以外,還有一整條「用什麼統計量比較兩個分布」的分類軸(那正是本課 ② 段的主題)。

**依據**:ar5iv 全文查證 arXiv:1701.00160 Fig. 9 及其說明段落。

**建議**:
1. 575 行改成:`出處:改編自 Goodfellow, NIPS 2016 Tutorial (arXiv:1701.00160) Fig. 9。原圖根節點為 maximum likelihood,且含 Boltzmann machine / GSN / kernel moment matching 三葉;本圖為配合本課主軸重繪。`
2. 564 行的「六類」改成「五類」,或直接改寫成:「這棵樹之所以是這個形狀,不是因為有人想把模型分成幾類,而是因為……」
3. `FamilyTree.vue:64` 的「**唯一的**分類依據」改成「**這棵樹的**分類依據」——保留力度,去掉可被單一反例擊倒的絕對量詞。

---

### [MAJOR] M3 · Arjovsky & Bottou Thm 2.5 的轉述漏掉了前提,而那個前提正是上一句要否定的東西

**位置**:**1211、1213、1217 行**;1239 行講者備註

**問題**:
> 1211:「V = 2JSD − 2log2 **只在最佳判別器下成立**,而那正是梯度消失的地方」
> 1213:「改用 non-saturating loss 之後,**生成器實際在最小化的是** KL(p_g‖p_data) − 2JSD(p_g‖p_data)」

實查原文:**Theorem 2.5**,前提是 "the optimal discriminator, **fixed for a value θ₀**",結論是**梯度相等**:`E_z[−∇_θ log D\*(g_θ(z))] = ∇_θ [KL(P_g‖P_r) − 2JSD(P_g‖P_r)]`。

兩個問題:

1. **前提一樣**。簡報的敘事是「JSD 那個式子要最佳判別器,所以實務上不是 JSD;換成 non-saturating 之後實際在最小化的是這個」。但這個替代式**同樣要求最佳判別器**。讀者會得到「換了 loss 就脫離了最佳判別器假設」的錯誤印象,而 Arjovsky & Bottou 的論點其實是相反的:他們是在**同一個假設下**證明兩種 loss 都有病(前者梯度消失,後者梯度變異數無限大且方向自相矛盾)。
2. **「實際在最小化」不精確**。定理講的是**在 θ₀ 這一點的梯度方向**與該泛函的梯度相同,不是說生成器在全域最小化那個泛函(該泛函帶負 JSD,並非有下界的良好目標——這其實正是 1226–1229 行「第二項帶負號 → 訓練不穩」要講的事)。

**依據**:ar5iv 全文查證 arXiv:1701.04862 Theorem 2.5,含前提字句 "the optimal discriminator, fixed for a value θ₀"。

**建議**:1211–1217 改寫為:

> V = 2JSD − 2log2 只在最佳判別器下成立,而那正是梯度消失的地方。
> 換成 non-saturating loss 並**不會**脫離這個假設。Arjovsky & Bottou (2017) Thm 2.5 證明:在同一個「最佳判別器、固定於 θ₀」的假設下,生成器收到的**梯度方向**等於
> ∇[KL(p_g‖p_data) − 2JSD(p_g‖p_data)]
> 注意這是梯度的等式,不是「生成器在最小化這個泛函」——第二項帶負號,它根本沒有下界。

1239 行備註補上定理編號:`Arjovsky & Bottou (2017) Thm 2.5 是這頁的依據(前提:最佳判別器,固定於 θ₀)。`

---

### [MAJOR] M4 · 「Diffusion / FM 還是 forward KL、還是 MLE 路線」與「兩者都退化成回歸」都需要限定

**位置**:**1413–1414 行**;`components/DecompAxes.vue:2, 47`;連帶 572 行、`FamilyTree.vue:21`

**問題**:
> 1413–1414:「還是 forward KL,還是 MLE 路線。差別只在沿什麼軸做鏈鎖分解:AR 沿序列,擴散沿噪聲尺度。**兩者的訓練目標都因此退化成簡單回歸,所以它們一樣穩定。**」

兩個獨立的問題:

**(a) 「還是 MLE 路線」。** DDPM 論文自己的摘要就寫 "training on a **weighted** variational bound"。Kingma & Gao (NeurIPS 2023, arXiv:2303.00848) 把這件事說死了:所有常用的 diffusion 目標都等於**不同噪聲尺度上 ELBO 的加權積分**,而「**均勻加權才對應到最大化 ELBO**;實務上為了樣本品質使用的是非均勻加權」,只有**單調**加權才能寫成「ELBO + 高斯資料增強」。也就是說,實務用的 `L_simple` 一般**不是** ELBO。Flow Matching 更遠:CFM 是純回歸目標,與 likelihood 的關係要另外經過重加權或 variational 版本才建立得起來。

這一點對本課特別要緊,因為 1413 是把 diffusion 收進 forward KL 陣營的**唯一一句論證**,而 `FamilyMatrix.vue:16` 給 Diffusion/FM 的「mode 覆蓋 = 2」也是靠這句話撐的。

**(b) 「兩者都退化成簡單回歸」。** AR 每一步是**分類**(categorical CE),不是回歸;`DecompAxes.vue` 左半自己寫的是「預測下一個 token / Σₜ KL(·)」,右半才寫「退化成回歸」。1414 行把兩邊都說成回歸,與自己的圖不一致。

**依據**:
- Ho, Jain & Abbeel, *DDPM*, arXiv:2006.11239 v1 2020-06-19,摘要 "weighted variational bound"
- Kingma & Gao, *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation*, NeurIPS 2023, arXiv:2303.00848(查證結論:uniform weighting ↔ ELBO;monotonic weighting ↔ ELBO + 資料增強;實務為非均勻)
- Lipman et al. arXiv:2210.02747:CFM 為回歸目標,論文本身不宣稱最大化 likelihood

**建議**:1413–1414 改為:

> 還是 MLE 血統,但要加一個星號。差別只在**沿什麼軸做鏈鎖分解**:AR 沿序列,擴散沿噪聲尺度。
> 兩者的每一步都退化成一個**有監督的逐點預測問題**(AR 是分類,擴散/FM 是回歸),都不需要對抗訓練——**這才是它們一樣穩定的原因**。
> ※ 星號:實務用的 `L_simple` 是**加權**變分下界,只有均勻加權才等於 ELBO(Kingma & Gao, 2023);Flow Matching 的 CFM 目標則是純回歸,與 likelihood 的關係要另外建立。「MLE 路線」指的是血統,不是說每一個實作都在最大化概似。

同時建議在 1851–1855 的資源清單補上 Ho et al. (2020) 與 Sohl-Dickstein et al. (2015)(見下方「建議補充的文獻」)。

---

### [MAJOR] M5 · 分類法層級混用:Flow Matching 訓練的就是 Normalizing Flow,卻被列成兩個家族

**位置**:`components/FamilyTree.vue:19` vs `:21`;`components/FamilyMatrix.vue:12` vs `:16`;590–596 行講者備註;1430–1467 行

**問題**:簡報用整整一頁(1430 行起)論證「diffusion 與 flow matching 不是兩個家族」,這頁是對的、也是這份簡報的亮點之一。但它漏掉了另一半:**Flow Matching 訓練的模型是 continuous normalizing flow(CNF)**——Lipman et al. 的第 2 節標題就是 "Preliminaries: Continuous Normalizing Flows"。所以分類樹與體質表同時擺著:

- `FamilyTree.vue:19` `Normalizing Flow`,掛在「**可精確計算**」下;
- `FamilyTree.vue:21` `Diffusion / Flow Matching`,掛在「**取下界 / 近似**」下;
- `FamilyMatrix.vue:12` Normalizing Flow **密度 = 2**(精確);
- `FamilyMatrix.vue:16` Diffusion / Flow Matching **密度 = 1**(只有下界)。

也就是同一個東西(用 ODE 把先驗搬到資料的可逆連續變換)被放在樹的兩個不同分支,還被指派了兩個互相矛盾的密度屬性。這正是簡報自己在 1430 行批評的毛病(「太多教材把兩者並列成兩個家族、還各給一組優缺點」),只是換了一組對象。

而且簡報自己兩頁後(1466 行、1477–1478 行)就引用了 probability-flow ODE——實查 Song et al. §4.1 標題就是 **"Exact likelihood computation"**。所以「Diffusion/FM 密度只有下界」在簡報內部被自己的引用推翻。

**依據**:
- Lipman et al., arXiv:2210.02747 §2 "Preliminaries: Continuous Normalizing Flows"(ar5iv 目錄實查)
- Song et al., arXiv:2011.13456 §4.3「Probability flow and connection to neural ODEs」→ §4.1 級別的 exact likelihood computation(ar5iv 實查,§4.3 引入 ODE,可用於精確概似計算)

**建議**(不需要重畫樹,只要把區分軸講明白):
1. `FamilyMatrix.vue:12` 的 Normalizing Flow `note` 改成「**離散層堆疊**的可逆網路(RealNVP / Glow)」;`:16` 的 note 改成「**連續時間**可逆流 + 學出來的向量場」。
2. `FamilyMatrix.vue:16` 的「密度」欄由定值 `1` 改成區間 `[1, 2]`(元件已支援區間,見 `:27` 的 `cell()`),備註「變分下界;走 probability-flow ODE 可精確計算,但要付積分成本」。這與 585–587 行「虛線格代表該欄的高低是框架內的設計選擇」的既有說明完全一致,不需要新機制。
3. 1432–1434 行那段「兩者都在學一個時間相關的向量場」後面加一句:「**順帶把第三個名字也收進來**:這個向量場積分出來的映射是可逆的,所以 Flow Matching 訓練出來的其實就是一個連續時間的 normalizing flow。樹上 NF 與 FM 分在兩個分支,分的是『離散層 vs. 連續時間』與『密度算得便不便宜』,不是兩種血統。」

---

### [MAJOR] M6 · ComputeMap 把變分推斷歸進「需要 reward / energy 代理」,但 VI 之所以可行正是因為它不需要

**位置**:`components/ComputeMap.vue:15-22`(reverse KL 列:`算 p_data 密度 ✗` → `reward / energy 代理` → `VI · RLHF`);連帶 636–643 行

**問題**:reverse KL 那一列說「算 p_data 密度 ✗,所以需要 reward / energy 代理」,然後把 **VI** 與 **RLHF** 並列為結果。這對 RLHF 是對的(reward model 確實是學出來的代理),對 **VI 是反的**:變分推斷最小化 KL(q(z)‖p(z|x)),而 `p(z|x) ∝ p(x,z)`,**`log p(x,z)` 是寫得出來的閉式**(先驗 × 似然),只差一個與 q 無關的歸一化常數 log p(x)。這正是 ELBO 能被寫下來、VI 能被最佳化的全部原因——簡報自己在 1024–1030 行就把這件事推了一遍。

**依據**:簡報自身 1024–1038 行的 ELBO 推導。

**建議**:把 reverse KL 那一列拆成兩個情境,或至少把 `stand` 欄改成條件式:

```js
{
  ruler: 'reverse KL', c: '#ff6b9d',
  needs: [
    { t: '從 p_θ 取樣', ok: true },
    { t: '算目標密度', ok: 'partial' },   // 需要新增第三種狀態
  ],
  stand: '能寫出非歸一化密度 → 不用代理;寫不出 → reward 代理',
  family: 'VI(不用代理)· RLHF(reward 代理)',
}
```
最省事的改法:`stand` 改成「(VI 不用)· RLHF 用 reward 代理」,`family` 保持不變,再在 546 行下方那段講者備註補一句:「VI 是這張表唯一的例外:它的目標密度只差一個歸一化常數,所以 reverse KL 直接算得動。**這也是為什麼 RLHF 的 β·KL 項可以有閉式,而 reward 項不行。**」

這條同時會讓 636–643 行(「reward 代理與判別器代理是同一件事」)更精確:兩者補的都是**寫不出來的那一項**,而不是「reverse KL 一律需要代理」。

---

### [MAJOR] M7 · 貫穿全課的那條軸把「JSD · GAN」放在正中央,與簡報自己的三處結論衝突

**位置**:`components/SpectrumAxis.vue:12`(`{ at: 50, text: 'JSD · GAN' }`);該元件出現於 **100、429、1649、1898** 四頁

**問題**:這條軸是全課的招牌視覺,GAN 被放在 covering ↔ seeking 的正中間(50%)。但簡報在三個地方推翻了這個位置:

- **398–404 行**:「兩峰一旦不對稱,**JSD 的解會倒向大峰**……跟 reverse KL 收斂到同一個地方」「『在中間』不等於『取平均』」;
- **1217–1224 行**:non-saturating loss 的第一項就是 reverse KL,「→ mode-seeking 的來源」;
- **`FamilyMatrix.vue:15`**:GAN 的「mode 覆蓋 = **0**」,與 reverse KL 端等值。

一個學生看完 ② 段再回頭看這條軸,會直接發現矛盾;而這條軸是下堂課要繼續填的骨架,位置錯了會一路傳下去。

**依據**:簡報自身 398–404、1217–1224 行與 `FamilyMatrix.vue:15`。

**建議**:把 `SpectrumAxis.vue:12` 拆成兩個 chip:

```js
{ at: 46, text: 'JSD(理論)', c: '#ffb454' },
{ at: 72, text: 'GAN(non-sat. 實務)', c: '#ffb454' },
```

並在 401 行那段補一句回指:「**所以這條軸上 GAN 的位置比你以為的偏右**——理論上的 JSD 在中間,實務用的 loss 已經滑到 reverse KL 那邊了。」這剛好把 ② 段最精彩的那頁(澄清四)接進全課骨架。

---

### [MAJOR] M8 · LossTravel 把「RLHF 的 β·KL」放進 mode-covering 車道,與 1577 行直接矛盾

**位置**:`components/LossTravel.vue:14-15`(`behav: 'mode-covering'` 的車道,travel 清單含 `'RLHF 的 β·KL'`)vs **1575–1577 行**

**問題**:
- `LossTravel.vue:14-15`:上車道標 `mode-covering`,遷移目的地清單含「RLHF 的 β·KL」;
- **1576–1577 行**:「寫 KL(π_θ‖π_ref) 得到 seeking。**RLHF 用的是後者**,這是對齊後多樣性塌陷的直接來源。」

同一堂課的兩張投影片,一張說 RLHF 的 KL 是 covering,一張說是 seeking。而 1577 那句是對的(標準 PPO-RLHF 的懲罰項是 KL(π_θ‖π_ref)),所以錯的是元件。

同車道的其他三個成員沒問題:知識蒸餾用 KL(teacher‖student)、TRPO/PPO 信賴域用 KL(π_old‖π_θ)、label smoothing,都是 covering 方向。

**依據**:簡報自身 1576–1577 行;PPO-RLHF 的標準 KL 懲罰形式。

**建議**:兩種改法擇一——
(a) 把 `'RLHF 的 β·KL'` 從上車道移到下車道之外的**第三個位置**,並在圖上標出方向;
(b) 更省事:保留在上車道但改成 `'RLHF 的 β·KL(反向!)'`,並把車道的 `behav` 標籤從 `mode-covering` 改成 `'固定基準 · 方向可選'`。

我建議 (b),因為 1569–1578 行那一頁的核心訊息正是「**認出句型之後,方向是可選的**」——把 RLHF 當成同一句型的反向實例,比把它藏起來更有教學價值。改完之後 1606 行的備註(「左欄那句『方向是可以選的』要停 15 秒」)才真的有東西可指。

---

### [MAJOR] M9 · 「只能用代理指標:IS、FID」——遺漏了本課主軸最需要的那類指標

**位置**:**1510 行**;連帶 1523 行備註、作業 B(**1748 行**「記錄覆蓋到幾個 mode」)

**問題**:
> 「評估 | test log-likelihood(是下界) | **只能用代理指標:IS、FID**」

「只能」是錯的,而且錯得剛好打到本課的靶心。整堂課的論點是「品質與覆蓋是兩件事、由散度的選擇決定」,而 **FID 是一個單一純量,結構上無法區分這兩者**——Sajjadi et al. (2018) 的摘要就是這麼開場的:FID 無法判斷低分是來自高 precision(樣本逼真)還是高 recall(涵蓋廣)。而 precision/recall 系列指標存在,就是為了把這兩軸拆開:

- Sajjadi, Bachem, Lucic, Bousquet & Gelly, *Assessing Generative Models via Precision and Recall*, NeurIPS 2018
- Kynkäänniemi, Karras, Laine, Lehtinen & Aila, *Improved Precision and Recall Metric for Assessing Generative Models*, NeurIPS 2019
- Naeem et al., *Reliable Fidelity and Diversity Metrics*(Density & Coverage), ICML 2020

**缺了它,1748 行的作業 B 就沒有工具**:作業要求學生「記錄覆蓋到幾個 mode」,但簡報只教了 FID,而 FID 恰恰量不出這件事。學生只能靠肉眼數點。

**依據**:Sajjadi et al. 2018 摘要(查證原文轉述:"FID and related density metrics cannot determine whether low FID indicates high precision (realistic images), high recall (large variation), or anything in between");Kynkäänniemi et al. 2019 為其改良版。

**建議**:1510 行改為:

> | 評估 | test log-likelihood(是下界) | 代理指標:IS、**FID**;要分開品質與覆蓋則用 **precision / recall**(Sajjadi 2018;Kynkäänniemi 2019) |

並在 1523 行的備註加一句:「**FID 是單一純量,量不出『品質高但覆蓋窄』與『覆蓋廣但模糊』的差別——這正是本課那條軸的兩端。** 想量那條軸,要用 precision/recall 這一類把兩軸拆開的指標。作業 B 若有人問『覆蓋到幾個 mode 要怎麼量才算數』,答案就在這裡。」

---

### [MAJOR] M10 · 術語:「評估基準」同時指散度與模型評測,是全檔最容易誤解的一個詞

**位置**:作為「散度 / 尺」使用:142、179、573、1177、1185、1327、1333、1504、1532、1539、1599 行;`LossTravel.vue:11`(`ruler:`);`FamilyTree.vue:22`
作為「模型評測」使用:**1510 行**(表格「評估」列)、1803 行(「評分重點」)

**問題**:「評估基準」在中文 ML 語境裡的預設讀法是 evaluation benchmark(評測基準、benchmark suite)。簡報用它翻譯 divergence / objective,而**同一份簡報又在 1510 行用「評估」指 FID、IS、test log-likelihood 這些真正的評測指標**。1504 行(「評估基準 | KL…| 判別器…」)與 1510 行(「評估 | test log-likelihood | IS、FID」)在同一張表裡上下相鄰,一個指訓練目標、一個指評測指標,只差一個字。

這不是吹毛求疵:1532 行的段落標題「⑥ 評估基準是可攜的」,若讀成「評測基準是可攜的」,整段意思就變成「FID 可以搬到別的任務」——與原意完全無關。

**建議**:標題已經定了一個好詞——**「尺」**(第 3 行、24 行)。建議把技術用語統一為:

| 概念 | 建議統一用詞 | 目前散落的說法 |
|---|---|---|
| 訓練用的分布差異量 | **差異度量**(口語:**尺**) | 評估基準、基準、尺、散度、距離 |
| f-散度類 | **散度** | ✓ 已一致 |
| Wasserstein 等 | **距離** | ✓ 已一致(1338、1351) |
| 模型好壞的量測 | **評測指標** | 評估、評估基準 |

具體最小改動:把 142、1177、1185、1327、1504、1532、1539、1599 行與 `LossTravel.vue:11` 的「評估基準」改成「**差異度量**」或「**尺**」,1510 行維持「評估」。第 481 行的段落標題「③ 散度能不能算」在引入 Wasserstein(1338、1351)之後也不夠廣,建議改成「③ 這把尺能不能算」。

---

### [MAJOR] M11 · 「字串的 log-likelihood 與 token 切法無關」是錯的,而且與同檔兩處自相矛盾

**位置**:**747 行**;矛盾對象:**762 行**、**923 行**

**問題**:
> 747:「鏈鎖法則會 **telescope**:字串的 log-likelihood **與 token 切法無關**,會變的只有分母。」
> 762:「算的是 canonical 切法,是真實字串機率的**上界**;**不同 tokenizer 鬆緊不同**」
> 923:「同一個語意,**不同切法 → 不同的 token 序列 → 不同的 sequence probability**」

747 說無關,762 與 923 說有關。正確的是後兩者:模型算的是**某一條特定 token 序列**的機率,而真實字串機率是**所有能還原該字串的切法之和**,所以單一切法的值是下界(取負對數後是上界),不同 tokenizer 的鬆緊差異真實存在——762 行自己就寫對了。

另外「telescope」是誤用:鏈鎖法則是條件機率**連乘等於聯合機率**,不是望遠鏡式的相消。

**依據**:簡報自身 762 與 923 行。

**建議**:747 行改為:

> 鏈鎖法則保證同一條 token 序列的 log-likelihood 就是該序列的 log 聯合機率,**與模型怎麼切無關的是「單位」而不是「數值」**:換 tokenizer 之後 T 變了、每一項也變了,所以要比較就得把分母換成與切法無關的 bytes。

這樣 751 行的 BPB 公式與 762 行的三個 caveat 就一致了,而且 762 行的「上界」也有了根據。「telescope」直接刪掉,或改成「連乘」。

---

### [MINOR] m1 · StyleGAN 2/3 已經不用 AdaIN

**位置**:**1380 行**

**問題**:「**StyleGAN 1/2/3** | 2018–21 | style-based 生成器:z→w 空間、**AdaIN 逐層注入**」。StyleGAN2 就是為了消除 AdaIN 造成的「水滴」偽影,才把 AdaIN 換成 **weight demodulation**(modulated conv + demod)。把 AdaIN 寫成 1/2/3 的共同特徵是錯的。

**依據**:Karras et al., *Analyzing and Improving the Image Quality of StyleGAN*, CVPR 2020 / arXiv:1912.04958(查證:blob artifacts 追溯到 AdaIN,以 weight demodulation 取代)。

**建議**:「style-based 生成器:z→w 空間、逐層注入 style(**v1 用 AdaIN,v2 起改為 weight demodulation**)」。

---

### [MINOR] m2 · DALL·E 用的是 dVAE + Gumbel-softmax,不是 VQ-VAE

**位置**:**1143 行**「催生 DALL·E 路線」

**問題**:方向是對的(離散潛在 + AR prior 的路線確實由 VQ-VAE 開啟),但 DALL·E (Ramesh et al. 2021) 用的是 **dVAE**:同樣是離散 codebook,但用 Gumbel-softmax 從類別分布取樣,而不是 VQ-VAE 的最近鄰查表 + straight-through。1085 行的講者備註其實把這兩種離散化都提到了,只是沒接起來。

**建議**:1143 行改成「啟發 DALL·E 的離散 token 路線(DALL·E 實際用 dVAE + Gumbel-softmax)」,並在 1085 行備註後補一句:「這兩條離散化路線之後就分家了:VQ-VAE 的最近鄰 → VQGAN / Parti;Gumbel-softmax → DALL·E 的 dVAE。」

---

### [MINOR] m3 · Kalai et al. (2025) 的後半段與 889–891 行的結論有張力

**位置**:**889–891 行**;**910 行**(把該文列為配套閱讀)

**問題**:889–891 行說「用 forward KL 訓練的模型,結構上就沒有『拒絕回答』這個選項,**除非後訓練另外教它**」,而 910 行把 Kalai et al. (2025) 掛在同一頁當配套閱讀。但該文的核心論證是**兩段**的:預訓練的統計壓力產生幻覺(這半支持簡報),**以及後訓練/評測階段的誘因結構讓幻覺存續**——因為主流 benchmark 用二元計分,棄答與答錯同樣得 0 分,所以「猜」在期望上優於「說不知道」。作者明言解法是**改現有 benchmark 的計分方式**,而不是加新的幻覺評測。

也就是說,簡報的「除非後訓練另外教它」正好是該文說**目前做不到、而且原因不在技術**的那一步。學生讀了原文會覺得簡報把它讀反了一半。

**依據**:arXiv:2509.04664 摘要(查證原文轉述:"language models are optimized to be good test-takers, and guessing when uncertain improves test performance";主張修改既有 benchmark 計分而非新增幻覺評測)。

**建議**:891 行後補一句:

> 而後訓練也不會自動教會它——Kalai et al. (2025) 指出主流評測用二元計分,棄答與答錯同樣是 0 分,所以「猜」在期望上永遠划算。**要讓模型學會停,得先讓計分規則允許它停。**

這句話對「虛假前提檢測」組的實務價值遠高於現在那個裸的閱讀清單條目。

---

### [MINOR] m4 · Lipman et al. 的章節指引錯了

**位置**:**1476 行**「Lipman et al. … 的 **§3** 就是把 diffusion path 寫成 CFM 的特例」

**依據**:ar5iv 實查 arXiv:2210.02747 目錄——§3 是 "Flow Matching",diffusion 為特例出現在 **§4.1 "Special instances of Gaussian conditional probability paths", Example I: Diffusion conditional VFs**。

**建議**:改成 `§4.1(Example I: Diffusion conditional VFs)`。同段的 Song et al. §4.3 指引經查是**正確的**,不用動。

---

### [MINOR] m5 · 「結構上就沒有拒絕回答這個選項」是作者詮釋,寫成了性質陳述

**位置**:**889–891 行**

**問題**:嚴格說,forward KL / MLE 只要求模型復刻語料中的 p(y|x)。如果語料裡有人回答「這個問題的前提不成立」,MLE 就會學到那個續寫。所以「**結構上**沒有這個選項」不是散度的性質,是**語料的性質**(自然文本中極少出現對虛假前提的顯式拒絕)加上散度的性質(0 機率會被無限懲罰,所以模型不敢把任何續寫壓到 0)兩者疊加的結果。

**建議**:改為:

> forward KL 只要求模型復刻語料裡 p(y|x) 的形狀。語料裡幾乎沒有人會回「這個問題的前提不成立」,所以模型也學不到。
> **這不是散度單獨造成的,是「散度只認語料」加上「語料裡沒有這種續寫」的合成結果——而這正好說明它為什麼難修:換模型不會好,得換訓練訊號。**

---

### [MINOR] m6 · 「JSD 依權重倒向一側」的普遍化來自單一 1D 玩具例

**位置**:**398–404 行**;409–412 行備註

**問題**:「兩峰一旦不對稱,JSD 的解會**倒向大峰**……跟 reverse KL **收斂到同一個地方**」是從一個特定設定(單一高斯 q、1D 雙峰、w=0.3)算出來的。「與 reverse KL 完全相同」這種精確重合是該設定的性質,不是 JSD 的一般性質;在別的 q 家族或別的權重下,JSD 的解會落在兩者之間的不同位置。

簡報的教學意圖(讓「折衷」變得可證偽)很好,不需要弱化,只需要把範圍講清楚。

**建議**:404 行改為:「『在中間』不等於『取平均』。在這個設定下它**依權重倒向其中一側**——而這已經足以說明:JSD 沒有承諾任何一種折衷,它的行為要看資料。」409 行的備註補上「這是 w=0.3、單一高斯 q 的結果;換 q 家族會落在別的位置」。

---

### [MINOR] m7 · 「鏈鎖法則」在 zh-TW 不是標準譯名

**位置**:570、747、781、1413 行;`DecompAxes.vue:2`

**建議**:台灣通用譯名為「**連鎖律**」或「**鏈式法則**」。統一改一種即可,五處都在同一個概念上,替換無風險。

---

### [MINOR] m8 · mode collapse 的中文說法在同一份簡報裡有六種

**位置**:120(多樣性塌陷)、285(塌陷)、449(模式塌縮 / 多樣性低)、1246 / 1308(mode collapse,英文)、1264–1265(塌縮)、1505 / 1520(模式覆蓋不足)、1577(多樣性塌陷)、1714(模式塌縮 / 多樣性喪失);`FailureScenes.vue:24`(多樣性塌陷)、`AdversarialLoop.vue:56` / `SpectrumAxis.vue:44`(覆蓋不足)

**問題**:六種說法橫跨兩個其實**不完全相同**的概念:
- **mode collapse**:訓練動力學層級的現象(生成器塌到少數幾個 mode,且會 hopping);
- **模式覆蓋不足 / 多樣性低**:結果層級的性質(輸出分布比目標窄)。

而作業 B(1765 行)要求學生「必須用到 mode-covering / mode-seeking / 逐點評分的盲點」這三個詞——如果講義本身用六種說法指涉這一組概念,學生很難知道該用哪個。這對一份把「用同一組詞彙描述三種行為」當成驗收點(1770 行)的課特別要緊。

**建議**:在 449 行那張對照表下方或 1246 行加一個一次性的用詞約定框:

> **本課用詞**:`mode collapse`(模式塌縮)= 訓練過程中生成器塌到少數 mode 的**現象**;`mode-seeking` = 造成它的**散度性質**;`覆蓋不足` = 觀察到的**結果**。三者不是同義詞,講因果時請分開用。

然後把 120 與 1577 的「多樣性塌陷」統一(LLM 側可保留這個詞,但要在上面那個框裡註明它是 mode collapse 在語言模型上的別名——**這正是本課論題「同一條軸的兩端」要建立的連結**,現在反而被用詞差異藏起來了)。

---

### [MINOR] m9 · 兩處年份標註方式與 1883 行的規則不一致(但可接受)

**位置**:1110 行(`Higgins et al., ICLR 2017`)、615 / 1852 行(`Xiao et al., ICLR 2022`)

**說明**:這兩處用會議年而非 arXiv 年。β-VAE 無 arXiv 預印本,是規則的合理例外;Xiao et al. 在 615 行同時給了 arXiv 編號,可追溯。兩者都**不算錯**,但建議在 1883 行的規則裡把例外寫明(見 M1 的建議),否則審閱者/學生無法判斷這是例外還是漏改。

---

### [MINOR] m10 · Diffusion 講了兩頁,但原始文獻一篇都沒引

**位置**:572 行、1408–1420 行、1430–1468 行、1851–1855 行(資源清單)

**問題**:簡報引了 Song et al. 與 Lipman et al.(兩篇都是**後續**的統一性工作),但 **Sohl-Dickstein et al. (2015)** 與 **Ho et al. (2020)** 一篇都沒有。572 行「顯式近似:……或一整條變分過程(diffusion)」這個定位的依據就是這兩篇。

**建議**:1851–1855 的「對應本堂特定段落」清單補兩行:
- `Sohl-Dickstein et al. (2015) — 擴散模型的原始構想(arXiv:1503.03585)`
- `Ho, Jain & Abbeel (2020) — DDPM,「加權變分下界」的來源(arXiv:2006.11239)`
(兩篇的 v1 日期均經查證:2015-03-12 / 2020-06-19。)

---

### [MINOR] m11 · 「潛在空間」與 latent space 中英混用

**位置**:中文:599、974、1099、1101、1508;英文:1153、611、622(latent diffusion)

**建議**:專有名詞(latent diffusion)保留英文,一般名詞統一用「潛在空間」。1153 行的「先把影像壓進 latent space」改成「潛在空間」即可。

---

## 建議補充的文獻(附:缺了它會讓哪一句站不住)

按「缺席造成的傷害」排序。前四項我認為是必補,後四項是加分。

| # | 文獻 | 缺了它,哪一句站不住 |
|---|---|---|
| **1** | **Goodfellow, Pouget-Abadie, Mirza et al., *Generative Adversarial Nets*, NeurIPS 2014, arXiv:1406.2661** | **361 行**的 D\*(x) = p/(p+q)、**365 行**的 V(D\*,G) = 2JSD − 2log2、**1191 行**的 min-max 目標、**1202 行**的 non-saturating loss——**四個結果全部來自這一篇,而它在全檔一次都沒出現**(grep 確認)。384 行還請學生口頭推導 D\*,卻不告訴他們這是誰的 Proposition 1。這是全份簡報最明顯的引用缺口。 |
| **2** | **Li, Swersky & Zemel, *GMMN*, ICML 2015;Li, Chang, Cheng et al., *MMD GAN*, NeurIPS 2017** | **546 行**「只剩下訓一個分類器去逼近那個比值這條路」——MMD 是純樣本統計量,不需要密度也不需要學分類器,而且在 support 不重疊時仍有梯度。缺了它,全課「最該被記住的一頁」(550 行)的結論句是假的。見 C1。 |
| **3** | **Sajjadi, Bachem, Lucic et al., *Assessing Generative Models via Precision and Recall*, NeurIPS 2018;Kynkäänniemi et al., NeurIPS 2019** | **1510 行**「只能用代理指標:IS、FID」是錯的;而且 **1748 行**的作業 B 要學生「記錄覆蓋到幾個 mode」,簡報卻沒給任何能量測覆蓋的工具——FID 是單一純量,結構上分不出品質與覆蓋。這一對指標正是本課那條軸的量測版本,不補等於論點做不出實驗。見 M9。 |
| **4** | **Lin, Khetan, Fanti & Oh, *PacGAN*, NeurIPS 2018, arXiv:1712.04086;Salimans et al., *Improved Techniques for Training GANs*, NeurIPS 2016** | **`AdversarialLoop.vue:56`**「這條迴路上沒有任何地方能傳遞它」與 **1276–1278 行**「答案是沒有」。這兩篇把整批樣本餵給 D,PacGAN 還用 Blackwell 的結果證明了 packing 會懲罰塌縮。補上之後論點反而更利:問題出在**介面**,不是出在對抗訓練。見 C3。 |
| 5 | **Nowozin, Cseke & Tomioka, *f-GAN*, NeurIPS 2016, arXiv:1606.00709**(v1 2016-06-02,經查證) | **422 行**「選擇散度 = 選擇我們願意承擔哪一種錯誤」目前只示範了三個散度,像是三個選項;f-GAN 證明**任何 f-散度**都能用變分下界估出來、都能拿來訓練生成器(原文:"the generative-adversarial approach is a special case of an existing more general variational divergence estimation approach")。缺了它,**481 行**「散度能不能算,決定它變成哪一個家族」看起來像三選一,實際上是一個連續的設計空間——而這正是本課想傳達的東西。 |
| 6 | **Theis, van den Oord & Bethge, *A note on the evaluation of generative models*, ICLR 2016, arXiv:1511.01844**(v1 2015-11-05,經查證) | **1057 行**「VAE 報出來的 likelihood 拿去跟 AR 的精確 likelihood 比是不公平的」——這句話目前只論證了「一個是下界、一個是精確值」,而真正更強的論證在這篇:平均 log-likelihood、Parzen 估計、樣本視覺品質**三者在高維下彼此大致獨立**,所以「likelihood 高 ⇒ 樣本好」本來就不成立。這也是 **1510 行**評估列最該引的一篇。 |
| 7 | **Kingma & Gao, *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation*, NeurIPS 2023, arXiv:2303.00848;Ho, Jain & Abbeel, *DDPM*, arXiv:2006.11239** | **1413 行**「還是 forward KL,還是 MLE 路線」。DDPM 摘要自述訓練於 "weighted variational bound";Kingma & Gao 證明只有均勻加權才等於 ELBO。見 M4。 |
| 8 | **Minka, *Divergence measures and message passing*, MSR-TR-2005-173** | **222 / 234 行**的 `zero-avoiding / zero-forcing` 這組術語的出處。不補不影響正確性,但研究生課給一個可查的來源,學生日後看到 α-divergence 家族才接得上。**(此條為我依記憶提供的推薦,編號未經本次線上查證,建議作者自行確認。)** |

另外針對 **1577 行**「RLHF 用的是後者,這是對齊後多樣性塌陷的**直接來源**」:這個因果歸屬在領域裡是**有支持但仍有爭議**的(reward over-optimization、best-of-n 式的取樣、以及 reverse KL 三者都被指認過)。目前寫成無條件的因果句。建議改成「**主要來源之一**」並補一個引用;相關工作我未在本次查證中逐一確認,建議作者自行挑一篇近期的 RLHF 多樣性研究補上,或把「直接來源」降級為「與 ② 段的 reverse KL 分析完全一致」。

---

## 術語一致性檢查

### 需要處理

| 概念 | 目前用詞(行號) | 問題 | 建議 |
|---|---|---|---|
| 訓練用的分布差異量 | 評估基準(142/179/573/1177/1185/1327/1504/1532/1539/1599、`LossTravel.vue:11`)、尺(3/24)、散度(全檔)、距離(1338/1351) | **「評估基準」與 1510 行的「評估」(FID/IS)撞詞**,且 1504 與 1510 在同一張表上下相鄰 | 統一為「**差異度量**」(口語「尺」);「評估」專留給模型評測。見 M10 |
| mode collapse | 多樣性塌陷 / 塌陷 / 模式塌縮 / mode collapse / 塌縮 / 模式覆蓋不足 / 多樣性喪失 / 多樣性低(共 6 種說法,9 處) | 跨越「現象 / 散度性質 / 結果」三個層級,但用同一組詞;作業 B(1765)卻要求學生精確用詞 | 加一個一次性用詞約定框,區分 `mode collapse` / `mode-seeking` / `覆蓋不足`。見 m8 |
| chain rule | 鏈鎖法則(570/747/781/1413、`DecompAxes.vue:2`) | zh-TW 非標準譯名 | 統一為「連鎖律」或「鏈式法則」。見 m7 |
| latent space | 潛在空間(599/974/1099/1101/1508)/ latent space(1153) | 中英混用 | 一般名詞用中文,`latent diffusion` 保留英文。見 m11 |
| telescope | 747 行 | 術語誤用(鏈鎖法則是連乘,不是望遠鏡相消),且該句內容本身有誤 | 刪除或改「連乘」。見 M11 |

### 檢查後確認一致、不用動

| 概念 | 用詞 | 說明 |
|---|---|---|
| zero-avoiding / zero-forcing、mode-covering / mode-seeking | 222、234、449、1505、`SpectrumAxis.vue:38-44` | 與領域慣例一致,英文保留、中文對照穩定 |
| 判別器 / critic | 判別器(全檔);1351 行明確指出 WGAN 的 D「變成不設上限的 critic」 | 正確且必要的區分,做得很好 |
| support | 全檔保留英文(220/274/302/1333 等) | 一致 |
| 散度 / 距離 / 度量 | 散度用於 KL、JSD;距離用於 Wasserstein(1338/1351);度量公理用於 √JSD(322) | **這三者的區分是正確的**,只是缺一個涵蓋三者的上位詞(見 M10) |
| 重建項 / 正則項 | 1028、1093、`VaeDefects.vue:12-13` | 一致 |
| posterior collapse / prior hole | 1101、1131、`VaeDefects.vue:16-19` | 一致,且 `prior hole` 的定義(逐樣本壓向先驗 ≠ 整團 q(z) 蓋滿先驗)是正確的 |
| 機率路徑 / 向量場 | 1434、1450、594 | 與 Lipman et al. 用語一致 |
| ELBO / 下界 / 間隙 | 1028–1038、1056、`ElboGap.vue` | 一致,且 1036 行的等式形式(間隙 = KL(q‖真後驗))是正確且常被教材略過的一條 |

---

## 附:未在本次審查範圍內、但順手記下的兩點

僅供作者參考,不列為意見:
- 599 行「一張 512×512 影像等於 26 萬次前向」:512² = 262,144,若是逐 sub-pixel 的 PixelCNN 則為 786,432。取決於作者想指哪一種,可能要加半句限定。(數學細節,屬 R3/R4 範圍)
- 1264 行「min-max 沒有位能函數 → 軌跡循環」是正確的領域共識,但目前無出處。若要補,可指 Mescheder et al., *The Numerics of GANs* (2017) 或 Balduzzi et al., *The Mechanics of n-Player Differentiable Games* (2018)。**(此兩條為記憶提供的推薦,未經本次線上查證,建議作者自行確認。)**
