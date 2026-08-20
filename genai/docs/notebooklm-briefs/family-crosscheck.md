# lecture-02 家族段落 × NotebookLM 交叉查核

查核日期:2026-08-16。每家族各問 2 題(AR notebook 72ac495f、VAE notebook 32796a50、GAN notebook c918505c、DPM/FM notebook 4fc29268)。引用格式:〔家族-Q1/Q2〕指該 notebook 的第 1/2 次回答。

---

## AR

### 矛盾/需修正

無。查核點全數獲確認〔AR-Q1〕:

- forward KL 沿序列分解、前綴期望取自資料分布 $p$、teacher forcing 是此分解的直接實作:確認。
- reverse KL 分解的前綴期望取自模型 $q$;訓練目標中沒有任何一項在模型自生前綴上取期望,exposure bias 即此結構後果:確認。

### 可補強

1. **誤差累積的量化形狀**:exposure bias 的誤差放大在 behavior cloning 分析中隨生成長度呈**平方級** $O(\epsilon H^2)$ 成長〔AR-Q2〕。目前投影片只說「誤差沿 t 累積」,補上量級可讓 demo(拉長生成長度看分離加速)有理論對照。
2. **scheduled sampling 與 Transformer 的相容性問題**:SS 假設逐步展開,Transformer 訓練是全前綴平行的,實作需 two-pass decoding 或 Parallel Scheduled Sampling〔AR-Q2〕。demo 講稿已提 SS「補丁緩解、結構不動」,此點可解釋為何補丁本身也有代價。
3. **contrastive decoding 的介面成本與護欄**:CD 在 logprob 介面上做 $s = (1{+}\beta)s^e - \beta s^a$,需 Adaptive Plausibility Constraint 防止放大 amateur 的負 logit 病理,且每步要兩個模型各一次前向、serving 成本加倍〔AR-Q2〕。第一堂已引 Li et al. 2023;「logprob 介面方法的推論成本」與 DDO 免 CFG 那頁的「每步兩次前向」是同一筆帳,可呼應。

---

## VAE

### 矛盾/需修正

1. **VQ-VAE 的動機描述偏窄**(timeline:「離散 latent,繞開 Gaussian likelihood 的平滑」)。Notebook 強調 VQ-VAE 的動機是資料本質離散(詞、音素、物件)與**繞開 posterior collapse**(離散 bottleneck + 事後學 prior,decoder 無法忽略 latent);其重建項實際上仍是 MSE 系,銳利化主要來自後續的 perceptual/對抗損失(即 VQ-GAN 那一步)〔VAE-Q1、VAE-Q2〕。建議改寫為「離散 latent:decoder 無法忽略,繞開 posterior collapse」,把「銳利」留給 VQ-GAN 條目。

其餘查核點確認〔VAE-Q1〕:Gaussian likelihood ≡ 負 MSE(固定 σ² 下)、MSE 最優解為條件均值、疊加 mode-covering 造成模糊、ELBO gap = KL(q_φ(z|x) ‖ p_θ(z|x)) 且 q 等於真後驗時取等號。

### 可補強

1. **Posterior collapse**:notebook 對 VAE 著墨最重的失效模式,投影片完全未提。機制是 inference lag:訓練早期 encoder 追不上移動中的真後驗,強大 decoder(尤其自迴歸型)乾脆忽略 z,之後 encoder 再無梯度、永久塌到 prior;修法有 cyclical KL annealing、free bits、aggressive encoder training〔VAE-Q2〕。demo 講稿的「β 過小:latent 空洞化」旁邊正好缺這一塊:ELBO 兩項失衡的另一種壞法。
2. **ELBO gap 的兩層分解**:approximation gap(diagonal Gaussian 變分族表達力不足,最佳化到完美也不歸零)+ amortization gap(共享 encoder 無法對每筆資料輸出最優後驗)〔VAE-Q1〕。投影片講稿說「gap 取決於 encoder 逼近程度」,這個二分讓它更精確。
3. **Prior holes**:aggregate posterior q(z) 呈狹窄流形,與 N(0,I) prior 不重合的高機率區是「洞」,prior 抽樣落入即解碼成糊;對應修法 VampPrior、latent 空間再訓一個 diffusion prior〔VAE-Q2〕。這正是 demo「環狀資料、prior 蓋不乾淨」的正式名稱,也直通 Latent Diffusion 的今日角色。

---

## GAN

### 矛盾/需修正

無。查核點全數獲確認〔GAN-Q1〕:

- 原始 minimax 在最優 D 處 = 2·JSD − 2log2;支撐集分離時 JSD 飽和於 log2、梯度歸零:確認。
- non-saturating(−log D)在最優 D 附近 = KL(p_g‖p_data) − 2·JSD(差一常數):確認,含兩項病理歸因(reverse KL 項 → mode-seeking/塌縮;−2JSD 項推離資料分布 → 推拉衝突、震盪不穩):確認。
- generator 損失中沒有資料項、訊號只經逐點判別器:確認,含「漏掉整個眾數不受罰」的結構論證。

### 可補強

1. **Minibatch discrimination 就是「幫逐點介面開分布欄位」的字面實作**:對 batch 內每對樣本算 closeness,總和 o(x_i)=Σ_j c(x_i,x_j) 直接接進 D 的輸入特徵;塌縮批次的 closeness 異常飆高,立即被抓〔GAN-Q2〕。講稿已點名 Salimans et al.,正文若加一句「修法 = 給介面加欄位」,與 mode collapse 頁的框架嚴絲合縫。
2. **WGAN 的一行對照例**:兩個平行分布相距 θ 時,KL = ∞、JSD 飽和為常數、而 W = |θ| 呈線性——梯度因此不消失〔GAN-Q2〕。比「支撐集分離時仍有梯度」更具體,適合放講稿。
3. **Unrolled GAN 的動力學圖像**:交替最佳化呈「剪刀石頭布」循環——G 過擬合當前 D、塌到單一安全眾數、D 追上後 G 跳到下一個眾數;unrolling 讓 G 對 k 步後的 D 最佳化,消掉 limit cycle〔GAN-Q2〕。可作為講稿中「更深層成因」的一句話版本。

---

## DPM / Flow Matching

### 矛盾/需修正

無矛盾;一處可更精確:

- timeline 寫 DDIM「抽樣改為確定性 ODE」——DDIM 原文的推導是**非 Markov 前向過程族**(與 DDPM 共享邊際 q(x_t|x_0),故已訓練的 DDPM 權重免重訓直接可用),η=0 得到確定性抽樣;「它是 probability flow ODE 的一階 Euler 離散化」是 Score-SDE 之後的詮釋〔DPM-Q1〕。講稿其實已說對,正文條目可改成「非 Markov 過程族,η=0 確定性抽樣,免重訓、步數大減」。

其餘查核點確認:DDPM 的 ε-parameterization 把 ELBO 化簡為加權 MSE 迴歸〔DPM-Q1〕;Score-SDE 統一離散步驟為連續 SDE、PF-ODE 經 instantaneous change-of-variables 給出精確 log-likelihood〔DPM-Q1〕;Flow Matching 訓練 simulation-free(CFM 在插值路徑上迴歸速度場,線性路徑目標即 x₁−x₀ 的 MSE)、源分布不必是 Gaussian(diffusion 因熱力學加噪推導被綁死在 Gaussian prior,CNF 系不受此限,橋接/配對任務直接受益)〔DPM-Q2〕;CFG 訓練時隨機丟條件、推論每步兩次前向、延遲加倍〔DPM-Q2〕。

### 可補強

1. **「免重訓」是 DDIM 最好的介面語言賣點**:同一組 DDPM 權重,換抽樣器即可 10–100 步出圖——sample() 介面的實作可以在不動訓練的前提下整個換掉〔DPM-Q1〕。這比「步數大減」更貼合本堂的介面框架。
2. **軌跡平直與少步抽樣的機制**:獨立配對的 (x₀,x₁) 產生交叉路徑,交叉處模型平均相互衝突的速度、軌跡彎曲,數值積分需要小步;Rectified Flow 的 reflow(用自己上一輪的非交叉配對重訓)把軌跡拉直,直線速度恆定、積分誤差驟降,1–10 步可行〔DPM-Q2〕。demo 講稿的「平直程度對應少步可行性」可補上這個因果鏈。
3. **CFG 在 rectified flow 上的 off-manifold 病理**:CFG 是線性外插,套在確定性 flow 上會把軌跡推離資料流形,造成過飽和與結構崩壞;修法(如 Rectified-CFG++)改外插為內插的 predictor-corrector〔DPM-Q2〕。與 DDO 免 guidance 那頁互補:兩條路線都在處理「CFG 的代價」。

---

## 總計

| 家族 | 矛盾/需修正 | 可補強 |
|---|---|---|
| AR | 0 | 3 |
| VAE | 1(VQ-VAE 動機) | 3 |
| GAN | 0 | 3 |
| DPM/FM | 0(1 處精確化) | 3 |
