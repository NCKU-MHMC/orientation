# EBM 筆記摘要(NotebookLM:Energy-Based Models)

來源:NotebookLM notebook `adcb3f1e-48e5-459a-b5d6-24de589272af`,2026-08-16 查詢。
各節末標註對應提問(Q1–Q7)。僅收錄筆記回答支持的內容。

## 介面實作

- 定義:p(x) = exp(−E(x)) / Z(Boltzmann / Gibbs 分佈)。低能量 = 高機率,高能量 = 低機率。Z 為 partition function,確保積分(或加總)為 1。
- `logprob(x)`:**有,但只到未正規化為止**。log p(x) = −E(x) − log Z;因 log Z 難解,只能輕鬆算出未正規化的 −E(x)。這已足夠做相對比較與 OOD 偵測。
- `sample()`:**原生沒有**。EBM 只有一個 energy function,沒有 feed-forward 生成程序;抽樣必須靠迭代式近似方法(gradient-based MCMC,典型為 Langevin dynamics),從隨機噪聲出發沿 −∇ₓE(x) 走並在每步注入噪聲。
- Z 難解的原因:離散情形要對 2^d 個組態加總(例:224×224×3 二值影像 d=150,528,需加總約 10^45000 個狀態,比宇宙年齡還久);連續情形是對整個 ℝ^D 的高維積分,E(x) 為深度網路時無 closed form。(Q1)

## 訓練目標與散度

- **MLE(正負相位梯度)**:∇θ log p(x) = −∇θE(x)(positive phase,壓低真實資料能量)+ E_{x'∼p_θ}[∇θE(x')](negative phase,拉高模型自生樣本能量)。−∇θ log Z 化為對模型分佈的期望,需 MCMC(SGLD)抽 negative samples;跑到平衡態代價高,實務上用 persistent chain / replay buffer。
- **Contrastive Divergence(Hinton)**:Markov chain 從真實資料點起跑,只走 k 步(常 k=1,CD-1)。目標是 CD_k = KL(p_data‖p_θ) − KL(p_k‖p_θ) 兩個 KL 之差;兩個 log Z 梯度完全對消,得到可行的梯度近似,不需平衡態抽樣。
- **Score Matching**:改建模 score s(x) = ∇ₓ log p(x) = −∇ₓE(x),Z 對 x 為常數故消失;經分部積分化為含 Hessian trace 的可解損失,完全免抽樣。對應 **Fisher divergence**。
- **Denoising Score Matching**:避開 Hessian trace 的高維成本;把資料加已知(高斯)噪聲,噪聲核的 score 有解析形式,訓練變成預測所加噪聲的監督任務(等同 denoising autoencoder)。對應噪聲平滑後資料分佈與模型間的 **Fisher divergence**。
- **Noise-Contrastive Estimation**:化為二元分類——判斷樣本來自真實資料或已知噪聲分佈 q(x);Z 當成可訓練的純量參數(如 bias)一起學。目標是 binary cross-entropy,噪聲樣本比例趨大時漸近等價於最小化 **KL divergence**。(Q2)

## 抽樣

- 方法:SGLD——由任意起點(隨機噪聲)出發,沿能量梯度下坡並每步注入高斯噪聲;噪聲防止立刻凍在最近的局部極小。∇ₓ log Z = 0,故只需未正規化能量梯度。
- **Mode mixing 問題**:目標分佈有分離良好的多個 mode 時,鏈易困在單一能量谷;跨越高能量障壁需指數級 mixing time,有限長度的鏈難以還原各 mode 的正確比例。
- **計算成本**:每步 Langevin 需一次完整 forward + backward;VAE 等一次 forward 即可生成,EBM 需數百至數千步(每次梯度更新可達 10,000 步)。在 ImageNet 規模上,單一 epoch 訓練時間會從數小時膨脹到數年。(Q3)

## 特徵性失效

- **訓練不穩定 / 發散**:MLE 梯度無約束,能量值可無界成長;訓練中會產生訓練資料沒有的 spurious local minima(OOD 區域的尖銳能量井),Langevin sampler 被困其中,能量地景發散。
- **與 normalization layer 不相容**:Batch Normalization 尤其不合——真實資料與高噪聲 MCMC 樣本的統計量劇烈波動,使最佳化震盪或不收斂。
- **評估困難**:Z 難解 → 無法直接算 exact likelihood;只能用 Annealed Importance Sampling 等昂貴且難調參、難擴展的近似法。
- Mode mixing(見「抽樣」節)也是特徵性失效之一。(Q3, Q4)

## 改進史(附論文與年份)

- **1982 Hopfield Networks**(John J. Hopfield):對稱連接網路,把物理自旋系統映射到聯想記憶,奠定「能量地景支配神經系統狀態」的概念。
- **1983/1985 Boltzmann Machines**(Ackley, Hinton, Sejnowski):加入隨機二值 hidden/visible units,用 Boltzmann 分佈做非監督特徵抽取;因平衡態抽樣困難,訓練極慢。
- **1986 RBM / Harmonium**(Paul Smolensky):二部圖限制(層內無連接),條件獨立使 Gibbs sampling 大幅簡化。
- **2002 Contrastive Divergence + 2006 疊層預訓練**(Hinton;Hinton & Salakhutdinov):CD 以短鏈繞過平衡態 MCMC;疊層 RBM 逐層預訓練深度網路,引爆深度學習復興。
- **2016 Deep Generative ConvNets**(Xie, Lu, Zhu, Wu):以現代 CNN 參數化能量函數,進入深度時代。
- **2019 Implicit Generation and Modeling**(Du & Mordatch):連續深度 EBM 擴展到 ImageNet;以 persistent sampler replay buffer 穩定 SGLD 訓練。
- **2020 JEM**(Grathwohl et al.):證明標準 softmax 分類器暗中就是 joint EBM;以 free energy 重詮釋 logits,同時做分類、生成與 OOD 偵測。
- **2005–2021 Score-based 橋接 diffusion**:Hyvärinen 2005 奠定 score matching;Song & Ermon 2019 多尺度 denoising score matching(另有 Sliced Score Matching),建立通往 reverse-time SDE diffusion models 的連續橋樑。(Q5)

## 與其他家族的關係

- **EBM → score → diffusion**:score = ∇ₓ log p(x) = −∇ₓE(x)(Z 的梯度為零)——score model 就是能量地景的空間梯度場(力場)。Diffusion(DDPM)以多尺度(annealed)DSM 學時間條件化的 s(x,t) ≈ −∇E_t(x_t);關係雙向,亦可把學好的 score 網路蒸餾回純量能量模型。
- **GAN discriminator 作為 energy**:GAN 可視為隱式模型,discriminator 扮演能量函數,學真實資料與合成樣本間的相對能量差。Adversarial training 亦隱式塑造能量地景(PGD attacker 相當於非收斂的 contrastive sampler)。
- **JEM(分類器即 EBM)**:E(x,y) = −f_θ(x)[y];對 y marginalize 得 E(x) = −LogSumExp(logits)——任何預訓練分類器可零改動、免重訓直接當 EBM 用。
- **Contrastive learning**:InfoNCE(CLIP、SimCLR、MoCo)概念上根植於 EBM;CLIP 可解讀為學跨模態(影像–文字)的 joint energy function。
- **MaxEnt RL / Soft Q-Learning**:policy π(a|s) ∝ exp(Q(s,a)/α),Q 值即負能量。(Q6)

## 應用

- **OOD / 異常偵測**:−E(x) 對齊 likelihood,克服 softmax overconfidence;LogSumExp energy score 改善視覺 OOD;語音方言辨識;胸腔 X 光異常偵測(免標註離群樣本);robust 分類器上做能量梯度下降產生 visual counterfactual explanations。
- **計算生物 / 化學**:蛋白質摺疊(Anfinsen 假說——原生結構為全域自由能極小;Folding@Home);machine-learned potentials 作為 conservative EBM 重現 potential energy surface,以 DFT 等級精度模擬水、IR 光譜、藥物–溶劑交互作用。
- **離散文字生成(Residual EBM)**:整句評分緩解 autoregressive 的 exposure bias;P_θ(x) ∝ P_LM(x)·exp(−E_θ(x)),以 LM 當噪聲分佈用 NCE 訓練,可讓 BERT/RoBERTa 類雙向模型做序列級生成。
- **傳統視覺與結構化輸出**:影像去噪 / 修復(最小化 joint energy E(X,Y))、分割與結構化標註、人臉辨識與姿態;CRF、Max-Margin Markov Networks、判別式 HMM 皆可寫成線性 EBM。(Q7)

## 適合投影片的關鍵句

- 「EBM 只給半個介面:logprob 有——但只到 −E(x),差一個算不出來的 log Z;sample 沒有——要用 MCMC 一步一步爬出來。」(Q1)
- 「MLE 梯度是雕刻:positive phase 壓低真實資料的能量,negative phase 拉高模型自己幻想樣本的能量。」(Q2)
- 「CD 的巧思:兩個 KL 相減,log Z 的梯度自己對消。」(Q2)
- 「Score matching 換一個微分方向:對 x 微分,Z 是常數,直接消失——代價是改最小化 Fisher divergence。」(Q2)
- 「VAE 生成一次 forward;EBM 生成要幾千次 forward+backward——ImageNet 上一個 epoch 從幾小時變幾年。」(Q3)
- 「訓練失敗的樣子:能量地景長出訓練資料裡不存在的假井,sampler 掉進去,模型發散。」(Q4)
- 「JEM 一句話:你的分類器一直都是 EBM,E(x) = −LogSumExp(logits)。」(Q6)
- 「score = −∇E:diffusion model 學的就是能量地景的力場——EBM 通往 diffusion 的橋。」(Q6)

## 筆記未涵蓋

- 筆記未涵蓋:EBM 與 normalizing flow 的直接對照、具體評測數字(FID 等)、CD 偏差(biased gradient)的理論分析細節。
