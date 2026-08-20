# Normalizing Flows 筆記摘要（NotebookLM 73273189）

依據 NotebookLM 筆記本問答整理；僅收錄筆記支持的主張。

## 介面實作

- 核心：以可逆可微變換 f 將簡單 base distribution（多元標準高斯）映射到資料分佈；change-of-variables formula 給出 log p(x) = log p_Z(f⁻¹(x)) + log|det J_{f⁻¹}(x)|。
- Jacobian determinant 項是「體積修正因子」：空間拉伸處密度變小、壓縮處變大，保證映射後仍是正規化的機率分佈（Normalizing 一詞由此而來）。
- 兩個硬條件：(1) bijection（一對一、可逆、輸入輸出同維度；數學上須為 diffeomorphism）；(2) tractable Jacobian determinant——任意 D×D 行列式為 O(D³)，NF 靠架構限制使 Jacobian 呈三角矩陣，行列式=對角線相乘，降為 O(D)。
- exact log-likelihood：逆向映射得 z，算 base 密度，累加各層 log-Jacobian——三步得到精確值，可直接以 MLE 梯度下降訓練（對比 VAE 只有 ELBO 下界、GAN 無法評估密度）。
- one-step sample：z ~ N(0, I) 一步採樣，前向傳播 x = f(z) 即完成（對比 autoregressive 逐維 O(D)、diffusion 數十至數百步迭代）。

## 架構與其成本不對稱

- NICE（additive coupling）：切成兩半，只變換一半，參數由另一半決定；逆向只需減法，det 恆為 1（volume-preserving）。
- RealNVP（affine coupling）：加入 scale 與 translation 網路 s(·)、t(·)，log|det J| = Σ s(x_a)_j，非體積保持；因每層只動一半維度，需插 permutation 層（反轉順序或棋盤格 mask）混合維度。
- Glow：把固定 permutation 換成可學習的 invertible 1x1 convolution（permutation 矩陣是其特例）；log|det J| = H·W·log|det K|，只需對 C×C 小矩陣求行列式，並以 LU 分解參數化 K = PL(U+diag(s)) 使 log|det K| = Σ log|s_i|。
- MAF vs IAF 成本不對稱：
  - MAF：x_i 依賴 x_{1:i-1}。logprob 快（給定 x 全維已知，MADE 一次並行前向）；sampling 慢（逐維串行 O(D) 次網路呼叫）。適合 density estimation / MLE 訓練。
  - IAF：參數依賴 u_{1:i-1}（隱空間噪聲）。sampling 快（u 一次抽完，並行生成）；logprob 慢（給定外部 x 須串行還原 u，O(D)）。適合即時生成、變分後驗。
  - Parallel WaveNet：MAF 當 teacher 高效 MLE 訓練，IAF 當 student 蒸餾——IAF 對「自己生成的樣本」算 logprob 是並行的（u 已知），因此蒸餾迴避了兩難，達成訓練快且生成快。

## 連續時間流與 Neural ODE

- Neural ODE（Chen et al., 2018）：離散層數→∞、步長→0 的極限，dz/dt = f(z(t), t; θ)；adjoint 法給出與深度無關的常數顯存。
- CNF：以 Neural ODE 建構的流。向量場 Lipschitz 連續時 ODE 解存在唯一（Picard–Lindelöf），軌跡不相交，可逆性天然成立，不再需要架構約束。
- Instantaneous change-of-variables：d log p(z(t))/dt = −Tr(∂f/∂z)。連續極限下，非線性的 determinant 被線性的 trace 取代，O(D³)→O(D)；trace 是線性算子，可任意疊網路，解鎖 free-form Jacobian。
- FFJORD（Grathwohl et al., 2018）：以 Hutchinson stochastic trace estimator + vector-Jacobian product 無偏估計 trace，不需顯式建 Jacobian，使高維 CNF 的 MLE 訓練可行。
- 痛點：MLE 訓練須反覆呼叫數值 ODE solver，慢且不穩——這是 Flow Matching 出現的直接動機。

## 特徵性失效與限制

- OOD 悖論：在複雜影像上 MLE 訓練好的 NF，會給無關的簡單影像（SVHN、MNIST 等）更高的 log-likelihood。機制：
  - forward KL 是 mass-covering，只要求資料處有高機率，不禁止別處也高；MLE 對「什麼是 OOD」漠不關心。
  - 低複雜度（低熵）影像經可逆映射後聚集在高斯原點附近的高密度區，log p(z) 極高。
  - coupling 的 mask 結構偏向擬合局部像素相關性（紋理）而非全局語義——學到的是通用像素壓縮器。
- 拓撲限制：diffeomorphism 不能改變拓撲；base 是 ℝ^d，但真實流形可能有週期（分子二面角在環面）或多峰不連通——強行映射會在邊界產生梯度爆炸的奇點、數值不穩。
- Lipschitz 限制：iResNet / Residual Flow 要求殘差的 Lipschitz 常數 < 1，每層「彎曲空間」的速度受限，塑造多峰分佈需堆大量層。
- 維度必須保持：不能用 pooling、strided convolution 降維，無法形成分層語義抽取；每層都維持全維特徵圖。
- 深度成本：單層表達力弱迫使極深堆疊（Glow 在 32×32 CIFAR-10 就要數十到上百個可逆塊）；理論上可逆網路可重建激活省顯存，實務上浮點誤差造成 numerical drift，大圖訓練仍極貴。
- 為何影像品質輸給 diffusion：
  1. 似然 vs 感知：像素空間 MLE 把容量浪費在人眼看不見的高頻噪聲上，忽略全局結構；diffusion 的 denoising score matching 目標近似 perceptual loss。
  2. 確定性軌跡脆弱：高維高斯質量集中在薄殼（concentration of measure），確定性路徑易受數值誤差偏軌；diffusion 的隨機去噪每步把粒子拉回流形，具糾錯能力。
  3. 潛在空間壓縮難配合：LDM 靠不可逆 VAE 壓縮 8 倍；NF 與凍結 VAE 拼接受限於隱空間不平滑，joint 訓練又被維度不變約束卡住，難以規模化。

## 改進史（附論文與年份）

| 年份 | 論文 | 貢獻 |
|---|---|---|
| 2014 | NICE（Dinh, Krueger, Bengio） | additive coupling，det 恆 1，可逆且零行列式開銷 |
| 2016 | RealNVP（Dinh, Sohl-Dickstein, Bengio） | affine coupling，非體積保持 + 多尺度架構 |
| 2016 | IAF（Kingma, Salimans et al.） | 逆自迴歸流，O(1) 並行採樣，適合變分推論 |
| 2017 | MAF（Papamakarios, Pavlakou, Murray） | 反轉 IAF，MADE 單次前向並行評估精確似然 |
| 2018 | Glow（Kingma, Dhariwal） | 可學習 invertible 1x1 conv + LU 分解 |
| 2018 | Neural ODE（Chen, Rubanova, Bettencourt, Duvenaud） | 連續時間極限，adjoint 法常數顯存 |
| 2018 | FFJORD（Grathwohl, Chen et al.） | Hutchinson trace 估計，free-form CNF |
| 2019 | Residual Flow（Chen, Behrmann, Duvenaud, Jacobsen） | Lipschitz < 1 的可逆殘差網路，解 iResNet 表達力與穩定性瓶頸 |
| 2022 | Flow Matching（Lipman, Chen et al.） | simulation-free 訓練 CNF，直接回歸條件向量場，拉直軌跡 |

## 與其他家族的關係（flow matching、diffusion）

- Flow Matching 本質是「免模擬訓練 CNF 的技術」：不跑 ODE 積分，直接對線性插值路徑 x_t = (1−t)x_0 + t·x_1 誘導的條件向量場 u_t = x_1 − x_0 做 MSE 回歸。
- 直線（optimal transport）路徑使採樣無彎曲累積誤差，一階 Euler 走 1–10 步即可生成，解決 Neural ODE 的採樣瓶頸。
- 與 diffusion 的等價性：高斯 flow matching 與 diffusion 數學等價（「同幣雙面」）。SDE 依 Fokker–Planck 對應到邊際分佈完全相同的確定性 probability-flow ODE，其中出現 score function ∇log p_t；FM 是對此 PF-ODE 更優雅的重參數化——diffusion 預測 ε 或 x_0（路徑受 noise schedule 影響而彎曲，需 50–1000 步），FM 直接預測速度 v = x_1 − x_0 並主動拉直軌跡。這是 Flux、Stable Diffusion 3 能 1–4 步生成的數學核心。
- 對 VAE：IAF 為 VAE 提供表達力更強的後驗近似（Rezende & Mohamed 2015 首先把 NF 引入變分推論，突破 mean-field 限制）。
- 對 autoregressive：MAF/IAF 是把機率鏈式法則引入流模型的分支；Parallel WaveNet 以 teacher-student 蒸餾連接兩者。

## 應用

- 語音：WaveGlow（Glow + WaveNet，mel-spectrogram → 波形，非自迴歸並行生成，GPU 上比實時快數十至上百倍）、Flowtron、Parallel WaveNet、zero-shot 語音合成與聲音創造。
- 科學計算：Boltzmann Generators（一步採樣分子系統平衡態、算自由能，繞過 rare event sampling）、PathFlow（隱空間插值找過渡路徑）、Smooth Normalizing Flows（C∞ 光滑，可算力、做 force matching）、lattice QCD 場論採樣。
- 宇宙學：RealNVP 學非高斯物質密度先驗做 MAP 去噪（參數限制精度提升約 2 倍）；emuflow 以 NF 擬合各巡天實驗邊緣化後驗，數分鐘完成跨實驗聯合推論。
- 變分推論：NF 後驗近似（Rezende & Mohamed 2015；IAF）。
- OOD / 異常檢測（改良版）：approximate mass（似然梯度範數）、語義特徵上訓練 NF、cycle-masking、流形距離結合密度。
- 醫學：LAMNr Flows（多視角共享隱空間，閉式條件推論、缺失影像插補、反事實分析）、conditional NF 生成合成健康紀錄解類別不平衡。

## 適合投影片的關鍵句

- 「NF 是唯一同時給出 exact log-likelihood 和 one-step sampling 的家族——代價是每一層都必須可逆且同維度。」
- 「Jacobian determinant 是體積修正因子：拉伸處密度變稀、壓縮處變密，總機率守恆。」
- 「三角 Jacobian 把 O(D³) 的行列式變成 O(D) 的對角線連乘——整個離散 NF 架構史就是在設計三角結構。」
- 「MAF 算似然快、採樣慢；IAF 採樣快、算似然慢——Parallel WaveNet 用蒸餾同時拿到兩邊的快。」
- 「連續極限下 determinant 退化成 trace：非線性算子變線性算子，架構約束隨之解除。」
- 「在影像上 MLE 訓練的 NF 會給沒看過的簡單影像更高的似然——精確似然不等於可靠的 OOD 偵測。」
- 「Flow Matching 不是新模型家族，而是 CNF 的免模擬訓練法；高斯 FM 與 diffusion 是同一條 probability-flow ODE 的兩種參數化。」
