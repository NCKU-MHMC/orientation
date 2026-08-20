# R1 — Mathematical and Technical Correctness (Round 2, finished decks)

Reviewed files:
- `/media/8tsp/projects/orientation-2026/genai/lecture-01.md`
- `/media/8tsp/projects/orientation-2026/genai/lecture-02.md`
- against `/media/8tsp/projects/orientation-2026/genai/docs/Generative_Models_Intro_Two_Session_Outline_EN.md`
- plus formula-bearing components `components/GuidanceForm.vue`, `components/FamilyMatrix.vue` (both consistent with the decks; no findings there)

## Verified correct (no action)

KL definition and both asymmetry arguments; JSD definition, 0 ≤ JSD ≤ log 2, Jeffreys contrast, √JSD metric (Endres & Schindelin 2003); D\* = p/(p+q), value 2·JSD − 2 log 2 at the optimum, σ(log p/q) = p/(p+q) identity and its one-line check in the note; JSD = I(X;Z) note; temperature-as-degenerate-ratio-term algebra; RLHF objective, closed form π\* ∝ π_ref·exp(r/β), and the equivalence max E[r] − βKL(π‖π_ref) ⇔ min KL(π‖π\*) (note's derivation checks out); DPO reward inversion and L_DPO (matches Rafailov et al. 2023 eq. 7); DDO tempered optimum p_ref^{1−1/β}·p_data^{1/β} and its log form ("事實二" and the ②-table row are mutually consistent); sign table of the DDO gradient (qualitative direction correct); chain-rule KL decomposition in both directions with correct expectation subscripts; CE = H(p) + KL(p‖q) and the one-hot specialization; no-unbiased-entropy-estimator claim; BPB = (T/N_bytes)·log₂PPL; ELBO with gap KL(q_φ(z|x)‖p_θ(z|x)); NS-GAN divergence KL(p_g‖p_data) − 2·JSD (slide statement; see finding 4 for the note); CFG formula in the Ho & Salimans convention; 2^65536 ≈ 10^19728; speculative-decoding distribution-preservation claim; DDO numbers (EDM CIFAR-10 1.79→1.30, EDM2 ImageNet-64 1.58→0.97, EDM2 ImageNet-512 1.96→1.26, VAR-d30 ImageNet-256 4.74→1.79, <1% pretraining epochs per round) all match the reference set exactly, and the "halved inference cost" reading (dropping CFG's second forward pass) is sound; attributions checked and correct: Kim et al. (QA)² 2023, Arjovsky & Bottou 2017, Xie et al. 2022, Liu et al. 2024, Wang et al. 2023, Kirk et al. 2024, Kalai et al. 2025, Xiao/Kreis/Vahdat 2022, Zheng et al. 2025 (ICML), and the family-history years (DCGAN 2015, WGAN 2017, StyleGAN 2019, BigGAN 2019, β-VAE/VQ-VAE 2017, VQ-GAN 2021, DDPM/DDIM 2020, Score-SDE 2021, LDM 2022, FM 2023, CM 2023, InstructPix2Pix 2023).

## Findings

### 1. MUST — DDO optimum stated as p_θ = p_data, valid only at β = 1
- File: `lecture-01.md`, slide "DDO:用自己的 logprob 當判別器"
- Line: "BCE 的最優判別器是 $\sigma(\log(p_{\text{data}}/p_{\text{ref}}))$,對照兩式,最優解就是 $p_\theta=p_{\text{data}}$"
- Problem: the slide defines $d_\theta=\sigma(\beta\log(p_\theta/p_{\text{ref}}))$ two bullets earlier. Matching $\beta\log(p_\theta^*/p_{\text{ref}})=\log(p_{\text{data}}/p_{\text{ref}})$ gives $p_\theta^*\propto p_{\text{ref}}^{1-1/\beta}p_{\text{data}}^{1/\beta}$ — exactly what the later slide "DPO、DDO 與統一式" states. $p_\theta=p_{\text{data}}$ follows only when β = 1. As written, the deck asserts two different optima for the same loss.
- Fix: change the bullet to "對照兩式,$\beta=1$ 時最優解為 $p_\theta=p_{\text{data}}$;一般 $\beta$ 給出 $p_\theta^*\propto p_{\text{ref}}^{1-1/\beta}p_{\text{data}}^{1/\beta}$(後頁)". Alternatively note that under self-play (p_ref updated each round) the fixed point is p_data for any β — but then say that, not the unconditional claim.

### 2. MUST — DDO gradient formula is the β = 1 special case of the loss just defined
- File: `lecture-01.md`, slide "梯度在做什麼"
- Line: "$\nabla_\theta L=\int(1-d_\theta(x))(p_\theta(x)-p_{\text{data}}(x))\nabla_\theta\log p_\theta(x)\,dx$"
- Problem: for the BCE loss with $d_\theta=\sigma(\beta\log(p_\theta/p_{\text{ref}}))$, the gradient is $\nabla_\theta L=\beta\int[p_{\text{ref}}\,d_\theta-p_{\text{data}}(1-d_\theta)]\nabla_\theta\log p_\theta\,dx$. The displayed factorization uses $p_{\text{ref}}\,d_\theta=(1-d_\theta)\,p_\theta$, which requires $d_\theta/(1-d_\theta)=p_\theta/p_{\text{ref}}$, i.e. β = 1. With the β-scaled discriminator of the previous slide the formula is wrong as an identity (the qualitative raise/suppress table survives, since the sign of the bracket still tracks $p_\theta$ vs $p_{\text{data}}$ near the optimum only in the β = 1 form).
- Fix: add "(取 $\beta=1$)" to the equation line, or display the general bracket form and keep the sign table.

### 3. SHOULD — RLHF closed form misattributed to Ouyang et al. (2022) appendix
- File: `lecture-01.md`, slide "RLHF 的目標與閉式解"
- Line: "這個目標有閉式最優解(Ouyang et al., 2022 附錄;變分法一步)"
- Problem: the InstructGPT paper states the KL-regularized RL objective but does not derive $\pi^*\propto\pi_{\text{ref}}\exp(r/\beta)$. The standard written derivation is Rafailov et al. (2023), Appendix A.1 / eq. 4 (building on Peters & Schaal 2007; Korbak et al. 2022).
- Fix: cite "(推導見 Rafailov et al., 2023 附錄;經典結果可溯及 KL-control 文獻)" and keep Ouyang et al. as the citation for the objective itself (the ②-table row is fine).

### 4. SHOULD — speaker note overstates Arjovsky & Bottou Thm 2.5 as a value identity
- File: `lecture-02.md`, slide "理論上的 JSD,實務上的別種東西" (speaker note)
- Line: "non-saturating 的 −log D 目標在最優 D 附近等於 KL(p_g‖p_data) − 2JSD + 常數"
- Problem: Thm 2.5 is a gradient statement: $\mathbb E_{z}[-\nabla_\theta\log D^*(g_\theta(z))]=\nabla_\theta[\mathrm{KL}(p_g\|p_{\text{data}})-2\,\mathrm{JSD}]$. The objective's value is $\log 2+\mathrm{KL}(p_g\|p_{\text{data}})-\mathrm{KL}(p_g\|m)$, which differs from KL − 2JSD by $\mathrm{KL}(p_{\text{data}}\|m)$ — not a constant in θ. The slide's own wording ("對應的散度變成") is acceptable; the note's "等於 … + 常數" is not.
- Fix: reword the note to "non-saturating 目標在最優 D 下的**梯度**等於 KL(p_g‖p_data) − 2JSD 的梯度 (Thm 2.5)".

### 5. SHOULD — JSD interface deficit stated inconsistently across the two decks
- File: `lecture-02.md`, slide "三個散度,三種補法" (vs. `lecture-01.md`, slide "每個散度需要哪些介面")
- Line: deck 2 table, JSD row: "缺什麼:**兩個都缺**"; deck 1 says "至少缺一".
- Problem: under the course's working assumption (the model provides logprob), only $p_{\text{data}}$.logprob is missing for JSD; "兩個都缺" is true only after choosing a logprob-free generator (GAN), which is the conclusion the row is supposed to motivate, not its premise.
- Fix: in the deck-2 cell write "$p_{\text{data}}$.logprob 必缺(GAN 的 generator 連 $p_\theta$.logprob 也無)", or align both decks on "至少缺一" and let the GAN section state the second absence.

### 6. SHOULD — "嚴格符合此式" overstated for InstructPix2Pix
- File: `lecture-02.md`, slide "CFG 與 zero-shot 編輯"
- Line: "InstructPix2Pix 的雙 guidance scale(影像一個係數、指令一個係數)是嚴格符合此式的實例(Brooks et al., 2023)"
- Problem: IP2P's update is $\varepsilon(\varnothing,\varnothing)+s_I[\varepsilon(c_I,\varnothing)-\varepsilon(\varnothing,\varnothing)]+s_T[\varepsilon(c_I,c_T)-\varepsilon(c_I,\varnothing)]$ — a base plus **two** coefficient-weighted ratio terms. That is the natural multi-term generalization of the displayed single-ratio form, not strictly an instance of it.
- Fix: "是此式加到兩個比值項的推廣(每個條件各配一個係數)".

### 7. SHOULD — adversarial distillation loss credited under the Consistency Models entry
- File: `lecture-02.md`, slide "改進史(下):換空間、換目標、換步數"
- Line: timeline item "Consistency Models / 蒸餾 … 蒸餾損失常借對抗形式(Song et al.)"
- Problem: Song et al. (2023) Consistency Models use a consistency (self-distillation) loss with no adversarial term. The adversarial-loss distillation line is ADD (Sauer et al., 2023) and successors. As placed, the "(Song et al.)" tag attributes the adversarial form to the wrong paper.
- Fix: split the note: "Consistency Models(Song et al., 2023);對抗式蒸餾另見 ADD(Sauer et al., 2023)" — this also supports the GAN-section claim "蒸餾目標…對抗式蒸餾".

### 8. CONSIDER — reverse-KL interface row omits $p_\theta$.logprob
- Files: `lecture-01.md`, slide "每個散度需要哪些介面"; `lecture-02.md`, slide "三個散度,三種補法"
- Line: "reverse KL … $p_\theta$.sample + $p_{\text{data}}$.logprob"
- Problem: the integrand $\log p_\theta-\log p_{\text{data}}$ also calls $p_\theta$.logprob. Harmless under the session's working assumption, but the table purports to enumerate required interfaces, and the forward-KL row does list the model-side logprob.
- Fix: "$p_\theta$.sample + $p_\theta$.logprob + $p_{\text{data}}$.logprob"(或在表下註明模型側介面一律假設可得).

### 9. CONSIDER — ICL integral drops a conditional dependence
- File: `lecture-01.md`, slide "第 1 層.改變條件:prompt 即 conditioning"
- Line: "$p(y\mid\text{prompt})=\int p(y\mid\text{task})\,p(\text{task}\mid\text{prompt})\,d\,\text{task}$"
- Problem: as an identity this needs $y\perp\text{prompt}\mid\text{task}$; Xie et al.'s form keeps $p(y\mid\text{task},\text{prompt})$ inside the integral. The outline carries the same simplification, so this is a shared imprecision, but a mathematically literate student will notice.
- Fix: write $p(y\mid\text{task},\text{prompt})$ inside the integral, or add "(假設 task 給定後 y 與 prompt 條件獨立)" to the speaker note.

### 10. CONSIDER — CFG-for-LLM coefficient convention differs from Sanchez et al.'s γ
- File: `lecture-01.md`, slide "常見方法都是這條式子(上)"
- Line: CFG row — base $\log p(x\mid c)$, ratio $\log p(x\mid c)-\log p(x)$, coefficient $w$
- Problem: Sanchez et al. parameterize $\hat p\propto p(x\mid c)^\gamma p(x)^{1-\gamma}$, i.e. base $\log p(x)$ with coefficient γ; the deck's $w$ equals γ − 1. Correct as written (it is the Ho & Salimans convention), but students opening the cited paper will see a shifted coefficient.
- Fix: speaker-note one-liner: "Sanchez et al. 的 γ 對應此處 w + 1".

### 11. CONSIDER — β's necessity argued from the wrong quantity's magnitude
- File: `lecture-01.md`, slide "DDO:用自己的 logprob 當判別器"
- Line: "$\beta$ 是必要的縮放:$\log p_\theta$ 的量級可達 $10^3$,直接進 sigmoid 會使梯度消失"
- Problem: the sigmoid's input is the log **ratio** $\log(p_\theta/p_{\text{ref}})$, not $\log p_\theta$; the leading $10^3$-scale terms cancel between two similar models. The ratio does still grow with dimension (per-dimension discrepancies accumulate), so β is genuinely needed — but the stated reason cites the wrong scale.
- Fix: "log ratio 隨維度累積、量級可達數十至數百" (the outline carries the same phrasing; fix both or neither).

### 12. CONSIDER — JSD's interface requirement asserted before JSD is defined
- File: `lecture-01.md`, slides "每個散度需要哪些介面" and the divergence demo (both precede "JSD 的定義")
- Line: table row "JSD | 兩側 logprob | 至少缺一"
- Problem: the claim that JSD needs both logprobs rests on $m=(p+q)/2$, which appears only on the following slide. The ordering follows the outline, so this is a note, not a deviation; a half-sentence forward reference ("定義見下頁,分母含兩側密度") would close the gap.

### 13. CONSIDER — optimal discriminator omits the balanced-mixture hypothesis
- File: `lecture-01.md`, slide "JSD 的判別器讀法"
- Line: "設一個分類器判斷樣本來自 $p$ 還是 $q$。最優判別器有閉式解:$D^*(x)=\frac{p(x)}{p(x)+q(x)}$"
- Problem: $D^*=p/(p+q)$ assumes samples are drawn from $p$ and $q$ in equal proportion; otherwise the priors enter. One word fixes it: "以等量樣本訓練的分類器".

### 14. CONSIDER — Rectified Flow attributed only to Lipman et al.
- File: `lecture-02.md`, slide "改進史(下):換空間、換目標、換步數"
- Line: "Flow Matching / Rectified Flow … (Lipman et al.)"
- Problem: Rectified Flow is Liu et al. (2023, ICLR); Lipman et al. is Flow Matching. Add "Liu et al., 2023" or drop "Rectified Flow" from the item.

### 15. CONSIDER — scaling-laws timeline item lacks its citation
- File: `lecture-02.md`, slide "AR 的改進史"
- Line: "scaling laws, 2020 — 損失隨規模冪律下降,投資有可預測回報"
- Problem: the outline's production rule 3 requires experimental results to carry a paper; every other dated timeline item names its authors. Add "(Kaplan et al.)".

## Prerequisite-order check (remit item 3)

Deck 1's forward references (RLHF/DDO rows in the ② table before ④; DDO in "兩件已經在手上的事實") are sanctioned by the outline and resolved within the deck; the only substantive ordering gap is finding 12. Deck 2 builds strictly on deck-1 material; no violations found.

## Severity summary

- MUST: 2 (findings 1–2)
- SHOULD: 5 (findings 3–7)
- CONSIDER: 8 (findings 8–15)
