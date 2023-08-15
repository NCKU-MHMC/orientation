---
theme: seriph
background: https://source.unsplash.com/collection/94734566/1920x1080
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
drawings:
  persist: false
transition: slide-left
title: Deep Learning 1
download: true
---

# Deep Learning 2
## Use Unlabeled Data Effectively

<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/NCKU-MHMC/orientation" target="_blank" alt="GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

---

# Overfitting, Generalization and Robustness

<div class="grid grid-cols-5">

<div class="col-span-2 pt-10">
<img class="m-auto" src="/img/overfitting.png" />
<div class="text-center">cite: <a href="https://www.analyticsvidhya.com/blog/2020/02/underfitting-overfitting-best-fitting-machine-learning/" target="_blank">Overfitting And Underfitting in Machine Learning</a></div>
</div>

<div v-click class="col-span-3">
<img class="m-auto pl-10" src="/img/overfitting.svg" />
<div class="text-center">cite: <a href="https://www.mathworks.com/discovery/overfitting.html" target="_blank">What Is Overfitting?</a></div>
</div>

</div>

<p v-click class="pl-20 text-2xl">

- 當模型複雜度超過訓練資料集複雜度時，就容易發生 Overfitting

- 當訓練資料集複雜度超過模型複雜度時，就容易發生 Underfitting

</p>

---

# Overfitting, Generalization and Robustness

<div class="grid grid-cols-2 text-2xl">

<p>

## 避免 Overfitting 的方法

- 使用更多訓練資料
- **資料擴增（Data Augmentation）**
- Dropout
- L1/L2 Regularization
- 降低模型複雜度

</p>

<p>

## 避免 Underfitting 的方法

- 大力出奇蹟  
  疊更多參數能解決一切
- 錢是萬能的  
  用更強的硬體訓練更大的模型

</p>

</div>

---

# Overfitting, Generalization and Robustness

<div class="grid grid-cols-5 text-2xl">

<p class="col-span-2">

## Data Augmentation

<img class="m-auto" src="/img/SpecAugment.png"/>
<div class="text-center text-base">cite: <a href="https://arxiv.org/abs/1904.08779" target="_blank">SpecAugment</a></div>

</p>

<div class="col-span-3">

<img class="w-3/5 m-auto" src="/img/sphx_glr_plot_transforms_020.png"/>
<div class="text-center text-base">cite: <a href="https://pytorch.org/vision/main/auto_examples/plot_transforms.html#photometric-transforms" target="_blank">Pytorch: Illustration of transforms</a></div>

<p class="pl-10 text-xl">

### Text Augmentation

- Replacing Words or Phrases with Their Synonyms
- Back Translation
- Text Generation
- Mixing-based Text Augmentation

</p>

</div>

</div>


---

# Overfitting, Generalization and Robustness

<div class="grid grid-cols-2 text-2xl">

<p>

## Generalization

- 模型應用到其他資料分佈的能力

> The classic approach towards the assessment of any machine learning model revolves around the evaluation of its generalizability i.e. its performance on unseen test scenarios.

</p>

<p>

## Robustness

- 模型對抗干擾的能力

> Evaluating such models on an available non-overlapping test set is popular, yet significantly limited in its ability to explore the model’s resilience to outliers and noisy data / labels (i.e. robustness).

</p>

</div>

<div class="text-center">cite: <a href="https://arxiv.org/abs/1804.00504" target="_blank">Generalizability vs. Robustness: Adversarial Examples for Medical Imaging
</a>,
<a href="https://datascience.stackexchange.com/questions/102931/robustness-vs-generalization" target="_blank">Robustness vs Generalization</a></div>

---

# Garbage In, Garbage Out


<div class="grid grid-cols-2">

<div>

<img class="w-3/4 m-auto" src="/img/GIGO.png"/>

<div class="text-center">cite: <a href="https://medium.com/marketingdatascience/%E7%9B%A1%E4%BF%A1%E8%B3%87%E6%96%99-%E4%B8%8D%E5%A6%82%E7%84%A1%E8%B3%87%E6%96%99-6f7551b0164b" target="_blank">盡信資料，不如無資料</a></div>

</div>

<p class="text-2xl">

- **資料稀缺**  
  <span v-click>使用預訓練（Pretrain）好的模型微調（fine-tune）</span>
- **缺乏標記**  
  <span v-click>使用 Self/Semi Supervised Learning</span>
- 標記錯誤 <span v-click>[Awesome-Noisy-Labels](https://github.com/songhwanjun/Awesome-Noisy-Labels)</span>
- 分佈不平衡  
  <span v-click>[Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)</span>

</p>

</div>

---

# Unlabeled Data

<div class="grid grid-cols-2 text-2xl text-justify">

  <p>

  ## Supervised

  - 資料需要有人工標記
  - 標記成本高昂、費時
  - 無法輕易擴展資料集

  </p>

  <p v-click=3 class="text-2xl">

  ## Semi-Supervised

  訓練資料（同時使用）：
  - 少量帶有人工標記的資料
  - 大量的未標記資料

  </p>

  <p v-click=1>

  ## Representation Learning

  - 任務目標：抽取良好的資料特徵
  - 可以是 Supervised or Unsupervised

  </p>

  <p v-click=2>

  ## Self-Supervised

  - 屬於 Unsupervised Learning 的一類
  - 從大量的未標記資料中學習特徵
  - 訓練好的模型應用在多個下游任務

  </p>

</div>

---

# Semi-Supervised

<img src="/img/SemiSL-survey.png" />
<div class="text-center">cite: <a href="https://ieeexplore.ieee.org/abstract/document/9941371?casa_token=etI0VkiYdfcAAAAA:cZ25qi2_yxSiYL4K_qrf2hYocsPd2Bzn2eeKeoGZi_Dp0m2s9e_CqZ3y8Jo1DIyU6JcmNN4bjWo" target="_blank">A Survey on Deep Semi-supervised Learning</a></div>

<myellipse v-click x="810" y="250" rx=35 ry=15 color="#c69" width="2" />

---

# Semi-Supervised
## Noisy Student Training

<img class="m-auto w-7/12" src="/img/Noisy-Student-Training.png" />
<div class="text-center">cite: <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.html" target="_blank">Self-training with Noisy Student improves ImageNet classification</a></div>

---

# Noisy Student Training

<span></span>

<p class="text-xl mr-5">

1. 在有人工標記的資料集 $\mathcal{D}$ 上訓練模型（Teacher）。
2. 在大量的未標記資料 $\mathcal{U}$ 上使用 Teacher Model 推斷標籤得到 $\mathcal{\tilde U}$。
3. 使用通過資料增強的 $\mathcal{D} \cup \mathcal{\tilde U}$ 訓練新的模型（Noisy Student）。
4. 回到步驟 2，並使用 Noisy Student 取代 Teacher。
</p>

<img v-click class="m-auto w-2/3" src="/img/ablation-study-of-noising.png" />
<div v-after class="text-center">cite: <a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.html" target="_blank">Self-training with Noisy Student improves ImageNet classification</a></div>

---

# Self-Supervised

<div class="grid grid-cols-2">

  <div>
    <img src="/img/SSL.png" />
    <div class="text-center">cite: <a href="https://arxiv.org/abs/2110.09327" target="_blank">Self-Supervised Representation Learning: Introduction, Advances and Challenges</a></div>
  </div>

  <p class="pl-10 text-2xl">

### <a href="https://superbbenchmark.org/leaderboard" target="_blank" class=text-3xl>Audio</a>
  - [CPC](https://arxiv.org/abs/1807.03748)
  - [Wav2Vec](https://arxiv.org/abs/1904.05862)/[2.0](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf)
  - [HuBERT](https://ieeexplore.ieee.org/abstract/document/9585401?casa_token=zBF2Y2pPmYcAAAAA:R0sxq526QDltxqKwGsaxqjwe_TTbKJc4DtWQMqw27K1crKg2Th1teVe_q6RnZs4N-03UkcSA--4)
### <a href="https://huggingface.co/spaces/mteb/leaderboard" class=text-3xl>Text</a>
  - [BERT](https://arxiv.org/abs/1810.04805)
### Image
  - [SimCLR](https://proceedings.mlr.press/v119/chen20j.html)
  - [BYOL](https://papers.nips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)
  - [DINOv1](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)/[v2](https://dinov2.metademolab.com/)

  </p>

</div>

---

# Self-Supervised

<img class="m-auto w-8/11" src="/img/pseudo-label-generation-processes.png" />
<div class="text-center">cite: <a href="https://ieeexplore.ieee.org/abstract/document/9770283?casa_token=BhPw-6_G1toAAAAA:FsA1L9ftQe4Z42g_Z3TDZo1f5EKfiUIUHNSU0El6s9h1yxHTB3anapNASH35kQaxQCKuzM9Uq6g" target="_blank">Self-Supervised Representation Learning: Introduction, Advances and Challenges</a></div>

---

# Self-Supervised
## SimCLR

<div class="grid grid-cols-2">

  <div class="text-2xl mr-5">
  
  - Contrastive Learning  
    <span class=text-xl>
    $$l(i,j)=-log\cfrac{\text{exp}(s_{i,j}/\tau)}{\sum_{k=1}^{2N}\mathbf{1}_{k\neq i}\text{exp}(s_{i,k}/\tau)}$$
    $s_{i,j}$ is the cos similarity between $z_i$ & $z_j$.
    </span>
    <arrow x1="430" y1="215" x2="430" y2="185" color="#c69" width="4" />
    <arrow x1="480" y1="240" x2="480" y2="270" color="#c69" width="4" />
  - 正樣本特徵越接近越好
  - 負樣本特徵差越多越好
  - $g(\cdot)$ 是輔助骨幹網路 $f(\cdot)$ 訓練的 projector，訓練完就會丟棄
    
  </div>

  <div>
    <img class="" src="/img/SimCLR.png" />
    <div class="text-justify">cite: <a href="https://proceedings.mlr.press/v119/chen20j.html" target="_blank">A Simple Framework for Contrastive Learning of Visual Representations</a></div>
  </div>

</div>

---

<div class="grid grid-cols-2">

  <div class="text-2xl mr-5">

  # Self-Supervised
  ## SimCLR
  - 使用資料擴增建構正樣本
  - 正樣本特徵**越接近**越好
  - 負樣本特徵**差越多**越好
  - MLP=$g(\cdot)$，輔助訓練的 projector
  - CNN=$f(\cdot)$，骨幹網路
    
  </div>

  <div>
    <img class="w-9/10 m-auto" src="/img/CL.gif" />
    <div class="text-justify">cite: <a href="https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html" target="_blank">Advancing Self-Supervised and Semi-Supervised Learning with SimCLR</a></div>
  </div>

</div>

---

# Self-Supervised Contrastive Learning

<span class="text-2xl">Issues and Solutions</span>

<p class="text-2xl">

  1. 需要大 Batch Size 才有足夠多的負樣本用來提昇性能
  <p v-click class="pl-10">

  - 重複使用 [Memory Bank](https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.html) 暫存的特徵當成正負樣本目標 
  - [Momentum-updated Encoder](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html) 減少計算正負樣本目標消耗的記憶體
  </p>

  2. 需要從無標記的資料中建構正負樣本
  <p v-click class="pl-10">

  - 設計無須使用負樣本的方法  
  e.g. [BYOL](https://papers.nips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html), [SimSiam](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html), [DINOv1](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html)/[v2](https://dinov2.metademolab.com/)
  </p>

</p>

---

# Contrastive Learning Without Negative Sample
<span class="text-2xl"></span>

## Why Do We Need Negative Samples?
<p class=text-2xl>

Contrastive Learning = 
正樣本特徵越接近越好 +
負樣本特徵差越多越好
<myline v-click x1="565" y1="177" x2="810" y2="177" color="#c69" width="4" />
<span v-after>$\to$ 只要模型輸出**常數**就能輕鬆達成，又稱為 Collapse (Trivial Solution)</span>

</p>

<div v-click>
  <img src="/img/SimSiam-collapse.png" />
  <div class="text-center">cite: <a href="https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html" target="_blank">Exploring Simple Siamese Representation Learning</a></div>
</div>

---

# Bootstrap Your Own Latent
<span class="text-2xl">SSL without Negative Sample</span>

<img class="w-9/10 m-auto" src="/img/BYOL.png" />
<div class="text-center">cite: <a href="https://papers.nips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html" target="_blank">Bootstrap Your Own Latent
A New Approach to Self-Supervised Learning</a></div>


<mycircle v-click x="700" y="350" r=40 color="#c69" width="4" />
<myellipse v-after x="760" y="150" rx=70 ry=40 color="#c69" width="4" />

---

# Bootstrap Your Own Latent
<span class="text-2xl">SSL without Negative Sample</span>

<img class="w-4/5 m-auto" src="/img/BYOL-vs-SimCLR.png" />
<div class="text-center">cite: <a href="https://papers.nips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html" target="_blank">Bootstrap Your Own Latent
A New Approach to Self-Supervised Learning</a></div>

---

# SSL Without Negative Sample

<img class="w-4/5 m-auto" src="/img/SimSiam-Comparisons.png" />
<div class="text-center">cite: <a href="https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html" target="_blank">Exploring Simple Siamese Representation Learning</a></div>

<p class="m-30 text-2xl">

- 實作簡單
- 性能優異
- 無需定義負樣本
- <span v-click class="color-[#f46]">若超參數沒設好，有發生 collapse 的風險</span>
</p>

---

# Masked AutoEncoder


<div class="z-1 absolute left-10 top-25 w-[500px]">
  <img class="" src="/img/BERT.png" />
  <div class="text-center">cite: <a href="https://arxiv.org/abs/1810.04805" target="_blank">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a></div>
</div>

<div class="absolute left-110 top-60 w-[500px]">
  <img class="" src="/img/MAE.png" />
  <div class="text-center">cite: <a href="https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper" target="_blank">Masked Autoencoders Are Scalable Vision Learners</a></div>
</div>

<p v-click class="absolute left-140 top-20 text-2xl">

- 穩定、容易訓練
- 不會 collapse
- 適合 Transformer 學習全域特徵
</p>

---

# Representation Learning


<div class="grid grid-cols-2">

  <div>
    <img src="/img/CLIP-1.svg" />
    <div class="">cite: <a href="https://openai.com/research/clip" target="_blank">Learning Transferable Visual Models From Natural Language Supervision</a></div>
  </div>

  <p class="pl-10 text-2xl">

  ### Text & Text
  - [Sentence-BERT](https://arxiv.org/abs/1908.10084)
  ### Text & Image
  - [CLIP](https://openai.com/research/clip)
  - [CoCa](https://openreview.net/forum?id=Ee277P3AYC)
  ### Multimodal
  - [Data2Vec](https://proceedings.mlr.press/v162/baevski22a.html)
  - [ImageBind](https://imagebind.metademolab.com/)

  </p>

</div>

---

# Survey
## Self-Supervised

- [Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)
- [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
- [Awesome Self-Supervised Learning](https://github.com/jason718/awesome-self-supervised-learning#nlp)
- [自監督式學習 Self-Supervised Learning for Computer Vision 之概述](https://medium.com/ching-i/%E8%87%AA%E7%9B%A3%E7%9D%A3%E5%BC%8F%E5%AD%B8%E7%BF%92-self-supervised-learning-for-computer-vision-%E4%B9%8B%E6%A6%82%E8%BF%B0-b0decf770abf)
- [A Survey on Contrastive Self-supervised Learning](https://arxiv.org/abs/2011.00362), 2020
- [Self-Supervised Representation Learning: Introduction, Advances and Challenges](https://ieeexplore.ieee.org/abstract/document/9770283?casa_token=BhPw-6_G1toAAAAA:FsA1L9ftQe4Z42g_Z3TDZo1f5EKfiUIUHNSU0El6s9h1yxHTB3anapNASH35kQaxQCKuzM9Uq6g), 2021
- [Self-Supervised Learning for Videos: A Survey](https://arxiv.org/abs/2207.00419), 2022
- [Survey on Self-Supervised Learning: Auxiliary Pretext Tasks and Contrastive Learning Methods in Imaging](https://www.mdpi.com/1099-4300/24/4/551), 2022
- [Self-Supervised Speech Representation Learning: A Review](https://arxiv.org/abs/2205.10643), 2022
- [A Survey of Self-Supervised Learning from Multiple Perspectives: Algorithms, Theory, Applications and Future Trends](https://arxiv.org/abs/2301.05712), 2023

---

# Survey
## Semi-Supervised

- [An Overview of Deep Semi-Supervised Learning](https://arxiv.org/abs/2006.05278), 2020
- [A Survey on Deep Semi-supervised Learning](https://arxiv.org/abs/2103.00550), 2021
