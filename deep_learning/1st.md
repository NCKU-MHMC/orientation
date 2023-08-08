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

# Deep Learning 1

<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/NCKU-MHMC/orientation" target="_blank" alt="GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---

# Deep Learning <span v-after=2>& Gradient Descent</span> 

<div class="text-3xl">

$$\begin{split}f:X\longmapsto &\;Y\\
&\;\;\;\; {\mathcal{L_{\theta}}}\\
\hat{f}_{\theta}:X\longmapsto &\;\hat{Y}\end{split}$$

<arrow x1="555" y1="210" x2="555" y2="150" color="#c69" width="2" arrowSize="1" />
<arrow x1="555" y1="150" x2="555" y2="210" color="#c69" width="2" arrowSize="1" />
</div>

<div class="grid grid-cols-2 text-3xl">

<div>
<div v-click>

$$\mathcal{L}_{\theta}(x, y)=||\hat{f}_{\theta}(x) - y||$$

</div>
<div v-click>

$$\theta\leftarrow\theta-\gamma\nabla_{\theta} \mathcal{L}_{\theta}(x, y)$$

</div>
</div>

<p v-click>

- $\hat{f}$, $\theta$: Model & Parameters
- $\mathcal{L}$: Loss Function
- $\gamma$: Learning Rate

</p>

</div>

<div v-click class="text-4xl absolute top-35 right-20">

$\mathcal{L}$ 與 $\hat{f}$ 必須可微

</div>

<!--
DL 的 loss function   
就如同 SVM 的 hinge loss  
決策樹的information gain 與 gini index  
基因演算法的 fitness function
-->

---

# Learning Rate

<img class="w-9/10 m-auto" src="/img/learning_rate.png"/>

<div class="text-center">cite: <a href="https://cs231n.github.io/neural-networks-3/" target="_blank">Stanford cs231</a></div>

---

# Mini Batch Gradient Descent

<img src="/img/batch.png" />

<div class="absolute text-3xl right-13 bottom-35">

Batch Size

- 過小容易震盪難以收斂
- 過大需要耗費更多計算資源

</div>

<div class="text-center">cite: <a href="https://stackoverflow.com/a/62839342" target="_blank">Mini-batch performs poorly than Batch gradient descent?</a></div>

<!--
在原始 GD 中，會計算完所有樣本的 loss 後才計算梯度並更新參數，但在需要龐大計算量與資料量的 DL 中實在是非常區乏效率，因此使用了 Mini-batch 的概念，每次在訓練時會從資料集中隨機採樣出數量為 batch size 的資料來進行訓練。
-->

---

# Activation Function

<span class="text-2xl">Linear & Nonlinear</span>

<div class="text-3xl">

$$\begin{split}
\hat{f}_{\theta}(x)&=(h_n \circ \cdots \circ h_2 \circ h_1)(x)\\
h_i(x)&=\alpha(xW_i+b_i)\\
\theta&=[W_1,b_1,\ldots,W_n,b_n]
\end{split}$$
</div>

<div v-click class="text-2xl ml-50">

$\alpha$ is the Nonlinear Activation Function, e.g., 
- $ReLU(x)=max(x, 0)$
- $\sigma(x)=\cfrac{1}{1+e^{-x}}$

</div>

---

# Activation Function

<span class="text-2xl">Linear & Nonlinear</span>

<div class="text-3xl">

$$\begin{split}
\hat{f}_{\theta}(x)=&\;(h_n \circ \cdots \circ h_2 \circ h_1)(x)\\
=&\;xW_1 W_2 \cdots W_n \\
&\;b_1 W_2 \cdots W_n + \cdots +b_n\\
=&\;x W + b
\end{split}$$
</div>

<p v-click class="text-2xl ml-30">

當沒有使用激活函數時，無數線性層相疊等同個單線性層  
$\to$無法解決非線性問題。

</p>

---

# Gradient Issues
<span class="text-2xl">Vanishing and Exploding</span>

<div class="text-3xl">

$$\begin{split}
\mathcal{L}_{\theta}&=\mathcal{L}((h_n \circ \cdots \circ h_2 \circ h_1)(x), y)\\
\nabla_{W_1}\mathcal{L}&=\nabla_{h_n}\mathcal{L}\nabla_{h_{n-1}}h_n\cdots\nabla_{h_1} h_2\nabla_{W_1} h_1
\end{split}$$
</div>


<div v-click class="text-2xl">

## Gradient Vanishing

當 $|\nabla_{h_{i-1}}h_i| < 1$ 時，$\nabla_{W_1}\mathcal{L}$ 會依據層數指數縮小，梯度過小導致參數更新緩慢而難以訓練。
</div>


<div v-click class="text-2xl">

## Gradient Exploding

當 $|\nabla_{h_{i-1}}h_i| > 1$ 時，$\nabla_{W_1}\mathcal{L}$ 會依據層數指數上升，梯度過大使參數無法收斂。
</div>

---

# The Solution to the

<div class="grid grid-cols-2 text-2xl">

## Gradient Vanishing

## Gradient Exploding

</div>


<div class="grid grid-cols-2 text-2xl">

<div v-click>

- <a href="https://toonnyy8.github.io/PPT/resnet/index.html#/1" target="_blank">Skip Connection</a>
<img class="w-4/5 m-auto" src="/img/skip-connection.png"/>
<div class="text-center text-base">cite: <a href="https://tikz.net/skip-connection/" target="_blank">Skip Connection</a></div>

</div>

<div v-click>

- Gradient Clipping
<img class="w-4/5 m-auto" src="/img/gradient-clipping.webp"/>
<div class="text-center text-base">cite: <a href="https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem" target="_blank">Understanding Gradient Clipping</a></div>

</div>

</div>

<div class="text-2xl" v-click>

<div class="pl-80">

- Normalization

</div>

<img class="w-1/2 m-auto" src="/img/normalization.png"/>
<div class="text-center text-base">cite: <a href="https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html" target="_blank">Group Normalization</a></div>

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

<div class="grid grid-cols-2 text-2xl">

<p>

## Dropout
<br/>
<img class="w-9/10" src="/img/dropout.png" />
<div class="text-center text-base">cite: <a href="https://github.com/PetarV-/TikZ/tree/master/Dropout" target="_blank">Tikz/Dropout</a></div>

- 不會過度依賴某些特徵
- 加入雜訊增強穩健性

</p>

<p>

## L1/L2 Regularization

- $\mathcal{L}_\theta+\underbrace{\lambda||\theta||}_\text{L1}\; or\; \underbrace{\lambda||\theta||^2_2}_\text{L2}$

- L1: 將大多數貢獻不大的參數歸零，使模型變稀疏
- L2: 使模型不會過度依賴部份參數

- http://playground.tensorflow.org/

</p>

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
</a>, <br/>
<a href="https://datascience.stackexchange.com/questions/102931/robustness-vs-generalization" target="_blank">Robustness vs Generalization</a></div>

---

# Model Architecture
靈魂拷問：「你為什這樣設計？有什麼理論根據嗎？」
<p class="text-2xl">

$$
\left\{
\begin{array}{cc}
 CNN \\
 RNN \\
 Transformer
\end{array}
\right\} + 
\left\{
\begin{array}{cc}
 Encoder \\
 Decoder
\end{array}
\right\} + 
\left\{
\begin{array}{cc}
 Causal \\
 Non\text{-}causal
\end{array}
\right\}
$$

<div v-click>

||Receptive Field|Memory Usage|Inductive Bias|Parallelization|
|:-:|-|:-:|:-:|:-:|
|CNN|受模型架構限制| $O(L)$ | Y | Y |
|RNN|會遺失太遙遠的資訊| $O(L)$ | Y | N |
|TNN|與模型架構無關| $O(L^2)$ | N | Y |

</div>

</p>

---

# Model Architecture
<span class="text-2xl">Convolutional Neural Networks, CNN</span>

<div class="grid grid-cols-2">
<div>
<img src="/img/cnn.png.gif"/>
<div class="text-center">cite: <a href="https://user.phil.hhu.de/~petersen/SoSe17_Teamprojekt/AR/neuronalenetze.html" target="_blank">Convolutional Neural Networks am Beispiel eines selbstfahrenden Roboters 0.1 Dokumentation
</a></div>
</div>

<div v-click class="pl-10 text-2xl">

- Inductive Bias
  - 具備平移不變性  
  - 不同位置共享參數
  - 無須龐大資料就能有相對穩定的效能
- 可高度平行化
- Receptive Field 受架構設計所限
- https://poloclub.github.io/cnn-explainer/

</div>
</div>

---

# Model Architecture
<span class="text-2xl">Recurrent Neural Networks, RNN</span>

<div class="grid grid-cols-2">

<div v-click class="text-2xl">

- Inductive Bias
  - 不同位置共享參數
  - 無須龐大資料就能有相對穩定的效能
- 輸出與過去計算結果相關
  - 訓練時難以有效利用平行化加速

</div>

<div>
<img src="/img/rnn-text.gif"/>
<div class="">cite: <a href="https://research.aimultiple.com/rnn/" target="_blank">In-Depth Guide to Recurrent Neural Networks (RNNs) in 2023</a></div>
</div>

</div>

---

# Model Architecture
<span class="text-2xl"><a href="https://toonnyy8.github.io/PPT/Self-Attention/" target="_blank">Multihead Attention</a>, <a href="https://toonnyy8.github.io/PPT/Attention-is-all-you-need/" target="_blank">Transformer</a></span>

<div class="grid grid-cols-2">

<div>
<img class="w-4/5" src="/img/transformer.gif"/>
<div>cite: <a href="https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html" target="_blank">Transformer: A Novel Neural Network Architecture for Language Understanding</a></div>
</div>

<div v-click class="text-xl">

- w/o Inductive Bias
  - 不同位置共享參數
  - Receptive Field 依據訓練時給定的上限決定  
    <a href="https://spaces.ac.cn/tag/%E5%A4%96%E6%8E%A8/" target="_blank">about Length-Extrapolatable</a>
  - 需要額外加入位置資訊
  - 需要依靠龐大資料才能達成良好的性能
- 可高度平行化
-  $O(L^2)$ 的記憶體雜度

</div>

</div>


---

# Model Architecture

<div class="grid grid-cols-2 text-3xl">

<div>

$Data \overset{Encoder}{\underset{Decoder}{\rightleftharpoons}} Feature$

<img v-click class="w-19/20 m-auto" src="/img/diffusion.png"/>
<div v-after class="text-base text-center">cite: <a href="https://www.assemblyai.com/blog/how-dall-e-2-actually-works/" target="_blank">How DALL-E 2 Actually Works</a></div>


</div>

<div>

## Causal $p(y_t|x_{\le t})$, Autoregressive $p(y_t|y_{< t})$
<br/>

<img class="w-19/20 m-auto" v-click src="/img/wavenet.gif"/>
<div v-after class="text-base text-center">cite: <a href="https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio" target="_blank">WaveNet: A generative model for raw audio</a></div>

</div>

</div>

---

# Garbage In, Garbage Out


<div class="grid grid-cols-2">

<div>

<img class="w-3/4 m-auto" src="/img/GIGO.png"/>

<div class="text-center">cite: <a href="https://medium.com/marketingdatascience/%E7%9B%A1%E4%BF%A1%E8%B3%87%E6%96%99-%E4%B8%8D%E5%A6%82%E7%84%A1%E8%B3%87%E6%96%99-6f7551b0164b" target="_blank">盡信資料，不如無資料</a></div>

</div>

<p class="text-2xl">

- 資料稀缺  
  <span v-click>使用預訓練（Pretrain）好的模型微調（fine-tune）</span>
- 缺乏標記  
  <span v-click>使用 Self/Semi Supervised Learning</span>
- 標記錯誤 <span v-click>[Awesome-Noisy-Labels](https://github.com/songhwanjun/Awesome-Noisy-Labels)</span>
- 分佈不平衡  
  <span v-click>[Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)</span>

</p>

</div>

---

# Deep Learning <span v-after=2>& 調(煉)參(丹)</span> 

<div class="text-3xl">

$$\begin{split}f:X\longmapsto &\;Y\\
&\;\;\;\; {\mathcal{L_{\theta}}}\\
\hat{f}_{\theta}:X\longmapsto &\;\hat{Y}\end{split}$$

<arrow x1="555" y1="210" x2="555" y2="150" color="#c69" width="2" arrowSize="1" />
<arrow x1="555" y1="150" x2="555" y2="210" color="#c69" width="2" arrowSize="1" />
</div>

<div v-click=2 class="ext-3xl text-center">

## <a href="https://github.com/google-research/tuning_playbook/" target="_blank">Training Trick</a>

</div>

<br/>

<div v-click=1 class="grid grid-cols-2 text-3xl text-center">

<div>

## Data

<br/>

## Model

</div>

<div>

## Learning Algorithm

<br/>

## Loss Function

</div>

</div>
