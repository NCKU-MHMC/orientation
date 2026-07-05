---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://cover.sli.dev
# some information about your slides (markdown enabled)
title: Transformer & LLM
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# open graph
seoMeta:
  # By default, Slidev will use ./og-image.png if it exists,
  # or generate one from the first slide if not found.
  ogImage: auto
  # ogImage: https://cover.sli.dev
---

# Transformer & LLM

---

# What is Large Language Model (LLM)?

- A **LARGE** number of parameters
- **LARGE** amounts of training data  

<div v-click>

<strong class="text-xl">Training Stages</strong> (cite: [4 Stages of Training LLMs from Scratch, Avi Chawla](https://www.dailydoseofds.com/p/4-stages-of-training-llms-from-scratch/))
<div class="grid grid-cols-2">
<img src="/assets/llm-1.gif" />
<img src="/assets/llm-2.gif" />
<img src="/assets/llm-3.gif" />
<img src="/assets/llm-4.gif" />
</div>
</div>

<SlideCurrentNo class="absolute bottom-4 right-8" />
<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

<!--
Here is another comment.
-->

---

# The Origin of LLM

[Attention is All You Need](https://arxiv.org/abs/1706.03762) ([slides 1](https://toonnyy8.github.io/PPT/Self-Attention/index.html#/1), [slides 2](https://toonnyy8.github.io/PPT/Attention-is-all-you-need/#/))

<iframe
  title="Inline Frame Example"
  class="m-auto"
  width="700"
  height="400"
  src="https://toonnyy8.github.io/PPT/Self-Attention/index.html#/1">
</iframe>

<SlideCurrentNo class="absolute bottom-4 right-8" />

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

# Model Architecture
靈魂拷問：「你為什這樣設計？有什麼理論根據嗎？」
<div class="text-xl">

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

||Receptive Field|Memory Usage|Inductive Bias|Parallelization|
|:-:|-|:-:|:-:|:-:|
|CNN|受模型架構限制| $O(L)$ | Y | Y |
|RNN|會遺失太遙遠的資訊| $O(L)$ | Y | N |
|TNN|與模型架構無關| $O(L^2)$ | N | Y |


</div>

<SlideCurrentNo class="absolute bottom-4 right-8" />
<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

# Model Architecture--BERT
抽取上下文的語意資訊

<div class="text-xl">

$$
\left\{
\begin{array}{cc}
 CNN \\
 RNN \\
 \red{Transformer}
\end{array}
\right\} + 
\left\{
\begin{array}{cc}
 \red{Encoder} \\
 Decoder
\end{array}
\right\} + 
\left\{
\begin{array}{cc}
 Causal \\
 \red{Non\text{-}causal}
\end{array}
\right\}
$$

<img class="w-3/4 m-auto" src="/assets/bert-2phase.jpg">

</div>

<SlideCurrentNo class="absolute bottom-4 right-8" />

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

# Model Architecture--T5
Encoder+Decoder

<div class="text-xl">

$$
\left\{
\begin{array}{cc}
 CNN \\
 RNN \\
 \red{Transformer}
\end{array}
\right\} + 
\left\{
\begin{array}{cc}
 \red{Encoder} \\
 \red{Decoder}
\end{array}
\right\} + 
\left\{
\begin{array}{cc}
 \red{Causal} \\
 \red{Non\text{-}causal}
\end{array}
\right\}
$$

<div class="grid grid-cols-2">
<img class="w-1/1 m-auto" src="/assets/T5_encoder-decoder_structure.svg">
<div>

## Pre-training Task
<img class="w-1/1 m-auto" src="/assets/t5-pretraining.png">
</div>
</div>

</div>

<SlideCurrentNo class="absolute bottom-4 right-8" />

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

# Model Architecture--GPT


<div class="text-xl">

$$
\left\{
\begin{array}{cc}
 CNN \\
 RNN \\
 \red{Transformer}
\end{array}
\right\} + 
\left\{
\begin{array}{cc}
 Encoder \\
 \red{Decoder}
\end{array}
\right\} + 
\left\{
\begin{array}{cc}
 \red{Causal} \\
 Non\text{-}causal
\end{array}
\right\}
$$

<div class="grid grid-cols-3">
<img class="col-span-2 w-1/1 m-auto" src="/assets/GPT.png">
<div>

## Pre-training Task
<img class="w-1/1 m-auto" src="/assets/ARLM.png">
</div>
</div>

</div>

<SlideCurrentNo class="absolute bottom-4 right-8" />
<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

# 大力出奇蹟

- GPT: [Improving Language Understanding by Generative Pre-Training]()
- GPT-2: [Language Models are **Unsupervised Multitask Learners**]()
- GPT-3: [Language Models are **Few-Shot Learners**]()

<br/>
<br/>
<br/>

<div class="text-4xl text-center">

[In-context Learning](https://docs.google.com/presentation/d/1enVhx1YWyTOAguzRf9ES29ya3LlPBfdN/edit?usp=drive_link&ouid=114530471182659404980&rtpof=true&sd=true)
</div>

<SlideCurrentNo class="absolute bottom-4 right-8" />
<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

# Issues of LLM

<div class="grid grid-cols-2">

<div>

## Core Models & Efficiency

- Efficient Architectures
  1. Linear Attention, State Space Models  
  e.g. [Mamba](https://arxiv.org/abs/2312.00752)
  2. [Test-Time Training](https://arxiv.org/abs/2407.04620)
- Long-Context Processing
  1. [Lost in the Middle](https://arxiv.org/abs/2307.03172)
  2. [Retrieval-Augmented Generation](https://docs.google.com/presentation/d/1bpW8QyRY5aMe1u9-m1P_bhOYVtqiFJqb/edit?usp=drive_link&ouid=114530471182659404980&rtpof=true&sd=true)
  3. [Length Extrapolation](https://aclanthology.org/2024.findings-emnlp.582/)
      - [Train Short, Test Long](https://arxiv.org/abs/2108.12409)

</div>

<div>

## Reliability & Ethics

- Hallucination
- Interpretability
- Bias & De-biasing
  1. Training Free Guidance e.g. [Linear Alignment](https://arxiv.org/abs/2401.11458)
  2. [Unlearning](https://arxiv.org/abs/2310.10683)

## Applications & Interaction
- Agent AI e.g. [ReAct](https://arxiv.org/abs/2210.03629), [ReWOO](https://arxiv.org/abs/2305.18323), [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- Domain Adaptation
- Emergent Abilities
  1. Thoughts e.g. CoT, [GoT](https://arxiv.org/abs/2308.09687), [XoT](https://arxiv.org/abs/2311.04254), [FoT](https://arxiv.org/abs/2412.09078)
  2. Reasoning


</div>
</div>

<SlideCurrentNo class="absolute bottom-4 right-8" />
<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
layout: end
---