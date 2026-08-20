# Introduction to generative models: outline for two 2-hour sessions

**Audience** New lab members who have completed an introductory deep learning course
**Lab context** The lab's projects centre on LLM applications (memory agents, emotional support dialogue, false premise detection, the relationship between confidence and accuracy, LLM-ASR), and prompt engineering is the dominant method

**Division of labour between the two sessions**

- **Session 1 (theory)**: how distance between distributions is defined and computed, and what operations change a distribution once it is fixed. Throughout, a distribution is treated as an abstract object, with no reference to any specific model.
- **Session 2 (instances)**: how to build a distribution that can actually be measured, adjusted and trained. This covers the concrete form of each family (AR / GAN / VAE / DPM-FM), the taxonomy, the trilemma, the history of improvements, and the application areas.

> **Note for the slide editor | Why this split**
> The split is tools first, objects second. Session 1 delivers a model-agnostic analysis kit (divergences, the unified guidance form, alignment objectives), and Session 2 shows how each family connects to that kit, or fails to. The benefit is that Session 1's conclusions transfer to any new model. The cost is that Session 1 is abstract, and every technique for manipulating a distribution silently assumes that certain capabilities exist. This outline handles that by writing the assumptions out as an interface contract (below), so hidden assumptions become stated axioms.

> **Note for the slide editor | Production rules**
> 1. **Slides and speaker notes carry the subject matter only.** Anything explaining why the material is ordered this way, or what pedagogical function a passage serves, stays in these notes and never appears on a page or in a script the students can see. Students came to learn generative models, not course design.
> 2. **Prose follows academic writing conventions**: no hype or promotional wording ("the most critical", "completely transforms", "remarkable"); no runs of dashes, recast with commas, colons, parentheses or separate sentences; no empty emphasis ("it is worth noting that"); the "not A but B" construction only where the contrast does real work, never as decoration; paragraph length varies naturally rather than falling into parallel structure.
> 3. **Every claim carries its support**: mathematical properties come with a derivation or a pointer to one; experimental results come with a paper; no unsourced quantitative adjectives ("substantially", "significantly" must map to a specific number).
> 4. Technical terms stay in English and are not forced into translation.

---

## 0. Where this course sits

This course is not a general introduction to generative models. It is the theoretical layer underneath the lab's current research. It is organised around divergences rather than around a taxonomy of models: a taxonomy yields a list, whereas a divergence yields consequences that can be derived.

### Contract for the whole course: a distribution is an object with two interfaces (stated at the start of Session 1, referenced throughout)

Define "a distribution $p$" as an object that offers at most two interfaces:

```
p.sample()      → draw one sample from p
p.logprob(x)    → return log p(x)
```

Session 1 assumes throughout that both interfaces are available, and asks only what can be done to a distribution once you have them. Which construction supplies which interface, and at what cost, is the subject of Session 2.

> **Note for the slide editor | Why the contract is needed**
> Without it, Session 1 repeatedly leans on a fact it has not yet explained: the unified guidance form, DPO's implicit reward and DDO's implicit discriminator all require $\log p_\theta(x)$, and why some models can compute this quantity while others cannot belongs to Session 2. The contract turns that from a hidden assumption into a stated axiom, and it leaves a natural transition between the sessions: how do you train a model with no logprob? The answer (GAN) comes in Session 2.

### Thesis (raised at the start of Session 1, returned to on the last slide of Session 2)

> **The vague hedging of a base model and the uniformity of an aligned model are two ends of the same spectrum.**

### The mode-covering to mode-seeking spectrum, used throughout

```
mode-covering ←─────────────────────────────────→ mode-seeking
broad coverage, over-smoothed                 sharp, may drop modes

training objective   forward KL · MLE   JSD · GAN    reverse KL · RLHF
decoding settings    temperature T > 1   T = 1       tighter top-p, higher CFG
weight fine-tuning   SFT · still MLE                 DPO / DDO · small β
```

All three rows are drawn in full during Session 1. Session 2 uses this spectrum to read off which divergence each family trains against.

### Opening mapping table (shown once in Session 1 ①, returned to in Session 2 ⑥)

| Lab topic | The probabilistic question behind it | Position on the spectrum |
|---|---|---|
| prompt engineering | choosing the conditioning variable $c$, manipulating $p(y \mid c)$ | changes the base term only, not the coefficient |
| memory agent | constructing the conditioning set; under long context, $p(\text{task} \mid \text{prompt})$ is diluted | as above |
| emotional support dialogue | generic comfort phrases vs. diversity collapse after alignment | toward the right, and constrained by $\beta$ |
| false premise detection | $p(y\mid x)$ is always well-defined, even when $p(x)\approx 0$ | a structural consequence of the left end |
| confidence vs. accuracy | calibration; predictive entropy vs. semantic entropy | the measuring instrument itself (Appendix D) |
| LLM-ASR | noisy channel $p(\text{text}\mid\text{audio})\propto p(\text{audio}\mid\text{text})\,p(\text{text})$ | a linear combination in log space |

### Teaching discipline (for the instructor)

A divergence has to be the answer to a question before it is a definition. After each property, connect it within two minutes to a concrete phenomenon the students have already seen. All examples in Session 1 stay LLM-native (temperature, top-p, DoLa, CFG for LLM, RLHF, DPO). Examples from the image setting (inpainting, image-to-image and other zero-shot edits) wait until the DPM part of Session 2.

> **Note for the slide editor | Why Session 1 examples stay LLM-native**
> Session 1 has no concrete model, so the examples are the students' only anchor. These students use LLMs daily and have no comparable experience with image diffusion models. Teaching CFG with an image example in Session 1 would leave them with nothing concrete to attach it to. Moving zero-shot edits to the DPM part of Session 2 puts them where a concrete model exists, and it shows the same guidance expression taking a different form in another family, which ties the two sessions together.

---

# Session 1 (120 min): measuring and manipulating a distribution

## ① What a distribution is, and why approximation requires a choice of divergence (36 min)

### Opening: thesis, definition of a generative model, interface contract (about 12 min)

- State the thesis: the vague hedging of a base model and the uniformity of an aligned model are two ends of the same spectrum. The theory in this session is the argument for that claim, which Session 2 returns to at the end
- A discriminative model learns $p(y\mid x)$, whose output space is small and closed. A generative model learns a $p_\theta(x)$ that can be sampled from (or its conditional version $p_\theta(x\mid c)$), with the goal of approximating the unknown $p_{\text{data}}$
- A high-dimensional $p(x)$ cannot be tabulated: the state space grows exponentially with dimension
- State the contract: a distribution is an object offering `sample()` and `logprob(x)`. Write it in the corner of the board and keep it there for the whole session
- Take stock of which interfaces each actor has: $p_{\text{data}}$ has only `sample()` (a dataset is a set of samples, with no `logprob`), while both of $p_\theta$'s interfaces are assumed available in this session

> **Note for the slide editor | Why "$p_{\text{data}}$ has no logprob" is raised here**
> This asymmetry runs through the entire course: data gives samples, not densities. Reverse KL being incomputable, GAN needing a discriminator, RLHF needing a reward model and $H(p)$ being inestimable are all the same fact appearing in different places. Establish it during the interface inventory, and every later occurrence only needs a pointer back.

### The asymmetry of KL (about 12 min)

$$\mathrm{KL}(p\|q)=\int p\log\frac{p}{q}$$

The integral is weighted by $p$, so it only measures the discrepancy where $p$ has mass.

- **Forward KL** $\mathrm{KL}(p_{\text{data}}\|p_\theta)$: the penalty is unbounded where $p>0$ and $q\to0$, so $q$ must cover the whole support of $p$. Hence **zero-avoiding / mode-covering**
- **Reverse KL** $\mathrm{KL}(p_\theta\|p_{\text{data}})$: the weight becomes $q$, and there is no penalty for ignoring the other modes of $p$. Hence **zero-forcing / mode-seeking**

Label each divergence with the interfaces it requires:

| Divergence | Interfaces required | Availability |
|---|---|---|
| forward KL | $p_{\text{data}}$.sample() + $p_\theta$.logprob() | both available |
| reverse KL | $p_\theta$.sample() + $p_{\text{data}}$.logprob() | the second is unavailable |
| JSD | logprob on both sides | at least one missing |

The session stops here and does not open up the question of how the missing quantities are supplied, or which model families result. That is answered in Session 2.

> **Note for the slide editor | Why only the left half of the computability table appears here**
> "Which interfaces a divergence requires" is a theoretical statement and belongs to Session 1. "Which families the workarounds produce" refers to concrete models and belongs to Session 2. Splitting the table has a second benefit: each family name is unpacked the moment it appears, so it never remains an undefined term.

> **Figure B-1**: a unimodal $q$ fitted to a bimodal $p$. Forward KL covers both modes and bridges the gap between them, JSD compromises, and reverse KL locks onto one mode.

### JSD: three points that need clarifying (about 8 min)

$$\mathrm{JSD}(p\|q)=\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m),\quad m=\tfrac{p+q}{2}$$

1. **JSD is not symmetrised KL.** The Jeffreys divergence $\mathrm{KL}(p\|q)+\mathrm{KL}(q\|p)$ inherits the infinities from both sides. JSD has the mixture $m$ in the denominator and is therefore bounded: $0\le\mathrm{JSD}\le\log 2$, and $\sqrt{\mathrm{JSD}}$ satisfies the metric axioms
2. **JSD sits between the two, but penalises more weakly.** It saturates as soon as the supports stop overlapping, and the gradient vanishes (a concrete instance of this pathology comes with GAN in Session 2)
3. **The discriminator reading of JSD.** Under the optimal discriminator $D^*=\frac{p}{p+q}$, the adversarial objective takes the value $2\,\mathrm{JSD}(p\|q)-2\log 2$. JSD is the ability of an optimal classifier to tell which distribution a sample came from, that is, $I(X;Z)$ with $Z\sim\text{Bernoulli}(1/2)$

Point 3 is used directly in the DDO part of ④ below.

> **Note for the slide editor | Why the discriminator reading of JSD is taught in Session 1, even though GAN is in Session 2**
> DDO (in ④ below) needs exactly one piece of prior knowledge: that the optimal discriminator is $\sigma(\log p/q)$. That is a mathematical property of JSD, not an engineering detail of GAN, so it sits comfortably in the theory session. DDO can then be explained in full without reference to any concrete model, and when GAN comes up in Session 2 the discriminator already has a theoretical home.

### Core message (about 4 min)

> **Choosing a discrepancy measure is choosing which kind of error you are prepared to accept.** No divergence is neutral.

| | Forward KL | JSD | Reverse KL |
|---|---|---|---|
| behaviour | mode-covering | in between, but saturates | mode-seeking |
| symmetric | no | yes | no |
| upper bound | none | $\log 2$ | none |
| failure mode | over-smoothed, hedging | vanishing gradient or oscillation | collapse, loss of diversity |

**Exercise** (outside class time): fit a single Gaussian to a 1D bimodal mixture, minimising each of the three divergences in turn, solving for $\mu,\sigma$ numerically.

## ② Manipulating a distribution I: the unified form of guided generation (22 min)

Once the divergence is chosen and the distribution is fixed, the target distribution is often not the model distribution itself. You may want it sharper, conditioned on something, safer or more diverse. Guided generation moves along this spectrum, and most guidance methods can be written in a single form:

$$\log p_{\text{guided}} \;=\; \log p_{\text{base}} \;+\; w\,\big(\log p_A - \log p_B\big) \quad(\text{then renormalize})$$

| Method | base | ratio term | coefficient | interfaces needed |
|---|---|---|---|---|
| temperature | $\log p$ | none | $1/T$ | logprob (per token) |
| CFG (LLM version) | $\log p(x\mid c)$ | $\log p(x\mid c)-\log p(x)$ | $w$ | logprob under two conditions |
| contrastive decoding / DoLa | $\log p_{\text{strong}}$ | $\log p_{\text{strong}}-\log p_{\text{weak}}$ | $\lambda$ | logprob of two models |
| Autoguidance | $\log p_\theta$ | $\log p_\theta-\log p_\phi$ | $w$ | logprob of two models |
| RLHF optimum | $\log \pi_{\text{ref}}$ | $r(y)$ | $1/\beta$ | logprob of the ref + reward |
| DDO optimum | $\log p_{\text{ref}}$ | $\log p_{\text{data}}-\log p_{\text{ref}}$ | $1/\beta$ | logprob of both |

top-k / top-p are the hard-truncation version of the same move: discontinuous, but removing probability mass from the tail all the same.

**Limits of applicability**: almost every row above needs the logprob interface. The scope of this framework depends on whether a model provides that interface, and Session 2 will show that one family does not.

Conclusions:

1. **The coefficient is a coordinate along the spectrum.** A larger $w$ or $1/\beta$ moves further toward mode-seeking, so all of these control parameters are the same parameter under different names
2. **Doing it at inference time and doing it at training time differ only in timing.** DDO in ④ below is an instance of doing it at training time
3. **Prompt engineering replaces the base term only and leaves the coefficient alone.** Problems of insufficient diversity or excessive sharpening therefore cannot be fixed by editing the prompt: the prompt does not sit in the coefficient's position

Point 3 bears directly on how the lab works day to day, so allow about 3 minutes of discussion.

> **Note for the slide editor | Why the table has an "interfaces needed" column**
> The table implicitly assumes logprob exists. An interface column (a) marks the scope of the framework, so students do not misapply it later to a model or black-box API with no logprob, (b) sets up GAN in Session 2, and (c) trains students to analyse a new method by asking which interfaces it calls.
> Also: zero-shot edits used to sit in this part and have been moved to the DPM part of Session 2 (see the note on teaching discipline for the reason).

## ③ Manipulating a distribution II: four layers of intervention at inference time (20 min)

| Layer | Point of intervention | Method | Interfaces needed |
|---|---|---|---|
| 1 | change the condition $c$ | prompt, few-shot, RAG, memory | sample only (conditional version) |
| 2 | change the sampling | temperature, top-k/top-p/min-p, beam | logprob (per token) |
| 3 | change the logits | logit bias, constrained decoding, contrastive decoding, DoLa, CFG for LLM | logprob (per token) |
| 4 | change how samples are aggregated | self-consistency, best-of-n, MBR, reranking | sample (logprob optional) |

### Layer 1 · changing the condition

- A prompt is conditioning. ICL can be read as implicit Bayesian inference: $p(y\mid\text{prompt})=\int p(y\mid\text{task})\,p(\text{task}\mid\text{prompt})\,d\,\text{task}$
  - This is the theoretical frame for memory agents: memory is not data storage but a choice of which evidence enters the posterior
  - More conditioning is not always better: irrelevant context flattens $p(\text{task}\mid\text{prompt})$ (lost-in-the-middle, position bias)
- The probabilistic meaning of RAG (explicit conditioning) and fine-tuning (conditioning amortised into the weights), and the failure mode of each

### Layer 2 · changing the sampling

- Temperature divides the logits by $T$ and directly adjusts the entropy of the distribution; top-p truncates and renormalises
- Sampling settings for an emotional support system: low $T$ is safe but monotonous, high $T$ is more varied but riskier. This is a design choice, not a default value

### Layer 3 · changing the logits

- constrained decoding / grammar: renormalise over the subset of legal tokens. This is the right way to get structured output, and it is more reliable than asking for a format in the prompt
- contrastive decoding / DoLa / CFG for LLM correspond to rows of the table in ②; this is where the operational details go

### Layer 4 · changing the aggregation

- best-of-n / MBR / reranking: re-estimate the distribution from samples
- LLM-ASR segment: $p(\text{text}\mid\text{audio})\propto p(\text{audio}\mid\text{text})\,p(\text{text})$ is the noisy channel model, and it is itself a linear combination in log space
  - n-best rescoring and LLM error correction belong to layer 4; feeding a speech encoder into an LLM belongs to layer 1 (acoustic representations as the condition)
  - The LM weight and insertion penalty of classical ASR are the manually tuned version of the coefficient $w$ in ②

> **Note for the slide editor | Why the four layers are also labelled with interfaces**
> This continues the analytical habit from ② and brings out one fact: layer 1 is the only layer that does not need logprob. That explains why prompt engineering works against any black-box API, and why it is the lab's main method at present. The closing remarks of Session 2 rest on this.

## ④ Manipulating a distribution III: intervening at the weights (SFT, RLHF/DPO, DDO) (32 min)

### SFT → RLHF / DPO (about 14 min)

- SFT redoes MLE on new data and remains at the mode-covering end of the spectrum
- **The RLHF objective** $\max_\pi \mathbb{E}[r(y)]-\beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$
  - The closed-form solution $\pi^*\propto\pi_{\text{ref}}\exp(r/\beta)$ is one of the rows in the table in ②
  - The optimisation is equivalent to minimising $\mathrm{KL}(\pi\|\pi^*)$, which is reverse KL
  - Restate the role of the reward model in interface terms: it is the surrogate for the absent $p_{\text{data}}$.logprob, and it supplies exactly the entry marked unavailable in the interface table in ①
  - "Alignment makes the model safer but less varied" is not a side effect but a mathematical consequence of the objective. For an emotional support system, safety and diversity are governed by the same $\beta$
- **The structural limit of a pointwise scorer**: a reward model scores pointwise and cannot express a distribution-level property such as "this distribution is too narrow". The only thing suppressing collapse is the $\beta$ term, and what that term constrains is the distance to the reference model, not diversity. LLM-as-judge has the same limit

> **Note for the slide editor | Why RLHF is retold in the language of surrogates**
> The sentence "the reward model is a surrogate for an unavailable logprob" links the interface table in ①, RLHF, and the GAN discriminator in Session 2 as three instances of one limitation, and it explains why the reward model and the discriminator share a structural limit: both are pointwise functions, and their interfaces carry no distribution-level information.

### Capstone example: DDO, building a discriminator out of your own logprob (about 18 min)

DDO needs only two ideas, neither of which refers to a concrete model:

1. **Build a discriminator from the model's own likelihood**
2. **Relative to MLE, add a term that suppresses the probability the model assigns to its own samples**

**Derivation of the first idea**: from the discriminator reading in ①, the optimal discriminator is $d^*(x)=\sigma\!\left(\log\dfrac{p_{\text{data}}}{p_{\theta_{\text{ref}}}}\right)$. Any distribution that provides a logprob interface can simply set

$$d_\theta(x):=\sigma\!\left(\log\dfrac{p_\theta(x)}{p_{\theta_{\text{ref}}}(x)}\right)$$

and train it with the standard BCE loss, whose optimum is $p_\theta=p_{\text{data}}$ (Zheng et al., 2025). This needs no extra discriminator network, no alternating training, and no backpropagation through the sampling process.

> **Figure B-3**: real samples + reference model samples → implicit discriminator $\sigma(\beta\log p_\theta/p_{\text{ref}})$ → discrimination loss; the dashed line is the self-play feedback

**The gradient form of the second idea**:

$$\nabla_\theta L=\int (1-d_\theta(x))\big(p_\theta(x)-p_{\text{data}}(x)\big)\nabla_\theta \log p_\theta(x)\,dx$$

It raises the density where $p_\theta<p_{\text{data}}$ and suppresses it where $p_\theta>p_{\text{data}}$. MLE, by contrast, only raises the density at the data points and has no mechanism for removing the model's excess probability mass. This is the precise sense of "moving away from your own samples": the loss contains both $\mathbb{E}_{p_{\text{data}}}[\cdot]$ (raising) and $\mathbb{E}_{p_{\text{ref}}}[\cdot]$ (suppressing).

**Where DDO sits on the spectrum**: it combines the mass-raising behaviour of forward KL with the mass-suppressing behaviour of reverse KL, so it acts at both ends of the spectrum simultaneously.

**Relation to DPO**:

| | DPO | DDO |
|---|---|---|
| implicit parameterisation | reward $=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ | discriminator $=\sigma(\beta\log\frac{p_\theta}{p_{\text{ref}}})$ |
| objective | preference learning | distribution alignment |
| data | paired human annotation | raw training data, no pairing needed |

**Relation to guidance**: introducing $\alpha,\beta$ (which is necessary, since $\log p_\theta$ can reach the order of $10^3$ and feeding that straight into a sigmoid kills the gradient), the optimum is $p_\theta^*\propto p_{\text{ref}}^{1-1/\beta}\,p_{\text{data}}^{1/\beta}$, which is the last row of the table in ②. Conclusion for the board: guidance sharpens at inference time, and DDO internalises the same operation into the weights.

**Summary of the experimental results** (specific families and numbers are left to Session 2): according to Zheng et al. (2025), several generative models that provide logprob improved on generation quality metrics without guidance after applying DDO, and each round of fine-tuning cost less than 1% of pretraining. Two observations bear directly on this session's argument: (1) continuing to train with the original MLE loss produced no improvement and in fact degraded the model, which indicates that the forward KL objective had reached its ceiling and that the problem was not one of hyperparameters; (2) some models had been relying on top-k / top-p sampling to lift their metrics, and this kind of truncation actually lowers the effective temperature, concealing a defect in the distribution rather than correcting it.

> **Note for the slide editor | Why DDO can go in the theory session, and why it serves as the closing example**
> DDO calls on only two things already established in ①, the logprob interface and the discriminator reading of JSD, and needs no architecture at all. In interface language it is an abstract result rather than a technique specific to one paper. As the closing example it does three jobs: (a) it uses all three modules of the session (the choice of divergence, the unified guidance form, and intervention at the weights); (b) it shows a method acting at both ends of the spectrum, which answers the thesis raised at the start; (c) its interface requirement (logprob) leaves one last transition question for Session 2, namely which families DDO applies to and which one it does not.

## ⑤ Wrap-up and homework (10 min)

- The three rows of the spectrum diagram are now complete. Return to the opening thesis: the two failure modes really are two ends of one continuum, and we now know they can be shifted at three separate levels
- State the two assumptions this session made deliberately:
  1. Every operation here assumed a $\pi_{\text{ref}}$ with both interfaces available. How that assumption gets satisfied is the subject of Session 2
  2. Two open questions: what surrogate does each method use for the logprob that reverse KL lacks, and how do you train a model that provides no logprob at all?
- **Homework**: pick one of your own research topics, write down its probabilistic form (what plays the role of $x$, $y$ and $c$), mark its position on the spectrum, and list the interfaces the methods you use call on

> **Note for the slide editor | Why the assumptions and open questions are stated explicitly**
> Teaching operations before construction carries one risk: students may think the construction step was simply left out. Saying that the order is deliberate removes that doubt. The question about interfaces in the homework tests whether the interface contract has been internalised.

---

# Session 2 (120 min): building distributions that satisfy the interfaces

## ① Bridge from Session 1 (8 min)

Recap: the whole of Session 1 assumed that `sample()` and `logprob(x)` exist. The question for this session is how to build, with a neural network, an object that actually provides them. Each way of building one corresponds to a model family. Each interface comes at a price, and how that price is paid is what separates the families.

> **Note for the slide editor | Why Session 2 is framed as "building an object that satisfies the interfaces" rather than "building a distribution"**
> Under the framing "building a distribution", GAN becomes an exception that does not fit the chapter: it produces no writable $p_\theta(x)$, only a sampler. In interface language, "an object that implements sample but not logprob" is a legitimate cell in the interface matrix, and GAN turns from an exception into part of the syllabus.

## ② Surrogates for the three divergences, and how the families form (15 min)

Complete the right half of the interface table from Session 1 ①:

| Divergence | Interfaces required | What is missing | Surrogate | Family formed |
|---|---|---|---|---|
| forward KL | $p_{\text{data}}$.sample + $p_\theta$.logprob | nothing | none needed | MLE → AR / Flow / VAE / DPM |
| reverse KL | $p_\theta$.sample + $p_{\text{data}}$.logprob | $p_{\text{data}}$.logprob | reward / energy surrogate | VI, RLHF |
| JSD | logprob on both sides | both | train a classifier as surrogate | GAN |

The table shows that GAN uses a discriminator because neither of the two logprobs JSD requires is obtainable, not out of design preference.

Return to Session 1 ④: the reward surrogate and the discriminator surrogate are two forms of the same thing, and both stand in for the unobtainable $p_{\text{data}}$.logprob. What distinguishes DDO is that it belongs to the forward KL row (its logprobs are complete) yet borrows the discriminator construction from the JSD row, which is why it can act at both ends of the spectrum.

> **Note for the slide editor | Why this table opens Session 2 instead of closing Session 1**
> The left half is theory (which interfaces each divergence requires) and the right half is each family's engineering response. Placed at the end of Session 1, the family names would still be undefined terms at that moment. Placed at the start of Session 2, each name is unpacked as soon as it is spoken and immediately acquires content.

## ③ The taxonomy diagram is an interface capability matrix (15 min)

Within the forward KL row, the families can be separated further by how exact their logprob is and how many steps their sample takes:

| Family | `logprob` | `sample` | Training objective (which divergence) |
|---|---|---|---|
| AR | exact (chain rule) | token by token, serial (slow) | forward KL / MLE |
| Normalizing Flow | exact (change of variable, needs invertibility and a tractable Jacobian) | one step | forward KL / MLE |
| VAE | lower bound only (ELBO) | one step | a lower bound on forward KL |
| DPM / FM | lower bound; exact via the probability flow ODE | many iterative steps (slow) | another decomposition of forward KL |
| GAN | none | one step | JSD (theory) / non-saturating (practice) |

Three things can be read straight off the table:

1. **The empty logprob cell for GAN explains three facts**: (a) training can only proceed through a discriminator surrogate (②); (b) the guidance, DPO and DDO methods from Session 1 do not apply to it, which is the concrete case of the scope limit stated in Session 1 ②; (c) why it samples fast, since it maintains no normalised density and trades that away for one-step generation
2. **AR and Flow have equally exact logprob and differ in the form of sample**: dimension by dimension (slow, expressive) versus one-step invertible (fast, constrained by invertibility)
3. **Where DPM sits**: it splits one-step generation into many sub-problems, each a simple regression. It keeps both logprob and quality, and pays the price in sampling steps

```
                generative object (approximating p_data)
                                │
              ┌─────────────────┼─────────────────┐
        forward KL family     JSD family      reverse KL family
        (logprob available)   (no logprob)    (p_data.logprob missing)
              │                    │                    │
   ┌──────┬───┴──────┐           GAN               RLHF / VI
 exact   bound   multi-step                    (covered in Session 1 ④)
   │        │          │
 AR / NF   VAE      DPM / FM
```

> **Note for the slide editor | Why the taxonomy diagram becomes an interface matrix**
> A conventional taxonomy (explicit/implicit density, tractable/approximate) supplies classification labels. An interface matrix supplies a usable test: faced with a new model later, a student checks two interfaces and can then work out which divergence it can be trained against, which techniques from Session 1 apply, and where it starts out on the trilemma. The sample column also covers the speed dimension of the trilemma, which keeps ④ short.

## ④ The generative learning trilemma (12 min)

Three competing goals that no family currently satisfies at once: **sample quality / mode coverage (diversity) / sampling speed** (Xiao, Kreis & Vahdat, 2022).

```
              sample quality
                    ▲
                   ╱ ╲
                  ╱DPM╲       GAN: fast, high quality, poor diversity
                 ╱(slow)╲     VAE: fast, good diversity, over-smoothed samples
                ╱───────╲     DPM: good quality and diversity, slow sampling
               ╱ VAE  GAN ╲
              ╱─────────────╲
        sampling speed   mode coverage
```

- The "quality vs. diversity" edge is a restatement of the Session 1 spectrum
- The "speed" vertex corresponds to the sample column of the matrix in ③
- The history of each family is mostly a series of attempts to approach the third vertex without giving up the two it already has. This is the common thread through the improvement histories in ⑤

> **Note for the slide editor | Why the trilemma precedes the family-by-family survey**
> Placed first, it supplies an analytical frame: as students hear each family's improvement history, they can organise it by asking which vertex a step moves toward and what it costs. It also extends the one-dimensional continuum from Session 1 into a two-dimensional triangle, showing that Session 1's theory is one edge of this figure.

## ⑤ Family-by-family survey: instances, improvements and applications (60 min)

> **Note for the slide editor | The shared structure of the survey**
> Every family is presented in the same four parts: (1) how it implements the interfaces; (2) its characteristic failure mode, which follows from its training divergence and the cost of its interfaces; (3) its improvement history, that is, which vertex of the trilemma it moves toward; (4) its applications, that is, the tasks its interface combination suits. A fixed structure makes cross-family comparison easy, and it lets the claim that the failure mode follows from the training divergence be checked once per family.

### AR / LLM (20 min)

**(1) Interface implementation**: decompose by the chain rule, $\log p(x)=\sum_t \log p(x_t \mid x_{<t})$. logprob is exact and cheap to compute; sample is forced to be serial.

**(2) Characteristic failure mode and related details**

The relation between CE and KL: $H(p,q)=H(p)+\mathrm{KL}(p\|q)$. For classification the target is one-hot and $H(p)=0$, so CE equals forward KL. Once the target softens (label smoothing, distillation) the two come apart.

- **$H(p)$ cannot be estimated**: another consequence of $p_{\text{data}}$.logprob being unavailable. Four practical routes: compare differences on the same data (which makes it a shared constant), normalise against a reference model (the quantity DDO used in Session 1), use synthetic data with known entropy, or sidestep likelihood entirely (MAUVE, MMD, downstream metrics)
- **Comparing across tokenizers → BPB**: $\mathrm{BPB}=\frac{T}{N_{\text{bytes}}}\cdot\log_2 \mathrm{PPL}_{\text{token}}$

The chain rule decomposition and teacher forcing:

$$\mathrm{KL}(p\|q)=\sum_t \mathbb{E}_{x_{<t}\sim p}\Big[\mathrm{KL}\big(p(\cdot\mid x_{<t})\,\big\|\,q(\cdot\mid x_{<t})\big)\Big]$$

Note the subscript on the expectation: $x_{<t}$ is drawn from $p$, not from $q$. Three things follow:

1. Teacher forcing is not a training trick but a direct implementation of the forward KL decomposition
2. Exposure bias is in the same line: at inference the prefix comes from $q$, but the training objective never measures prefixes the model generated itself. This is one mechanism behind drift in long memory agent conversations
3. Changing the subscript from $p$ to $q$ gives the decomposition of reverse KL. The suppressing term of DDO in Session 1 fills exactly this gap

> **Figure B-2**: the training/inference trajectory divergence diagram. Session 1 ④ raised the problem, and this segment gives the full picture.

This also explains the background to false premise detection: a model trained with forward KL has no structural option to refuse, unless post-training adds one. That $p(y\mid x)$ is well-defined and that $x$ deserves an answer are two different things.

**(3) Improvement history**: n-gram → RNN/LSTM → attention → Transformer (parallel training, which made scaling possible) → scaling law → instruction tuning / RLHF (shifting the model from pure forward KL toward the mode-seeking end of the spectrum, see Session 1 ④). On the speed side: speculative decoding and multi-token prediction.

**(4) Applications**: dialogue, code, agents, and the $p(\text{text})$ term in LLM-ASR.

### VAE (10 min)

**(1) Interface implementation**: introduce a latent $z$; logprob is available only as a lower bound (ELBO); sample is one forward pass of the decoder.

**(2) Characteristic failure mode**: over-smoothed, blurred samples. The cause is the combination of a Gaussian likelihood (equivalent to MSE) with a mode-covering objective. The notation for latent variables and marginalisation carries over elsewhere: the implicit Bayesian reading of ICL in Session 1 uses the same language.

**Demo**: `vae-2d-interactive.html`, showing the over-smoothing caused by mode covering, the failure to cover a ring topology cleanly, and the two ways things break when β is too large or too small (3 min).

**(3) Improvement history**: vanilla VAE → β-VAE (adjust the coefficient on the KL term to buy disentanglement) → VQ-VAE (discrete latents, avoiding the over-smoothing induced by a Gaussian likelihood) → VQ-GAN (add an adversarial loss for sharper reconstruction) → its main role today, as a compressor for diffusion models, as in Stable Diffusion running diffusion in a VAE latent space (returned to in the DPM segment).

**(4) Applications**: representation learning, anomaly detection, and latent space infrastructure for other generative models.

### GAN (13 min)

**(1) Interface implementation**: sample only. A one-step map $G(z)$, no logprob, so training can only proxy JSD through a discriminator (②), and none of the logprob-based methods from Session 1 apply.

**(2) Characteristic failure mode**: JSD is the theoretical correspondence, not what is used in practice. The value $2\,\mathrm{JSD}-2\log 2$ holds only under the optimal discriminator, and that is exactly where the gradient vanishes. With the non-saturating loss instead, the corresponding divergence is (Arjovsky & Bottou, 2017)

$$\mathrm{KL}(p_g \| p_{\text{data}}) - 2\,\mathrm{JSD}(p_g \| p_{\text{data}})$$

The first term is reverse KL, which is where mode seeking comes from. The second carries a minus sign and is the source of unstable training.

| | original minimax | non-saturating |
|---|---|---|
| corresponding divergence | JSD | reverse KL $-$ 2·JSD |
| pathology | vanishing gradient | mode-seeking, prone to collapse |

Each loss comes with its own failure mode: replacing the loss to remove the vanishing gradient also gives up the mode-covering property.

Mode collapse has causes at four levels, and the class should cover at least the first: the generator's loss contains no data term, and $D$ judges pointwise. Coverage is a distribution-level property, and the discriminator's interface has no field for it. This structural limit is the same one the reward model had in Session 1, and the class should point out the isomorphism.

**Demo**: `gan-2d-interactive.html`, trained to collapse with the discriminator landscape overlaid (5 min). What to watch: the missed mode has a high value on the discriminator landscape, but the generator never receives that information, and nothing on the screen signals that a mode has been missed.

Return to DDO from Session 1: the prototype for "building a discriminator out of your own logprob" is the $d^*=p/(p+q)$ here. DDO trains no separate discriminator network and instead places that role inside the likelihood ratio, but only if the model provides logprob, so DDO does not apply to GAN. The transfer runs one way: GAN's discriminator idea moves into the logprob families, not the reverse.

**(3) Improvement history**: DCGAN (a stable convolutional architecture) → conditional GAN → WGAN (Earth Mover distance instead, to address JSD saturation) → StyleGAN (quality and controllability) → BigGAN (scale) → a shift in recent years: trained from scratch less often, and used more often as a distillation target, compressing a multi-step model into one step.

**(4) Applications**: low-latency and real-time settings, super-resolution, style and voice conversion, and distillation for faster diffusion models.

### DPM / Flow Matching (17 min)

**(1) Interface implementation**: forward KL decomposed along the noise scale (compare AR's decomposition along the sequence). Each step is a simple regression, and training is about as stable as AR. logprob is available as a lower bound and exactly via the probability flow ODE; sample takes many steps.

**(2) Characteristic failure mode**: slow sampling. Its advantages in quality and diversity are paid for almost entirely in the number of sampling steps (see the matrix in ③ and the triangle in ④).

**(3) Improvement history**:

- **DDPM** (Ho et al., 2020): discrete-time step-by-step denoising, needing on the order of a thousand sampling steps
- **Score-based SDE** (Song et al., 2021): unifies the discrete steps into a continuous-time stochastic differential equation
- **DDIM** (Song et al., 2020): sampling becomes a deterministic ODE, cutting the step count sharply
- **Classifier-free guidance** (Ho & Salimans, 2022): the original form of the unified expression from Session 1 ② within this family. Zero-shot edits (inpainting, image-to-image) belong here: set $p_A$ to the distribution conditioned on the original image, set $p_B$ to the unconditional distribution, and let $w$ control the extent of the change, which is the concrete form of the zero-shot edit row in the Session 1 ② table
- **Latent Diffusion** (Rombach et al., 2022): diffusion in a VAE latent space (returning to the VAE segment)
- **Flow Matching / Rectified Flow** (Lipman et al., 2023): learn the ODE vector field directly, with training as simulation-free regression and no requirement that the source distribution be Gaussian
- **Consistency Models / distillation** (Song et al., 2023): distil the multi-step ODE integration into one step, with the distillation loss often borrowing the adversarial form from GAN

**Demo**: `flow-matching-2d-interactive.html`, showing another decomposition of the same divergence and a source distribution that need not be Gaussian (3 min).

Compare the DDO results from Session 1 ④: quality improves without guidance and inference cost is unchanged. That is a different route to speed, changing the weights rather than the step count.

**(4) Applications**: image, video and audio generation, molecular design, and motion generation.

> **Note for the slide editor | Reasoning behind the time allocation in the survey (AR 20 / VAE 10 / GAN 13 / DPM 17)**
> AR gets the most time: it is the only family the lab touches in practice, and $H(p)$, BPB, exposure bias and false premises all connect directly to current projects. DPM comes second: its history is the fullest worked example of the trilemma, and it has to cover the zero-shot edits moved over from Session 1. GAN gets slightly more than VAE because it carries the explanation of the missing interface cell and its demo. VAE gets the least, while keeping its modern role as the compressor for diffusion.

## ⑥ Placing the lab's work, and choosing problems (10 min)

Replay the opening mapping table and locate the lab's six topics in three coordinate systems:

1. **The Session 1 spectrum**: which divergence was chosen, and where the work currently sits
2. **The interface matrix**: which interfaces the method calls, and which ones the model provides
3. **The trilemma triangle**: which vertex is being sacrificed

Closing remarks: the lab's current methods mostly sit at layer 1 and use the sample interface only. Layers 2 to 4 additionally require nothing but the logprob interface, every model the lab uses provides it, and the methods at those three layers need no extra training resources.

> **Note for the slide editor | How the closing remarks are phrased**
> Phrasing the closing remarks in interface language gives "the room to grow is at layers 2 to 4" a specific basis: it follows directly from the interface inventory, which rests on the contract stated at the start of the course, rather than standing as encouragement.

---

# Demo plan

## Existing 2D demos

| File | Purpose | Placement and time |
|---|---|---|
| `gan-2d-interactive.html` | mode collapse and the discriminator landscape | Session 2 ⑤, GAN segment, 5 min |
| `vae-2d-interactive.html` | over-smoothing from mode covering; topology; failure at extreme β | Session 2 ⑤, VAE segment, 3 min |
| `flow-matching-2d-interactive.html` | another decomposition of the same divergence | Session 2 ⑤, DPM segment, 3 min |

## Suggested new LLM-native demos (in priority order)

1. **Token probability browser**: show top-k probabilities and entropy token by token, with temperature and top-p sliders. For Session 1 ②③, to show that the coefficient is a coordinate along the spectrum and that the logprob interface really exists
2. **Calibration scatter plot**: reliability diagram, ECE and temperature scaling. Directly related to the lab's projects, and used with Appendix D
3. **False premise / semantic entropy**: compare prompt BPB for normal and false-premise prompts; draw n samples for one question, cluster them by meaning, and compute semantic entropy. Shared between the AR segment of Session 2 ⑤ and Appendix D

---

# References

## Divergences and probability background (Session 1 ①)

- Stanford CS236, Lectures 1 and 2: <https://deepgenerativemodels.github.io/notes/>
- Bishop & Bishop, *Deep Learning: Foundations and Concepts*: <https://www.bishopbook.com>
- Arjovsky & Bottou (2017), *Towards Principled Methods for Training GANs*

## Guidance and decoding (Session 1 ②③)

- Holtzman et al. (2020), *The Curious Case of Neural Text Degeneration*
- Li et al. (2023), *Contrastive Decoding*; Chuang et al. (2024), *DoLa*
- Sanchez et al. (2023), *Stay on topic with Classifier-Free Guidance* (the LLM version of CFG, used in Session 1)
- Ho & Salimans (2022), *Classifier-Free Diffusion Guidance* (used in the DPM segment of Session 2)
- Karras et al. (2024), *Autoguidance* (arXiv 2406.02507)
- Xie et al. (ICLR 2022), *An Explanation of In-context Learning as Implicit Bayesian Inference*: required reading for the memory agent group
- Liu et al. (2024), *Lost in the Middle*

## Alignment and DDO (Session 1 ④)

- Ouyang et al. (2022) InstructGPT; Rafailov et al. (2023) DPO
- Kirk et al. (2024), *Understanding the Effects of RLHF on LLM Generalisation and Diversity*
- Zheng et al. (ICML 2025), *Direct Discriminative Optimization* (arXiv 2503.01103): <https://research.nvidia.com/labs/dir/ddo/>
- Chen et al. (2024), *SPIN*; Xu et al. (2023), Iterative DPO

## Taxonomy and the trilemma (Session 2 ②③④)

- Tomczak, *Deep Generative Modeling*, 2nd edition (2024)
- Xiao, Kreis & Vahdat (ICLR 2022), *Tackling the Generative Learning Trilemma with Denoising Diffusion GANs*: the source of the trilemma

## AR and comparability (Session 2 ⑤)

- Kalai et al. (2025), *Why Language Models Hallucinate* (arXiv 2509.04664)
- Kim et al., *(QA)²: Question Answering with Questionable Assumptions*
- Jurafsky & Martin, *Speech and Language Processing*, 3rd edition online draft

## Evolution of the families (Session 2 ⑤)

- Higgins et al. (2017) β-VAE; van den Oord et al. (2017) VQ-VAE; Esser et al. (2021) VQ-GAN
- Radford et al. (2015) DCGAN; Arjovsky et al. (2017) WGAN; Karras et al. (2019) StyleGAN
- Ho et al. (2020) DDPM; Song et al. (2021) Score-SDE; Song et al. (2020) DDIM
- Rombach et al. (2022), *High-Resolution Image Synthesis with Latent Diffusion Models*
- Lipman et al. (2023), *Flow Matching for Generative Modeling*
- Song et al. (2023), *Consistency Models*
- Metz et al. (2017) Unrolled GANs; Salimans et al. (2016) minibatch discrimination

## Measurement (Appendix D)

- Kuhn, Gal & Farquhar (ICLR 2023), *Semantic Uncertainty*; Farquhar et al., Nature (2024)
- Kadavath et al. (2022), *Language Models (Mostly) Know What They Know*
- Guo et al. (2017), *On Calibration of Modern Neural Networks*
- Wang et al. (2023), *Self-Consistency Improves Chain of Thought Reasoning*

## Self-study

- MIT 6.S184: <https://diffusion.csail.mit.edu>
- Hung-yi Lee, the *Generative AI* lecture series

---

# Appendix A: pre-course self-check

1. What is the difference between a generative and a discriminative model?
2. What is the chain rule for conditional probability?
3. What is the definition of KL divergence, and why is it asymmetric?
4. State Jensen's inequality.
5. Why does the reparameterization trick let gradients pass through sampling?
6. What does the log density of a multivariate Gaussian look like?

# Appendix B: six key board figures

| Number | Content | Where it is used |
|---|---|---|
| **B-0** | the spectrum (three rows of techniques) | drawn progressively in Session 1 ②③④; returned to in Session 2 ④⑥ |
| **B-C** | the interface contract `sample()` / `logprob(x)` (corner of the board, kept for both sessions) | stated in Session 1 ①; referenced throughout |
| **B-1** | three divergences fitting a bimodal distribution | Session 1 ① |
| **B-2** | training/inference trajectory divergence diagram | raised in Session 1 ④; developed in the AR segment of Session 2 ⑤ |
| **B-3** | the DDO mechanism (implicit discriminator plus dashed self-play line) | Session 1 ④ |
| **B-4** | interface capability matrix (the taxonomy diagram) | Session 2 ③; returned to in each segment of ⑤ |
| **B-5** | the trilemma triangle | Session 2 ④; returned to in each segment of ⑤ |

# Appendix C: timing at a glance

| Session 1, theory: measuring and manipulating a distribution | min | Session 2, instances: building distributions that satisfy the interfaces | min |
|---|---|---|---|
| ① thesis + definitions + contract + choosing a divergence | 36 | ① bridge from Session 1 | 8 |
| ② the unified form of guided generation | 22 | ② surrogates and how the families form | 15 |
| ③ four layers of intervention at inference time | 20 | ③ the taxonomy diagram as interface matrix | 15 |
| ④ intervening at the weights (RLHF/DPO 14 / DDO 18) | 32 | ④ the trilemma | 12 |
| ⑤ wrap-up and homework | 10 | ⑤ family-by-family survey (AR 20 / VAE 10 / GAN 13 / DPM-FM 17) | 60 |
| | | ⑥ placing the lab's work and choosing problems | 10 |
| **Total** | **120** | **Total** | **120** |

# Appendix D: measurement (optional module)

If time allows, or as a separate one-hour measurement workshop:

1. token prob → sequence logprob → length normalisation, and what each of them measures
2. A failure case for predictive entropy: "Paris" and "Paris, France" are two token sequences with one meaning
3. **semantic entropy**: cluster by meaning first, then compute entropy, which removes the dependence on tokenization
4. Self-consistency is Monte Carlo marginalisation: $p(a\mid q)=\sum_r p(a\mid r,q)p(r\mid q)$
5. Why verbalized confidence is often inaccurate: what it measures is surface behaviour shaped by RLHF, not the underlying probability
6. Calibration tools: reliability diagram, ECE, temperature scaling
7. BPB for cross-model comparison; multiple-choice evaluation adds PMI normalisation $\log p(a\mid q)-\log p(a)$

> **Note for the slide editor | Why measurement is a separate module**
> Measurement has a natural home in interface language (whether the numbers read out of the logprob interface can be trusted), but once guidance, DPO and DDO moved into Session 1, the regular slots in both sessions were full. Including it would compress either the family-by-family survey or the depth of DDO. The literature on measurement is mature enough for self-study. If the lab treats the confidence and accuracy line as a priority, a separate one-hour workshop is the better option.
