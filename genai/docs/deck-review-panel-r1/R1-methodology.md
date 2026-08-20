# R1 Review — Methodology / Mathematical Rigour

Reviewer: R1 (methodology, mathematical and logical soundness of the deck organisation)
Object under review: `docs/deck-blueprint.md` (2026-08-16 rebuild), checked against the
authoritative outline `docs/Generative_Models_Intro_Two_Session_Outline_EN.md`.
Scope per remit: prerequisite ordering, dropped/distorted mathematical content, timing
arithmetic, page density, and the sample()/logprob interface-contract thread. Prose is out
of scope (it does not exist yet).

Verified before review: both session budgets sum to 120 min (L1: 36+22+20+32+10;
L2: 8+15+15+12+60+10). The DDO optimum algebra on L1 p46 is consistent with the ② table
(base log p_ref + (1/β)(log p_data − log p_ref) = (1−1/β)log p_ref + (1/β)log p_data).
Those are not findings; everything below is.

---

## Findings

**1. MUST — DPO is referenced three times but never taught.**
Blueprint L1 §④ (pp. 36–47) vs. outline §④, whose heading is "SFT → RLHF / DPO (about
14 min)". The blueprint's ④ has SFT (p37), RLHF (p38–40), then jumps to DDO (p41). Yet
p45's DPO/DDO comparison table uses DPO's implicit reward parameterisation
(reward = β log π_θ/π_ref), and L2 p9 asserts "guidance/DPO/DDO 不適用" for GAN — both
lean on a derivation that exists nowhere in either deck. Under production rule 3 the p45
table row is an unsupported claim. **Fix**: insert one page between p38 and p39: substitute
the RLHF closed form π* ∝ π_ref exp(r/β) into the preference loss to obtain DPO's implicit
reward (Rafailov et al., 2023). This is a 3-line derivation and it is exactly what the p45
table and the L2 p9 claim need. It also honours the outline's 14-min "RLHF/DPO" budget.

**2. MUST — Six new demos carry zero minutes; ~12–18 min of Session time is unaccounted.**
Blueprint L1 p13, p24, p35, p44 and L2 p7, p21 embed demos (divergence-2d,
guidance-playground, asr-noisy-channel, mle-vs-ddo-gradient, interface-contract,
exposure-bias-track) inside sections whose outline budgets (Appendix C) contain no demo
time. The outline's demo plan budgets minutes only for the three existing 2D demos
(GAN 5 / VAE 3 / FM 3, already inside the ⑤ family timings). At a conservative 3 min each,
the six unbudgeted demos consume 18 min — 15% of Session 1 — with no compensating cut.
Section ① alone (36 min, 18 pages incl. one demo) already averages 2 min/page. **Fix**: add
a per-demo minute column to the blueprint's "Demo 對應" table and show revised per-section
sums that still hit 36/22/20/32 and 23/20; cut or demote to backup any demo that does not
fit.

**3. MUST — L2 p9 states a causal claim that p10 falsifies one page later.**
L2 p9 (outline §③ reading 1c): GAN "無正規化密度換得一步生成" — no normalised density is
traded for one-step sampling. L2 p10 then presents Normalizing Flow: exact logprob AND
one-step sampling. So "maintains no density ⟹ samples fast" is not an implication, and the
deck refutes itself across adjacent pages. The true trade is architectural freedom: GAN's
generator is unconstrained by invertibility (NF) or sequential factorisation (AR), which is
what buys quality at one step. **Fix**: reword the p9 third bullet as a constraint-freedom
claim ("不維護密度,生成器不受可逆性或序列分解約束"), and have p10 explicitly close the
loop (NF shows density and one-step coexist; the cost is the invertibility constraint).

**4. MUST — L2 p34 packs four distinct blocks onto one 16:9 page.**
L2 p34 carries (a) a 6-stage improvement timeline DCGAN→cGAN→WGAN→StyleGAN→BigGAN→
distillation-target shift, (b) the DDO call-back (d* = p/(p+q) as prototype), (c) the
one-way-transfer argument (discriminator idea moves into logprob families, not back), and
(d) the applications list. In the outline these are three separate blocks of GAN §⑤(3),
the DDO paragraph, and §⑤(4). The DDO/transfer argument is the conceptual payoff of the
whole GAN segment and will be illegible as a footnote under a timeline. **Fix**: split into
p34a (timeline + applications, Timeline 元件) and p34b (DDO 回接 + transfer direction);
the 13-min GAN budget minus the 5-min demo leaves 8 min for 6 pages, which still works.

**5. SHOULD — The spectrum's three rows are not drawn in the order the outline requires.**
Outline front matter: "All three rows are drawn in full during Session 1", rows being
(1) training objective, (2) decoding settings, (3) weight fine-tuning — naturally owned by
§①, §②③, §④ respectively. Blueprint: SpectrumRows first appears at p25 labelled
"第一列" inside §② (whose conclusions — coefficient as coordinate — are row-2 material),
row 2 is never explicitly drawn anywhere, and p37 jumps to "第三列起點". As organised, the
progressive build is one row short and mislabelled, and p48 ("三列完整呈現") completes a
sequence the audience never saw. **Fix**: add a row-1 strip to p18 (it summarises exactly
forward KL / JSD / reverse KL), relabel p25 as row 2, keep p37 as row 3.

**6. SHOULD — L1 p23 asserts the RLHF and DDO optima 15–23 pages before their derivations,
under a writing spec that forbids forward references.**
The ② table rows "RLHF 最優解" (derived p38) and "DDO 最優解" (derived p41/46) are stated
as fact on p23. Production rule 3 demands support with each claim; blueprint spec rule 1
simultaneously forbids "之後會講" phrasing, so the standard escape hatch is banned. The
outline resolves this by having ④ point back to the table, not the table point forward.
**Fix**: on p23, present the two rows as cited statements (Rafailov et al., 2023; Zheng et
al., 2025) — citation is support under rule 3 and needs no forward reference; p38 and p46
then use 回接 phrasing ("②表中的一列") exactly as the blueprint already plans.

**7. SHOULD — β appears in Figure B-3 (p42) before it is introduced and justified (p46);
α is introduced but never used.**
The outline's B-3 spec draws the implicit discriminator as σ(β log p_θ/p_ref); the
blueprint places B-3 at p42, but the numerical justification for β (log p_θ ~ 10³ kills
the sigmoid gradient) and its definition arrive only at p46. Meanwhile p41 writes the
β-free d_θ = σ(log p_θ/p_ref) whose BCE optimum p_θ = p_data holds only in that β = 1
form — so the figure quietly changes the object mid-argument. Separately, p46 introduces
"α,β" but α appears in no downstream formula on any page. **Fix**: either strip β from the
p42 figure (draw the β-free version) or move the α,β introduction sentence from p46 up to
p41; and either state α's role (the loss-term weight in Zheng et al., 2025) or drop the
symbol.

**8. SHOULD — The σ-form of the optimal discriminator is used without the one-line
equivalence being established.**
p17 establishes D* = p/(p+q); p41 starts from d* = σ(log p_data/p_ref). The identity
σ(log p/q) = p/(p+q) is one line of algebra but it is the hinge of the entire DDO
derivation, and no page owns it. L2 p34 reuses the same identification. **Fix**: add the
identity to p17 (where D* is derived) as the closing line, so p41 can invoke it by direct
statement.

**9. SHOULD — The Session 1 wrap-up drops half of the outline's §⑤ content, weakening the
designed bridge into Session 2.**
Outline §⑤ requires stating two deliberate assumptions/open questions: (i) every operation
assumed a π_ref with both interfaces, and how that is satisfied is Session 2's subject;
(ii) *two* open questions — what surrogate each method uses for the missing logprob, and
how to train a model with no logprob at all. Blueprint p48–49 keep only the second open
question. The dropped surrogate question is precisely what L2 p3's surrogate table is
designed to answer; without it the table answers a question never asked. **Fix**: add both
the assumption statement and the surrogate question to p48 (or a p48/49 split), phrased as
open problems, not as previews.

**10. SHOULD — DDO experimental numbers are placed where the outline forbids them, and
absent where the outline promises them.**
Outline §④: "Summary of the experimental results (specific families and numbers are left
to Session 2)" — Session 1 is model-agnostic by contract. Blueprint L1 p47 specifies a
"數據表", which cannot carry meaning without naming model families, violating the
session's abstraction discipline; and no L2 page carries the deferred numbers (L2 p40's
DDO comparison is qualitative). **Fix**: keep p47 qualitative (the two structural
observations — MLE ceiling, truncation as hidden temperature — need no family names) and
add the families-and-numbers table to L2, e.g. as part of p40 or a p40b.

**11. SHOULD — All three outline-mandated LLM-native demos are dropped; five unlisted
demos are invented, without a recorded mapping.**
Outline demo plan, in priority order: (P1) token probability browser — explicitly built to
show "the coefficient is a coordinate along the spectrum and that the logprob interface
really exists" for §②③; (P2) calibration scatter; (P3) false premise / semantic entropy.
None appears in the blueprint; instead guidance-playground, asr-noisy-channel,
mle-vs-ddo-gradient, interface-contract, exposure-bias-track appear, none of which is in
the outline. Some substitutions may be improvements (guidance-playground plausibly
subsumes P1's first purpose), but the blueprint records no such decision, and P1's second
purpose — making the logprob interface tangible token-by-token — is not obviously covered.
**Fix**: add a mapping row per outline demo (covered-by / dropped-with-reason); if P1 is
not subsumed, restore it at L1 p31 (Layer 2) where temperature/top-p are taught. (P2/P3
may legitimately fall with Appendix D, but say so.)

**12. SHOULD — Three further pages exceed 16:9 capacity and must be split.**
(a) L1 p29: ICL-as-Bayes integral + memory-agent reading + lost-in-the-middle + two
citations — split into ICL formula page and implications page. (b) L1 p40: β-governed
safety/diversity consequence (Kirk et al., 2024) + pointwise-scorer structural limit +
LLM-as-judge — two arguments, two pages; the structural-limit argument is reused verbatim
at L2 p5 and p32 and deserves its own page for the call-backs to land. (c) L2 p23: full AR
improvement history + speed-side methods + four application domains on one page — split
history from applications as the VAE/GAN/DPM segments' page budgets already do
implicitly. L1 p17 (D* derivation + 2JSD−2log2 value + I(X;Z) reading) is borderline;
splitting is optional if finding 8's identity line moves there.

**13. CONSIDER — Near-empty pages.**
L1 p20 (one motivating sentence) can merge into p21's GuidanceForm anatomy page. L2 p4
("GAN 用判別器是必然", one sentence) is a row-reading of the p3 table and can merge into
p3 or p5. L2 p13 (three call-back sentences) can merge into p12's Trilemma page as
annotations on the triangle's edges. None is wrong, but at 23 min for 7 pages (L2 ①②)
the thin pages mask how tight the dense ones are.

**14. CONSIDER — Two mathematical claims lack the citation the outline's rule 3 requires,
and the reference does not exist in the outline's list either.**
(a) L1 p15: "√JSD 滿足度量公理" — the standard source is Endres & Schindelin (2003), *A
new metric for probability distributions*, absent from the outline references; add it or
drop the metric-axiom claim. (b) L1 p22 (false premise) carries no citation while the
outline's reference list supplies Kalai et al. (2025) and (QA)² for exactly this segment;
carry them onto the page. (c) L2 p36 invokes the probability flow ODE one page before its
citation (Song et al., 2021) appears on p38; cite at first use.

**15. CONSIDER — The "信心與正確率 → Appendix D" mapping-table row dangles.**
The lab mapping table (outline §0) gives the confidence-vs-accuracy topic the spectrum
position "the measuring instrument itself (Appendix D)". The blueprint reproduces the
table at L1 p3 and replays it at L2 p42, but drops Appendix D entirely (legitimately —
the outline marks it optional). As organised, one of six rows points at a module that no
longer exists. **Fix**: give the row a self-contained one-line position on p3/p42
("量測 logprob 介面讀數的可信度") and, if the separate workshop is planned, say so only
in the editor notes, not on the slide.

**16. CONSIDER — Two bookkeeping inconsistencies.**
(a) L1 header says "約 48 頁" but the tables enumerate 49 pages (p48–49 in §⑤). (b) The
outline requires the thesis to return "on the last slide of Session 2"; the blueprint puts
the return at L2 p43 with p44–46 (closing remarks, references, end page) after it. Either
move the thesis return to sit adjacent to the closing remarks (swap p43/p44) or accept and
note the deviation — the current order buries the course's stated capstone under logistics.

---

## Tally

- MUST: 4 (findings 1–4)
- SHOULD: 8 (findings 5–12)
- CONSIDER: 4 (findings 13–16)
