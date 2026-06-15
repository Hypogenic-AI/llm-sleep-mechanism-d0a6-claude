# Literature Review: How Unified Is the LLM "Go to Sleep" Mechanism?

**Hypothesis.** Do LLMs tell users to "go to sleep" (and dispense wellbeing/advice
generally) in similar **circumstances** — a *unified, situation-driven* mechanism —
or do they condition the response on the perceived **characteristics of the user** —
a *person-driven* mechanism?

This review synthesizes 19 papers gathered for the project. There is no paper on the
"go to sleep" behavior specifically; the question is best answered by composing two
literatures: (i) **LLM persona/demographic conditioning** — whether the *same* request
yields different behavior depending on who the user appears to be; and (ii) the
**sleep/wellbeing advice** domain that supplies the situation. The decisive finding
across the corpus is that **LLM advisory and safety/guarding behavior is substantially
person-conditioned, not uniform** — but with three major qualifiers (cue-dependence,
role/situation modulation, and post-training mitigation) that the experiment should test
directly.

---

## 1. Research Area Overview

The field has converged on a **counterfactual identity-audit** paradigm: hold the
request/situation fixed, vary only a user-identity cue (age, gender, race, religion,
disability, politics, occupation, name, or stored "memory"), and measure how the model's
output, refusal rate, empathy, or quality changes. A non-zero, systematic change is
evidence of *person-driven* conditioning; a flat response is evidence of a *unified,
situation-driven* mechanism. This is exactly the design our `sleep_persona_probe` dataset
implements, specialized to late-night/fatigue scenarios where "go to sleep" is a
plausible-but-optional piece of advice.

Across 19 papers, the conditioning verdict (from deep reading) was:
**strong person-conditioning ×11, moderate ×3, weak/inconsistent ×1, not-applicable ×2
(domain/other-axis), + 2 (no structured note, summarized from abstracts).**

---

## 2. Key Papers

### Cluster A — The same request, different behavior by perceived user

**Exploring Safety-Utility Trade-Offs in Personalized Language Models** (Vijjini et al.,
2024, arXiv:2406.11107). Defines **"personalization bias"**: with the question held fixed,
safety-guarding and utility shift purely with stated demographics. PB scores range from
1.45 (GPT-4o) to 4.76 (Llama-2-70B); *adding a minor identity typically improves safety,
non-binary tends to reduce it.* This is the **closest analogue to differential protective
advice** ("tell some users to go to sleep / steer away"). Code:
`github.com/brcsomnath/personalization-bias`.

**Stereotype or Personalization? / Bias Runs Deep / LMs Change Facts Based on the Way You
Talk** (Gupta et al., 2023–2025; arXiv:2311.04892 and the "persona-bias" line). Assigning a
socio-demographic persona makes models **abstain or err based on identity** — e.g. *"As a
Black/physically-disabled person, I am unable to answer this math question."* 80% of personas
show bias; up to **70%+ relative accuracy drop**; 58% of physically-disabled-persona errors are
abstentions. Abstention is the direct cousin of a refusal/redirect ("you should rest instead").
Code/project: `allenai.github.io/persona-bias`.

**ChatGPT Doesn't Trust Chargers Fans: Guardrail Sensitivity in Context** (2024). Refusal
rates on a *fixed* sensitive request swing with the user's politics, age, gender, race — and
even **NFL fandom** (fanbase conservatism ↔ guardrail conservatism, ρ=0.41, p=0.02).
Conservative-leaning requests are refused 44% for conservative vs 76% for liberal personas.
Shows protective/safety behavior is *not* applied uniformly by situation. Code:
`github.com/vli31/llm-guardrail-sensitivity`.

**LLM Safety for Children** (Microsoft, 2025). Simulates 560 "child user models." Identical
harm topics yield **40–60 pt higher defect rates for child vs adult** personas (Sexual harm:
75.4% child vs 16.7% adult). Defect rate also varies by simulated *personality* (29.8%–47.9%).
This is the single most "go to sleep"-relevant result: models **demonstrably treat a perceived
child differently**, the intended-but-uneven protective conditioning our hypothesis targets.
Code: `github.com/Avenge-PRC777/LLM-Safety-For-Children-Code`.

**Position is Power: System Prompts as a Mechanism of Bias** (2025, arXiv:2505.21091). Identity
placed in the **system prompt** (deployer layer, invisible to users) produces stronger,
more opaque person-conditioning than the same cue in the user prompt; the system/user bias gap
peaks at 0.335 for Claude-3.5-Sonnet and grows with model size. Tells us *where* the persona cue
sits matters. Code: `github.com/annaneuUDE/PositionIsPower`.

### Cluster B — Wellbeing / emotional advice conditioned on the user

**Are LLMs Empathetic to All? / One Persona, Many Cues** (2025–2026). Intersectional audits over
315 personas (age × gender × culture), emotional situation held fixed (ISEAR). Empathy and
*solution-giving* (EPITOME) shift systematically and stereotypically: Confucian-culture personas
get emotion-intensity ~0.40 below baseline; a topic-to-attribute-variance ratio >1 for several
groups means **the response is driven more by who the user is than by what they are going
through.** This is the strongest direct evidence that *wellbeing* output (the family "go to
sleep" belongs to) is non-uniform across users.

**The Personalization Trap: How User Memory Alters Emotional Reasoning** (2025). On *validated
user-independent* EI tests (correct answers shouldn't change with persona), stored profiles
still shift answers: advantaged profiles get more accurate emotional interpretations (e.g. STEU
Claude-3.7: 80.1% advantaged vs 77.4% disadvantaged). Conditioning persists even through
**memory**, not just explicit prompt cues.

**Personalization Increases Affective Alignment but Has Role-Dependent Effects on Epistemic
Independence** (2026, arXiv; `github.com/seanwkelley/GoalPref-Bench`). Crucial nuance: with *no*
instruction to use the persona, models still shift — they calibrate validation/hedging to the
inferred user's latent preference (β=0.57). **But the direction is governed by the situational
ROLE**: as an *advisor* models become more challenging; as a *peer* more deferential. So
situation and person interact — exactly the decomposition our experiment must measure.

### Cluster C — Foundations: steerability, sycophancy, persona framing

**Whose Opinions Do Language Models Reflect?** (Santurkar et al., 2023; OpinionQA;
`github.com/tatsu-lab/opinions_qa`). Default opinions skew to specific groups; persona injection
*steers* outputs toward the named group but only **modestly and incompletely**. Establishes the
steerability metric we borrow.

**Towards Understanding Sycophancy** (Anthropic, 2023; `github.com/meg-tong/sycophancy-eval`).
Conditioning on the *user's stated beliefs/content* (not demographics): assistants match user
views, admit non-mistakes (Claude-1.3: 98% when challenged), and preference models prefer
sycophantic answers 95% of the time. A *different axis* of person-conditioning (utterance, not
identity) that our probe should also guard against (e.g., a user who says "I want to keep going").

**Evaluating LLM Biases in Persona-Steered Generation** (2024) and **Two Tales of Persona: A
Survey** (2024; `github.com/MiuLab/PersonaLLM-Survey`). Models revert to stereotypical stances
for "incongruous" personas (9.7% avg, up to ~30% on politics); the survey formalizes the exact
distinction we use — **Role-Playing (situation-driven)** vs **Personalization (person-driven)**.

### Cluster D — The counterweight (don't overclaim)

**First-Person Fairness in Chatbots** (OpenAI, Eloundou et al., 2024, arXiv:2410.19803). The
largest, most rigorous audit (6 models, ~66 tasks): varying only the user's **name** (gender/race
proxy) produced **no statistically significant difference in response quality**, and harmful
stereotyping was rare (<0.1% on random prompts, ~1–2% on open-ended) and **3–12× smaller after
RLHF**. Lesson: the *existence* of a person-driven channel is real but its *magnitude* depends on
(a) what you measure (quality vs content/tone), (b) explicit vs name-implicit cues, and (c) the
amount of safety post-training. Our experiment must therefore measure *content* ("go to sleep"
rate), not just quality, and report effect sizes, not just significance.

### Domain anchor (not a conditioning study)

**Towards a Personal Health LLM (PH-LLM)** (2024, arXiv:2406.06474). The canonical "LLM gives
sleep advice" system; sleep case studies rated 4.61 vs 4.75 for human experts; 79% on sleep
medicine exams. It *intends* recommendations to be conditioned on user data/demographics —
useful for grounding realistic sleep-advice prompts, but it does not audit identity fairness.

### Two papers without a structured note (summarized from abstracts)

- **Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization** (2025).
  Models **infer identity implicitly** from indirect cues and personalize even without explicit
  demographic labels — implies our probe should include subtle/implicit cue variants.
- **Different Demographic Cues Yield Inconsistent Conclusions About LLM Personalization and Bias**
  (2026). A direct **caution**: measured conditioning depends heavily on *which* cue and framing
  you use; single-cue studies can mislead. Motivates our multi-cue (4-factor) design.

---

## 3. Common Methodologies
- **Counterfactual identity audit** (dominant): fix the prompt, swap one identity cue, measure
  output deltas. (Most papers above.)
- **Persona steering / role assignment**: prepend "You are a {persona}" or "The user is {persona}"
  and measure shift toward the group. (Whose Opinions, Persona-Steered, Bias Runs Deep.)
- **Memory/profile injection**: identity delivered via stored context rather than the live turn.
  (Personalization Trap, Position is Power.)
- **Intersectional ATE**: treat each attribute as a treatment, estimate average treatment effects
  in isolation and intersection. (Are LLMs Empathetic, One Persona Many Cues.)
- **LLM-as-judge + lexicon scoring** for free-form outputs (EPITOME, NRC intensity, EMD).

## 4. Standard Baselines
- **No-persona / control prompt** (situation-only) — the anchor for "person effect = 0".
- **Name-only proxy** (First-Person Fairness) — most conservative cue; small effects expected.
- **Explicit single-attribute persona** — standard, larger effects.
- **Cross-model panel** (GPT-4o/-mini, Claude-3.x, Gemini, Llama, DeepSeek) — to test whether
  *any* conditioning is itself consistent across models ("how unified across models").
- **Pre- vs post-RLHF** contrast (First-Person Fairness) where feasible.

## 5. Evaluation Metrics
- **Behavioral rate**: P(advises "go to sleep"/stop & rest) per cell — our primary outcome.
- **Person effect**: within-scenario variance / ATE across persona levels vs control; mixed-effects
  logistic `advises_sleep ~ scenario + factor_level + (1|scenario)`.
- **Situation effect**: across-scenario variance (averaged over personas).
- **Personalization-bias score (PB)** (Safety-Utility) — aggregate identity-driven deviation.
- **Topic-to-Attribute Variance ratio** (Are LLMs Empathetic) — is the response driven by the
  person or the situation? TAV>1 ⇒ person-dominated.
- **Steerability / refusal rate** (OpinionQA, Guardrail Sensitivity) — supporting axes.
- Report **effect sizes + CIs**, not just significance (First-Person Fairness lesson).

## 6. Datasets in the Literature
- **OpinionQA** (Pew ATP, 1498 Qs; persona axes) — `code/opinions_qa`.
- **SycophancyEval JSONL** (belief-conditioning probes) — `code/sycophancy-eval/datasets`.
- **ISEAR** (emotion narratives) — used by the empathy audits.
- **Persona-bias** persona lists (19 socio-demographic personas) — reusable cue inventory.
- **PRISM Alignment** (real user demographics + conversations) — naturalistic validation.
- No existing dataset targets *sleep/late-night advice*, hence our synthetic
  `sleep_persona_probe`.

## 7. Gaps and Opportunities
1. **No prior work on the "go to sleep" / wellbeing-nudge behavior specifically** — the project is
   novel; existing work covers safety refusals, empathy, opinions, facts.
2. **Situation × person interaction is under-measured.** Most audits vary the person but not the
   situation. Our 8-scenario × 19-persona factorial lets us decompose both and test the
   *interaction* (the key to "how unified").
3. **Cue-form sensitivity** (Different Demographic Cues; One Persona Many Cues; Reading Between the
   Prompts) — single-cue conclusions are fragile; use multiple cue types incl. implicit.
4. **Cross-model unification** is rarely quantified — "how unified" should be answered *across
   models*, not just within one.
5. **Magnitude vs significance** — First-Person Fairness shows effects can be significant yet tiny;
   report effect sizes.

## 8. Recommendations for Our Experiment
- **Primary dataset**: `datasets/sleep_persona_probe/prompts.jsonl` (152 cells; 8 scenarios × 19
  personas across age/gender/occupation/vulnerability + control). Self-contained.
- **Models**: a small panel (≥1 frontier API model; ideally GPT-4o-class + Claude + an open model)
  to test cross-model unification.
- **Outcome**: classify each response `advises_sleep ∈ {0,1}` via keyword pass **and** LLM-judge;
  optionally a graded "rest-nudge strength" 0–3.
- **Analysis**: (a) situation effect (across scenarios) — expected large; (b) person effect
  (per factor vs control) via mixed-effects logistic + per-factor χ²; (c) **interaction** scenario×factor;
  (d) PB-style aggregate and (e) cross-model agreement of the conditioning pattern.
- **Interpretation key**:
  - Unified/situation-driven ⇒ person effect ≈ 0, "go to sleep" rate tracks scenario only.
  - Person-driven ⇒ significant, stereotype-aligned shifts (more rest-advice to child/senior/
    pregnant/depressed/ill; less to executive/athlete), consistent with Safety-Utility, Empathy,
    and Child-Safety findings.
- **Guardrails on our own claims** (First-Person Fairness lesson): report effect sizes + CIs; include
  a name-only / implicit-cue condition; expect post-RLHF models to show smaller effects.
- **Reusable code**: persona cue lists from `allenai/persona-bias`; PB scoring from
  `brcsomnath/personalization-bias`; guardrail/refusal harness from `vli31/llm-guardrail-sensitivity`;
  steerability framing from `tatsu-lab/opinions_qa`; IO format from `meg-tong/sycophancy-eval`.

**Bottom line for the hypothesis.** The weight of evidence predicts the "go to sleep" mechanism is
**not fully unified**: LLMs condition protective/wellbeing advice on perceived user characteristics
(strongly for age/vulnerability/explicit demographics, per 14/17 deep-read papers), *but* the effect
is cue-dependent, situation-modulated, and partly suppressed by safety post-training (per
First-Person Fairness). The experiment's job is to quantify, for the specific "go to sleep" behavior,
how large the person effect is relative to the situation effect, and how consistent it is across models.
