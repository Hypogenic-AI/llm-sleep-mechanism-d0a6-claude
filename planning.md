# Planning: How Unified Is the LLM "Go to Sleep" Mechanism?

## Motivation & Novelty Assessment

### Why This Research Matters
Reports of chatbots telling users to "go to sleep" after long sessions raise a
governance-relevant question: is a protective/wellbeing nudge applied **uniformly**
(driven by the *situation* — it's late, the user is tired) or is it **conditioned on
who the user appears to be** (age, gender, occupation, vulnerability)? If the latter,
the same late-night plea gets a different protective response depending on perceived
identity — a fairness and consistency problem for deployed assistants, and a window
into how much a model's "care" behavior is situation- vs person-driven.

### Gap in Existing Work
The literature review (19 papers) finds **no study of the "go to sleep" / wellbeing-nudge
behavior specifically**. Adjacent work shows LLM advisory/safety behavior is substantially
*person-conditioned* (Safety-Utility PB scores; Child-Safety 40–60pt defect gaps; empathy
audits with TAV>1), **but** the OpenAI First-Person-Fairness counterweight shows name-only
effects can be significant-yet-tiny and shrink post-RLHF. Two cautions (Different Demographic
Cues 2026; One Persona Many Cues) warn single-cue conclusions are fragile. Crucially, most
audits vary the *person* but hold *situation* fixed, so the **situation × person interaction**
— the heart of "how unified" — is under-measured, and **cross-model unification** is rarely
quantified.

### Our Novel Contribution
A purpose-built **counterfactual identity audit of the go-to-sleep behavior**, run as an
**8-scenario × 19-persona factorial across a 5-model cross-provider panel**, that
**decomposes the variance** in the protective nudge into situation, person, model, and
interaction components. This directly operationalizes "how unified": situation-dominated
variance ⇒ unified mechanism; sizable person/interaction variance ⇒ person-driven.

### Experiment Justification
- **E1 — Response collection (5 models × 152 cells × 3 reps).** The raw behavioral substrate;
  multi-model panel is required to answer cross-model unification, replicates to estimate
  within-cell stochastic variance.
- **E2 — Dual labeling (regex + LLM judge).** Need a validated `advises_sleep` outcome and a
  graded `rest_nudge_strength` (0–3); two independent labelers + Cohen's κ guards against
  judge artifacts (a key reproducibility risk flagged in the lit review).
- **E3 — Variance decomposition + mixed-effects models.** The core test: how much of the
  variance is situation vs person vs interaction; per-factor effects vs control with effect
  sizes + CIs (First-Person-Fairness lesson).
- **E4 — Stereotype-direction & cross-model agreement.** Tests *whether* the person effect is
  stereotype-aligned (more rest-advice to child/senior/pregnant/depressed; less to
  executive/athlete) and whether models agree on the conditioning pattern.

---

## Research Question
Do LLMs tell users to "go to sleep" in **similar circumstances** (a unified, situation-driven
mechanism), or do they condition the nudge on the **perceived characteristics of the user**
(a person-driven mechanism)? And is whatever conditioning exists **consistent across models**?

## Hypothesis Decomposition
- **H1 (situation effect):** P(advises sleep) varies strongly across the 8 scenarios. *(expected large)*
- **H2 (person effect):** Holding scenario fixed, P(advises sleep) shifts with persona vs the
  no-persona control. *(expected non-zero)*
- **H3 (stereotype direction):** The shift is stereotype-aligned — higher for
  child/senior/pregnant/ill/depressed, lower for executive/athlete/high-status.
- **H4 (interaction):** The person effect depends on the scenario (situation × person interaction).
- **H5 (cross-model unification):** The *pattern* of conditioning is correlated across the 5 models.
- **Magnitude framing:** Even if H2–H4 are significant, report whether the **situation effect
  dwarfs the person effect** (→ "mostly unified") via variance decomposition.

Variables: **IV** = scenario (8), factor/level (19 personas across age/gender/occupation/
vulnerability + control), model (5). **DV** = `advises_sleep ∈ {0,1}`, `rest_nudge_strength ∈ {0..3}`.

## Proposed Methodology

### Approach
Counterfactual identity audit (dominant paradigm in the corpus): fix the situation, swap one
identity cue, measure the behavioral delta. Use **real LLM APIs** via OpenRouter for a
cross-provider panel.

### Experimental Steps
1. Collect responses: 5 models × 152 prompts × 3 replicates, temperature 0.7, seeded, async with
   on-disk cache for resumability.
2. Label each response twice: (a) deterministic regex/keyword classifier, (b) LLM judge
   (`gpt-5.4-mini`, temperature 0) returning binary `advises_sleep` + graded `strength` 0–3.
   Report regex↔judge agreement (Cohen's κ); judge is primary, regex is a robustness check.
3. Variance decomposition (ANOVA-style components) on the cell-mean P(sleep): scenario, factor,
   model, scenario×factor, residual.
4. Mixed-effects logistic `advises_sleep ~ C(level)` with scenario random intercept; per-factor
   χ²/Cramér's V vs control; per-level risk differences vs control with bootstrap 95% CIs.
5. Stereotype-direction test (signed contrast: vulnerable/young/old vs high-status).
6. Cross-model agreement: correlate per-cell P(sleep) and per-persona effect vectors across models.

### Baselines
- **No-persona control** (situation-only) — anchors person effect = 0.
- **Cross-model panel** — `gpt-5.4-mini`, `claude-sonnet-4.6`, `gemini-3.5-flash`,
  `llama-3.3-70b-instruct`, `deepseek-v3.2`.
- **Random/chance** reference for cross-model correlation.

### Evaluation Metrics
- Primary: per-cell P(advises_sleep); mean `rest_nudge_strength`.
- Person effect: mixed-effects logistic coefficients; per-factor χ², Cramér's V; risk differences vs control + CIs.
- Situation effect: between-scenario variance of P(sleep).
- Unification: **variance components** (ω²/η² for scenario vs factor vs model vs interaction).
- Cross-model: Pearson/Spearman of effect vectors; mean pairwise r.

### Statistical Analysis Plan
α = 0.05. Mixed-effects logistic via `statsmodels`. χ² for categorical factor×outcome.
Effect sizes (Cramér's V, Cohen's d, ω²) reported alongside p-values. Bonferroni/FDR for the
per-level family of contrasts. Bootstrap (10k) CIs for risk differences. Report effect sizes
+ CIs everywhere (First-Person-Fairness guardrail), not just significance.

## Expected Outcomes
- **Unified/situation-driven** ⇒ scenario explains the bulk of variance; person effect ≈ 0; flat across personas.
- **Person-driven** ⇒ significant, stereotype-aligned person effect; sizable factor + interaction variance.
- Lit-informed prediction: **partially unified** — large situation effect, real but smaller
  stereotype-aligned person effect (strongest for age/vulnerability), variable across models.

## Timeline and Milestones
- Setup + data check: done. Collection (E1): ~20 min. Labeling (E2): ~15 min. Analysis+figures
  (E3/E4): ~30 min. Reporting: ~25 min. Buffer 25%.

## Potential Challenges
- **Judge artifacts** → dual labeling + κ; manual spot-check.
- **Rate limits / model 404s** → async retry + on-disk cache + pinned model IDs verified live.
- **Stochasticity** → 3 replicates, temperature 0.7, report within-cell variance.
- **Refusals/derailments** → track non-answers as a separate category, exclude from P(sleep) denominator-sensitivity check.
- **Multiple comparisons** → FDR correction on the contrast family.

## Success Criteria
Clean labeled dataset (≥95% parseable), a defensible variance decomposition answering
"how unified", mixed-effects person-effect estimates with CIs, a cross-model agreement number,
and an honest situation-vs-person magnitude verdict in REPORT.md.
