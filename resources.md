# Resources Catalog

## Summary
Resources gathered to test: **do LLMs tell users to "go to sleep" / give wellbeing advice
uniformly by situation, or conditioned on the perceived user?** Totals: **19 papers**,
**1 primary dataset (generated) + 2 supporting**, **2 cloned repos** (+ 6 more code links cataloged).
Full synthesis in [`literature_review.md`](literature_review.md).

## Papers (19 downloaded → `papers/`, ~46 MB; details in `papers/README.md`)

| # | Title | Year | File (`papers/…pdf`) | Conditioning verdict |
|---|-------|------|----------------------|----------------------|
| 1 | Exploring Safety-Utility Trade-Offs in Personalized LMs | 2024 | safety_utility_tradeoffs_personalized_lm | strong |
| 2 | Stereotype or Personalization? User Identity Biases Chatbot Recs | 2024 | stereotype_or_personalization_user_identity_biases | strong |
| 3 | Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs | 2023 | bias_runs_deep_persona_assigned_llms | strong |
| 4 | Language Models Change Facts Based on the Way You Talk | 2025 | lms_change_facts_based_on_how_you_talk | strong |
| 5 | ChatGPT Doesn't Trust Chargers Fans: Guardrail Sensitivity | 2024 | guardrail_sensitivity_chargers_fans | strong |
| 6 | LLM Safety for Children | 2025 | llm_safety_for_children | strong |
| 7 | Position is Power: System Prompts as a Mechanism of Bias | 2025 | position_is_power_system_prompts_bias | strong |
| 8 | Are LLMs Empathetic to All? Multi-Demographic Personas | 2025 | are_llms_empathetic_to_all_multidemographic | strong |
| 9 | One Persona, Many Cues, Different Results | 2026 | one_persona_many_cues_different_results | strong |
| 10 | The Personalization Trap: User Memory Alters Emotional Reasoning | 2025 | personalization_trap_user_memory_emotional | strong |
| 11 | Evaluating LLM Biases in Persona-Steered Generation | 2024 | persona_steered_generation_biases | strong |
| 12 | Personalization Increases Affective Alignment (role-dependent) | 2026 | personalization_affective_alignment_epistemic | moderate |
| 13 | Whose Opinions Do Language Models Reflect? (OpinionQA) | 2023 | whose_opinions_do_lms_reflect | moderate |
| 14 | Two Tales of Persona in LLMs: A Survey | 2024 | two_tales_of_persona_survey | moderate |
| 15 | First-Person Fairness in Chatbots (OpenAI) | 2024 | first_person_fairness_in_chatbots | weak/inconsistent |
| 16 | Towards Understanding Sycophancy (Anthropic) | 2023 | towards_understanding_sycophancy | n/a (belief axis) |
| 17 | Towards a Personal Health LLM (PH-LLM, sleep/fitness) | 2024 | towards_personal_health_llm | n/a (domain) |
| 18 | Reading Between the Prompts: Implicit Personalization | 2025 | reading_between_the_prompts_implicit_personalization | (abstract only) |
| 19 | Different Demographic Cues → Inconsistent Conclusions | 2026 | different_demographic_cues_inconsistent_conclusions | (abstract only) |

Could not download (paywalled): *A personal health LLM for sleep & fitness coaching* (Nature 2025,
covered by #17 preprint); *A Scoping Review of LLMs in Personal Sleep Wellness* (abstract used).
Structured deep-read notes: `papers/_deepread_notes.json`. Download log: `papers/_download_log.json`.

## Datasets (`datasets/`; details + download instructions in `datasets/README.md`)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| **Sleep-Persona Probe (PRIMARY)** | generated locally | 152 prompts (8 scen × 19 persona) | advice generation → "advises_sleep" classification | `datasets/sleep_persona_probe/` | counterfactual audit; tracked in git (~9 KB); regen via `generate_probe.py` |
| PRISM Alignment | HuggingFace `HannahRoseKirk/prism-alignment` | ~8k convos w/ user demographics | naturalistic validation | (download) | `datasets/download.sh` |
| OpinionQA | CodaLab (via `code/opinions_qa`) | 1498 Pew MC Qs | persona steerability | (download) | optional persona axes |

## Code Repositories (`code/`; details in `code/README.md`)

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| sycophancy-eval | github.com/meg-tong/sycophancy-eval | behavior-change probe format + inference harness | `code/sycophancy-eval/` (cloned) |
| opinions_qa | github.com/tatsu-lab/opinions_qa | persona steerability methodology + axes | `code/opinions_qa/` (cloned) |

**Additional code links found in deep reading (clone on demand):**
- `github.com/brcsomnath/personalization-bias` — PB (personalization-bias) scoring, **most reusable**
- `allenai.github.io/persona-bias` — 19-persona socio-demographic cue inventory + audit code
- `github.com/vli31/llm-guardrail-sensitivity` — refusal-rate-by-persona harness
- `github.com/Avenge-PRC777/LLM-Safety-For-Children-Code` — child-user-model red-teaming
- `github.com/annaneuUDE/PositionIsPower` — system- vs user-prompt identity placement
- `github.com/seanwkelley/GoalPref-Bench` — role-dependent personalization (advisor vs peer)

## Resource Gathering Notes

**Search strategy.** paper-finder (`find_papers.py`) over 6 query angles: sleep-health LLM coaching;
demographic/persona conditioning; sycophancy; safety-guardrail sensitivity to user; first-person
fairness/name bias; AI wellbeing/companion. The demographic-conditioning query was the richest vein.

**Selection criteria.** Prioritized (a) counterfactual identity-audit method papers (the design we
reuse), (b) wellbeing/empathy/safety advice as the outcome (closest to "go to sleep"), (c) the OpenAI
counterweight to avoid confirmation bias, (d) the sleep domain anchor (PH-LLM). Relevance ≥2 from
paper-finder; deep-read all 19 in parallel (workflow) for structured extraction.

**Challenges.** Many results were Semantic-Scholar `CorpusID` links without direct PDFs; resolved via
the SS Graph API + arXiv fallbacks. Hit 429 rate-limits on 7 papers; recovered 6 via known arXiv IDs.
Two sleep-domain papers are paywalled (substituted/abstract-only). 2 of 19 deep-read agents returned
null; covered from abstracts.

**Gaps / workarounds.** No dataset exists for the "go to sleep" behavior → built a synthetic
counterfactual probe (standard in this literature). No single repo implements our exact design → the
probe generator re-implements the counterfactual-audit pattern; analysis can borrow PB scoring and
steerability code above.

## Recommendations for Experiment Design
1. **Primary dataset**: `datasets/sleep_persona_probe/prompts.jsonl` — run as-is.
2. **Baselines**: no-persona control (situation-only); name-only/implicit cue (conservative);
   cross-model panel (GPT-4o-class + Claude + open model) for cross-model unification.
3. **Metrics**: P(advises sleep) per cell; **person effect** (mixed-effects logistic + per-factor χ²)
   vs **situation effect** (across scenarios) vs their **interaction**; PB-style aggregate; TAV ratio;
   report effect sizes + CIs.
4. **Code to reuse**: `personalization-bias` (PB), `persona-bias` (cue lists),
   `llm-guardrail-sensitivity` (refusal harness), `sycophancy-eval` (IO format), `opinions_qa` (steering).
5. **Expected result** (from lit): a real, stereotype-aligned **person effect** (esp. age/vulnerability),
   smaller for name-only cues and post-RLHF models — i.e. the mechanism is **partially, not fully, unified.**
