# Downloaded Papers

19 PDFs supporting the hypothesis: *do LLMs give "go to sleep" / wellbeing advice
in similar **circumstances** (unified, situation-driven), or do they condition on
the perceived **characteristics of the user** (person-driven)?*

Papers are grouped by the role they play in answering that question. Detailed
structured notes are in `../literature_review.md`.

## A. Core mechanism — does LLM behavior change with perceived user identity?

1. **Stereotype or Personalization? User Identity Biases Chatbot Recommendations** (2024)
   — `stereotype_or_personalization_user_identity_biases.pdf`
   - Why: directly shows recommendations shift with user identity cues; the central design we adapt.
2. **Exploring Safety-Utility Trade-Offs in Personalized Language Models** (2024, arXiv:2406.11107)
   — `safety_utility_tradeoffs_personalized_lm.pdf`
   - Why: identity cues change *safety* behavior (over-/under-cautious) — exactly the "go to sleep = protective advice" axis.
3. **ChatGPT Doesn't Trust Chargers Fans: Guardrail Sensitivity in Context** (2024)
   — `guardrail_sensitivity_chargers_fans.pdf`
   - Why: shows guardrails/refusals vary with user persona attributes (age, gender, politics).
4. **Language Models Change Facts Based on the Way You Talk** (2025)
   — `lms_change_facts_based_on_how_you_talk.pdf`
   - Why: even factual answers (incl. medical) shift with user identity markers in the message.
5. **First-Person Fairness in Chatbots** (OpenAI, Eloundou et al. 2024, arXiv:2410.19803)
   — `first_person_fairness_in_chatbots.pdf`
   - Why: large-scale name/identity counterfactual audit of *who the user is* — methodological backbone.
6. **One Persona, Many Cues, Different Results** (2026)
   — `one_persona_many_cues_different_results.pdf`
   - Why: how the *form* of the identity cue changes the personalization effect (robustness of our probe).
7. **Different Demographic Cues Yield Inconsistent Conclusions About LLM Personalization and Bias** (2026)
   — `different_demographic_cues_inconsistent_conclusions.pdf`
   - Why: warns that conditioning effects depend on cue choice — informs our multi-cue design.
8. **Reading Between the Prompts: How Stereotypes Shape LLM's Implicit Personalization** (2025)
   — `reading_between_the_prompts_implicit_personalization.pdf`
   - Why: models infer identity *implicitly* and personalize even without explicit demographic labels.
9. **Position is Power: System Prompts as a Mechanism of Bias in LLMs** (2025, arXiv:2505.21091)
   — `position_is_power_system_prompts_bias.pdf`
   - Why: where the identity cue sits (system vs user) modulates conditioning.

## B. Foundations — persona conditioning, sycophancy, opinion reflection

10. **Towards Understanding Sycophancy in Language Models** (Anthropic, 2023, arXiv:2310.13548)
    — `towards_understanding_sycophancy.pdf`
    - Why: canonical case of response conditioned on the *user* rather than the truth/situation.
11. **Whose Opinions Do Language Models Reflect?** (Santurkar et al. 2023, arXiv:2303.17548)
    — `whose_opinions_do_lms_reflect.pdf`
    - Why: steerability — prepend a persona, outputs move toward that group. Code in `code/opinions_qa`.
12. **Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs** (2023, arXiv:2311.04892)
    — `bias_runs_deep_persona_assigned_llms.pdf`
    - Why: assigning a persona degrades/changes reasoning — a strong person-conditioning signal.
13. **Evaluating Large Language Model Biases in Persona-Steered Generation** (2024, arXiv:2405.20253)
    — `persona_steered_generation_biases.pdf`
    - Why: quantifies how steerable/biased generation becomes under persona conditioning.
14. **Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization** (2024)
    — `two_tales_of_persona_survey.pdf`
    - Why: survey framing of persona (who the model is) vs personalization (who the user is) — situates the hypothesis.

## C. Emotional / wellbeing advice conditioned on the user

15. **Are LLMs Empathetic to All? Influence of Multi-Demographic Personas on Empathy** (2025)
    — `are_llms_empathetic_to_all_multidemographic.pdf`
    - Why: empathy (akin to "go to sleep, take care of yourself") varies by user demographics.
16. **The Personalization Trap: How User Memory Alters Emotional Reasoning in LLMs** (2025)
    — `personalization_trap_user_memory_emotional.pdf`
    - Why: stored user attributes change *emotional* responses — central to caring/protective advice.
17. **Personalization Increases Affective Alignment but Has Role-Dependent Effects on Epistemic Independence** (2026)
    — `personalization_affective_alignment_epistemic.pdf`
    - Why: personalization makes the model emotionally agree more — relevant to comfort/sleep advice.

## D. The "go to sleep" domain + vulnerable users

18. **Towards a Personal Health Large Language Model** (PH-LLM, sleep & fitness, 2024, arXiv:2406.06474)
    — `towards_personal_health_llm.pdf`
    - Why: the canonical "LLM gives sleep advice" system; sets up what sleep recommendations look like.
    - Note: substitute/preprint for the paywalled Nature 2025 *"A personal health LLM for sleep and fitness coaching"* (same group/benchmarks).
19. **LLM Safety for Children** (2025)
    — `llm_safety_for_children.pdf`
    - Why: models treat a "child" user differently — a clear, intended form of person-conditioning relevant to protective "go to sleep" advice.

## Could not download (documented for completeness)
- **A personal health large language model for sleep and fitness coaching** (Nature, 2025; CorpusID:280661855) — paywalled; the arXiv preprint #18 (PH-LLM) covers the same content.
- **A Scoping Review of LLMs in Personal Sleep Wellness** (2025; CorpusID:282657347) — no open PDF located; abstract synthesized in `../literature_review.md` (4 use cases: educational QA, condition-specific support, personalized recommendations/coaching, CBT-I self-help).

See `_download_log.json` for raw resolution results.
