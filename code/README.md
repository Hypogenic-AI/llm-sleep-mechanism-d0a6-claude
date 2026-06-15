# Cloned Repositories

Code resources supporting the study of whether LLMs give "go to sleep" / wellbeing
advice **uniformly (situation-driven)** or **conditioned on the perceived user
(person-driven)**.

---

## Repo 1: `sycophancy-eval/` — methodological baseline for behavior-change probes

- **URL**: https://github.com/meg-tong/sycophancy-eval
- **Paper**: *Towards Understanding Sycophancy in Language Models* (Sharma et al., Anthropic, 2023) — arXiv:2310.13548
- **Location**: `code/sycophancy-eval/`
- **Purpose**: Datasets + a minimal inference harness for measuring how an LLM's
  free-form output *changes* as a function of how the user frames a request — the
  closest existing methodology to our question. Sycophancy is exactly a
  *person-conditioned* deviation (the model tells the user what the user seems to
  want). Useful as a **template** for our probe.
- **Key files**:
  - `utils.py` — `inference()` / `async_inference()` over LangChain model wrappers;
    `load_from_jsonl()`. We mirror its **prompt JSONL format**
    (`{"prompt":[{"type":"human","content":...}], "base":..., "metadata":...}`)
    in our `datasets/sleep_persona_probe`.
  - `datasets/answer.jsonl`, `are_you_sure.jsonl`, `feedback.jsonl` — prompt sets
    where user framing is varied and the response shift is measured.
  - `example.ipynb` — end-to-end example of running a dataset and scoring shifts.
- **How we use it**: adopt the prompt/IO format and the "vary the framing, measure
  the response shift" evaluation pattern; swap in our (scenario × persona) probe.
- **Requirements**: `langchain`, an LLM API key (OpenAI/Anthropic). Not installed
  here — the experiment runner can `uv add langchain langchain-openai` if reused.

## Repo 2: `opinions_qa/` — persona-steerability methodology + persona axes

- **URL**: https://github.com/tatsu-lab/opinions_qa
- **Paper**: *Whose Opinions Do Language Models Reflect?* (Santurkar et al., 2023) — arXiv:2303.17548
- **Location**: `code/opinions_qa/`
- **Purpose**: Establishes **steerability** methodology — prepend a persona/identity
  context and measure how model outputs shift toward that group. Directly relevant
  to the "person-conditioned" half of our hypothesis, and a source of demographic
  persona axes (age, gender, region, politics, etc.) drawn from Pew ATP surveys.
- **Key files**:
  - `steerability.ipynb` — measures whether persona conditioning moves outputs
    toward the targeted group (our "person effect" idea, applied to opinions).
  - `consistency.ipynb`, `representativeness.ipynb`, `refusals.ipynb` — related axes.
  - `helpers.py`, `process_results.ipynb` — distribution computation utilities.
- **Note**: The OpinionQA *data* (1498 Pew-based MC questions + human responses) is
  hosted on CodaLab, not in the repo — download separately if needed:
  https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69
- **How we use it**: borrow persona-steering framing and demographic axes; the
  steerability metric is analogous to our per-factor "person effect" test.

---

## Not cloned (no public code located)

- *Stereotype or Personalization?*, *Exploring Safety-Utility Trade-Offs*,
  *ChatGPT Doesn't Trust Chargers Fans*, *First-Person Fairness in Chatbots* — these
  define the **counterfactual identity-audit** design we adopt, but did not ship a
  reusable public repo at time of writing. Their methodology is summarized in
  `../literature_review.md`. Our `datasets/sleep_persona_probe/generate_probe.py`
  re-implements the core counterfactual-audit pattern from scratch.

## Suggested experiment harness (for the experiment runner)

1. Load `datasets/sleep_persona_probe/prompts.jsonl`.
2. Query 1+ chat LLMs per `prompt` (reuse `sycophancy-eval/utils.py` IO pattern).
3. Classify each response `advises_sleep ∈ {0,1}` (keyword + LLM-judge).
4. Decompose situation effect vs. person effect (see `datasets/README.md`).
5. (Optional) Borrow `opinions_qa/steerability.ipynb` framing for cross-model
   consistency of the conditioning.
