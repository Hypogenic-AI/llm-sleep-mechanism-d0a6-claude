# How Unified Is the LLM "Go to Sleep" Mechanism?

A counterfactual identity audit of whether LLMs tell users to "go to sleep" based on the
**situation** they're in or **who they appear to be** — run on **2,280 real API responses**
(8 late-night scenarios × 19 personas × 5 frontier models × 3 replicates).

## Key findings

- **Mostly situation-driven, within a model.** The late-night *scenario* explains ~3.4× more
  behavioral variance than the user's *persona* (ω² 0.110 vs 0.032; robust 3–9× across 4
  outcome definitions). P(clearly advises sleep) ranges 0.42→0.91 across scenarios.
- **The biggest determinant is *which model* you ask.** P(clearly advises sleep) goes from
  **0.15 (Gemini-3.5-Flash) to 0.82 (GPT-5.4-mini)** (ω²_model = 0.26). Models agree only
  moderately on *who* to tell to sleep (mean cross-model r = 0.36); Gemini is a structural outlier.
- **A real but small person effect — and *not* a demographic stereotype.** Significant for
  occupation/vulnerability/age (Cramér's V ≈ 0.14–0.19), not gender. It tracks **bodily stakes
  & role legitimacy**, not protective vulnerability: athletes (+0.18), the ill (+0.15),
  children/teens get *more* sleep advice; **night-shift nurses get less** (−0.15, their
  wakefulness is legitimized). The naïve "protect the vulnerable" prediction was refuted
  (Δ=−0.055, p=0.35, wrong direction).
- **Verdict:** the go-to-sleep mechanism is **partially unified**, with a clear driver
  hierarchy: **model ≫ situation > person.**

See **[REPORT.md](REPORT.md)** for the full analysis, figures, and limitations.

## Reproduce

```bash
source .venv/bin/activate          # uv-managed env; deps in pyproject.toml
export OPENROUTER_KEY=...           # required
python src/collect_responses.py    # E1: 2,280 generations (resumable cache) ~29 min, ~$4.5
python src/classify.py             # E2: dual labeling (regex + LLM judge)    ~5 min,  ~$1.3
python src/analyze.py              # E3/E4: stats -> results/analysis.json
python src/make_figures.py         # figures -> figures/*.png
```
All steps are cached/resumable and seeded (seed=42). Total API cost ≈ **$5.75**.

## File structure

```
planning.md                     Phase-0/1 plan: motivation, hypotheses, analysis plan
REPORT.md                       Primary deliverable: full research report
src/
  config.py                     Model panel, params, paths (model IDs verified live)
  collect_responses.py          Async, retrying, cached generation harness
  classify.py                   Regex classifier + LLM judge (strength 0–3)
  analyze.py                    Variance decomposition, χ², GEE, bootstrap, cross-model
  make_figures.py               Heatmaps, variance bars, persona-effect forest plot
results/
  model_outputs/raw_responses.jsonl   2,280 raw responses (+token usage)
  labeled.jsonl                 Per-response labels (regex, judge binary, strength, refusal)
  analysis.json                 All numeric findings + sensitivity
  cell_means.csv                Per model × scenario × persona P(sleep)
  qualitative_examples.json     Illustrative response excerpts
figures/                        5 PNGs used in REPORT.md
datasets/sleep_persona_probe/   Counterfactual probe (152 prompts) + generator
literature_review.md, resources.md   Pre-gathered background (19 papers)
```

## What was tested
Real LLMs via OpenRouter (no simulation): `openai/gpt-5.4-mini`,
`anthropic/claude-sonnet-4.6`, `google/gemini-3.5-flash`,
`meta-llama/llama-3.3-70b-instruct`, `deepseek/deepseek-v3.2`. Judge: `gpt-5.4-mini` (T=0).
