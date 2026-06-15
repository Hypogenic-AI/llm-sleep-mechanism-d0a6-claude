#!/usr/bin/env python3
"""Retry the rate-limited papers with longer backoff + known arXiv fallbacks."""
import json, os, time, urllib.request, urllib.error
OUT="papers"; UA={"User-Agent":"Mozilla/5.0 (research)"}

# slug -> list of candidate direct arXiv ids (tried first, fast, no rate limit)
ARXIV = {
    "safety_utility_tradeoffs_personalized_lm": ["2406.11107"],
    "position_is_power_system_prompts_bias":    ["2505.21091"],
    "whose_opinions_do_lms_reflect":            ["2303.17548"],
    "bias_runs_deep_persona_assigned_llms":     ["2311.04892"],
    "persona_steered_generation_biases":        ["2405.20253"],
    # ph_llm sleep/fitness is Nature (no free PDF); substitute the arXiv preprint
    # "Towards a Personal Health LLM" which is the same group/benchmarks.
    "towards_personal_health_llm":              ["2406.06474"],
}

def dl(url, path):
    req=urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r: data=r.read()
    if data[:4]!=b"%PDF": return False,len(data)
    open(path,"wb").write(data); return True,len(data)

for slug, ids in ARXIV.items():
    path=os.path.join(OUT,slug+".pdf")
    if os.path.exists(path) and os.path.getsize(path)>10000:
        print("SKIP",slug); continue
    done=False
    for aid in ids:
        try:
            ok,n=dl(f"https://arxiv.org/pdf/{aid}.pdf", path)
            if ok: print(f"OK   {slug}  {n//1024}KB  arXiv:{aid}"); done=True; break
            else: print("NOTPDF",slug,aid,n)
        except Exception as e: print("FAIL",slug,aid,e)
        time.sleep(2)
    if not done: print("UNRESOLVED",slug)
