#!/usr/bin/env python3
"""Resolve Semantic Scholar / arXiv IDs to open-access PDFs and download to papers/."""
import json, os, time, urllib.request, urllib.error

OUT = "papers"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research paper finder)"}

# (slug, identifier) ; identifier is either "CorpusID:xxx", "ss:<paperId>", or "arxiv:xxxx.xxxxx"
PAPERS = [
    # --- Core: persona / demographic conditioning of LLM behavior ---
    ("stereotype_or_personalization_user_identity_biases", "CorpusID:273228304"),
    ("safety_utility_tradeoffs_personalized_lm",            "CorpusID:270560229"),
    ("guardrail_sensitivity_chargers_fans",                "CorpusID:271064842"),
    ("lms_change_facts_based_on_how_you_talk",              "CorpusID:280271389"),
    ("one_persona_many_cues_different_results",             "CorpusID:285051247"),
    ("different_demographic_cues_inconsistent_conclusions", "CorpusID:285050261"),
    ("reading_between_the_prompts_implicit_personalization","CorpusID:278788896"),
    ("position_is_power_system_prompts_bias",              "CorpusID:278910677"),
    ("whose_opinions_do_lms_reflect",                      "ss:e38a29f6463f38f43797b128673b9e44d18a991e"),
    ("two_tales_of_persona_survey",                        "CorpusID:270212889"),
    ("are_llms_empathetic_to_all_multidemographic",        "CorpusID:282058037"),
    ("personalization_trap_user_memory_emotional",         "CorpusID:282057135"),
    ("bias_runs_deep_persona_assigned_llms",               "CorpusID:265050702"),
    ("persona_steered_generation_biases",                  "CorpusID:270123737"),
    ("personalization_affective_alignment_epistemic",      "ss:c7af894a86375750c0975d1642a74177da9343a5"),
    # --- Foundational baselines (manual arXiv) ---
    ("first_person_fairness_in_chatbots",                  "arxiv:2410.19803"),
    ("towards_understanding_sycophancy",                   "arxiv:2310.13548"),
    # --- Sleep / wellbeing domain (the "go to sleep" content) ---
    ("ph_llm_sleep_fitness_coaching",                      "CorpusID:280661855"),
    ("scoping_review_llms_personal_sleep_wellness",        "CorpusID:282657347"),
    # --- Safety / vulnerability ---
    ("llm_safety_for_children",                            "CorpusID:276421409"),
]

def fetch_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

def resolve_pdf(ident):
    """Return (pdf_url, meta) or (None, meta)."""
    if ident.startswith("arxiv:"):
        aid = ident.split(":",1)[1]
        return f"https://arxiv.org/pdf/{aid}.pdf", {"arxiv": aid}
    if ident.startswith("CorpusID:"):
        api = f"https://api.semanticscholar.org/graph/v1/paper/{ident}"
    else:  # ss:<paperId>
        api = f"https://api.semanticscholar.org/graph/v1/paper/{ident.split(':',1)[1]}"
    api += "?fields=title,year,externalIds,openAccessPdf"
    for attempt in range(5):
        try:
            d = fetch_json(api)
            ext = d.get("externalIds") or {}
            oa = d.get("openAccessPdf") or {}
            pdf = oa.get("url")
            if not pdf and ext.get("ArXiv"):
                pdf = f"https://arxiv.org/pdf/{ext['ArXiv']}.pdf"
            return pdf, {"title": d.get("title"), "year": d.get("year"), "externalIds": ext}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(3*(attempt+1)); continue
            return None, {"error": str(e)}
        except Exception as e:
            time.sleep(2);
            if attempt==4: return None, {"error": str(e)}
    return None, {"error": "rate-limited"}

def download(url, path):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if data[:4] != b"%PDF":
        return False, len(data)
    with open(path, "wb") as f:
        f.write(data)
    return True, len(data)

results = []
for slug, ident in PAPERS:
    path = os.path.join(OUT, slug + ".pdf")
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        results.append((slug, "exists", os.path.getsize(path))); print("SKIP exists", slug); continue
    pdf, meta = resolve_pdf(ident)
    if not pdf:
        results.append((slug, "no-pdf", meta)); print("NO-PDF", slug, meta);
        if ident.startswith("CorpusID") or ident.startswith("ss:"): time.sleep(1.2)
        continue
    try:
        ok, n = download(pdf, path)
        if ok:
            results.append((slug, "ok", n)); print(f"OK    {slug}  {n//1024}KB  {pdf}")
        else:
            results.append((slug, "not-pdf", pdf)); print("NOTPDF", slug, pdf)
    except Exception as e:
        results.append((slug, "dl-fail", str(e))); print("DLFAIL", slug, e)
    if not ident.startswith("arxiv:"): time.sleep(1.2)

print("\n==== SUMMARY ====")
for r in results: print(r[0], r[1], r[2] if len(r)>2 else "")
ok = sum(1 for r in results if r[1] in ("ok","exists"))
print(f"\nDownloaded/present: {ok}/{len(PAPERS)}")
json.dump(results, open("papers/_download_log.json","w"), indent=2, default=str)
