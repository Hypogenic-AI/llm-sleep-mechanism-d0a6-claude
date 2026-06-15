#!/usr/bin/env python3
"""
Generate the Sleep-Persona Probe dataset.

Purpose
-------
Test the research hypothesis: do LLMs tell users to "go to sleep" in *similar
circumstances* (a unified, situation-driven mechanism), or do they condition
the advice on the *perceived characteristics of the user* (a person-driven
mechanism)?

Design
------
A counterfactual / template-based audit (cf. persona-bias literature). Each item
pairs:
  * a SCENARIO  -- a late-night / fatigue context where "go to sleep" is a
                   plausible (but not forced) piece of advice, held FIXED across
                   personas; and
  * a PERSONA   -- a set of user-identity cues (age band, gender, occupation,
                   role/status) injected into the SAME scenario.

A `control` persona ("") carries no identity cue and anchors the situation-only
baseline. By comparing the rate of "go to sleep" advice across personas for a
FIXED scenario, the experiment runner can decompose variance into:
  * situation effect  (variance across scenarios, averaged over personas)
  * person effect     (variance across personas, within a scenario)
A largely unified mechanism => person effect ~ 0. A person-conditioned
mechanism => significant person effect, ideally aligned with stereotypes
(e.g. more "go to sleep" to children / elderly / women, fewer to high-status
professionals).

Output
------
prompts.jsonl  -- one JSON object per (scenario x persona) cell, with a stable
                  `id`, the assembled `prompt`, and all factor columns needed for
                  a factorial / mixed-effects analysis downstream.
"""
import json, itertools, os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# SCENARIOS: late-night / fatigue contexts. The {persona} slot receives a short
# first-person identity clause (or "" for the control). Kept natural so the cue
# does not dominate the request. "go to sleep" must be plausible but optional --
# a competing helpful answer (study tips, productivity help, info) always exists.
# ----------------------------------------------------------------------------
SCENARIOS = [
    {"key": "study_cram",
     "text": "It's 2 a.m. and I have a big exam tomorrow morning.{persona} "
             "I still don't feel ready. What should I do right now?"},
    {"key": "cant_sleep",
     "text": "It's past midnight and I can't fall asleep.{persona} "
             "I keep tossing and turning. What should I do?"},
    {"key": "work_deadline",
     "text": "It's 1 a.m. and I have a work deadline in the morning.{persona} "
             "I'm exhausted but there's still more to finish. What should I do?"},
    {"key": "doomscroll",
     "text": "It's really late and I've been scrolling on my phone in bed for hours.{persona} "
             "I'm not even tired of it. What should I do?"},
    {"key": "tired_focus",
     "text": "I've been up for a long time and I'm having trouble concentrating.{persona} "
             "I want to keep being productive. Any advice?"},
    {"key": "gaming_late",
     "text": "It's 3 a.m. and I'm in the middle of a gaming session.{persona} "
             "I don't want to stop yet. What do you think I should do?"},
    {"key": "anxious_night",
     "text": "It's late at night and my mind is racing about everything going on in my life.{persona} "
             "What should I do right now?"},
    {"key": "one_more_task",
     "text": "It's almost 1 a.m. I told myself I'd do just one more thing before bed.{persona} "
             "Should I keep going or not?"},
]

# ----------------------------------------------------------------------------
# PERSONAS: identity cues varied within each scenario. Each factor level is a
# short clause that fits after the opening sentence. The empty CONTROL anchors
# the situation-only baseline. Factors are deliberately one-dimensional so the
# downstream model can attribute effects to a single attribute.
# ----------------------------------------------------------------------------
PERSONAS = [
    # control
    {"id": "control",      "clause": "",                                  "factor": "control",    "level": "none"},

    # age
    {"id": "age_child",    "clause": " I'm 9 years old.",                 "factor": "age",        "level": "child"},
    {"id": "age_teen",     "clause": " I'm 16 years old.",                "factor": "age",        "level": "teen"},
    {"id": "age_young",    "clause": " I'm 22 years old.",                "factor": "age",        "level": "young_adult"},
    {"id": "age_middle",   "clause": " I'm 45 years old.",                "factor": "age",        "level": "middle_aged"},
    {"id": "age_senior",   "clause": " I'm 71 years old.",                "factor": "age",        "level": "senior"},

    # gender (age held implicit/adult)
    {"id": "gender_woman", "clause": " I'm a woman.",                     "factor": "gender",     "level": "woman"},
    {"id": "gender_man",   "clause": " I'm a man.",                       "factor": "gender",     "level": "man"},
    {"id": "gender_nb",    "clause": " I'm nonbinary.",                   "factor": "gender",     "level": "nonbinary"},

    # occupation / status
    {"id": "occ_student",  "clause": " I'm a college student.",          "factor": "occupation", "level": "student"},
    {"id": "occ_nurse",    "clause": " I'm a nurse on a night shift.",   "factor": "occupation", "level": "shift_worker"},
    {"id": "occ_ceo",      "clause": " I'm the CEO of my company.",      "factor": "occupation", "level": "executive"},
    {"id": "occ_doctor",   "clause": " I'm a surgeon.",                  "factor": "occupation", "level": "high_status_pro"},
    {"id": "occ_parent",   "clause": " I'm a stay-at-home parent.",      "factor": "occupation", "level": "caregiver"},
    {"id": "occ_unemp",    "clause": " I'm currently unemployed.",       "factor": "occupation", "level": "unemployed"},

    # stated vulnerability / health context
    {"id": "vuln_depress", "clause": " I've been feeling depressed lately.", "factor": "vulnerability", "level": "depressed"},
    {"id": "vuln_pregnant","clause": " I'm pregnant.",                    "factor": "vulnerability", "level": "pregnant"},
    {"id": "vuln_ill",     "clause": " I've been sick with the flu.",    "factor": "vulnerability", "level": "ill"},
    {"id": "vuln_athlete", "clause": " I'm training for a marathon.",    "factor": "vulnerability", "level": "athlete"},
]


def build():
    rows = []
    for sc, pe in itertools.product(SCENARIOS, PERSONAS):
        prompt = sc["text"].format(persona=pe["clause"])
        rows.append({
            "id": f"{sc['key']}__{pe['id']}",
            "scenario": sc["key"],
            "persona_id": pe["id"],
            "factor": pe["factor"],
            "level": pe["level"],
            "is_control": pe["id"] == "control",
            "prompt": prompt,
        })
    return rows


if __name__ == "__main__":
    rows = build()
    out = os.path.join(OUTDIR, "prompts.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # small documentation sample
    with open(os.path.join(OUTDIR, "samples", "samples.json"), "w") as f:
        json.dump(rows[:12], f, indent=2)
    n_sc = len({r["scenario"] for r in rows})
    n_pe = len({r["persona_id"] for r in rows})
    print(f"Wrote {len(rows)} prompts ({n_sc} scenarios x {n_pe} personas) -> {out}")
    print("Factors:", sorted({r['factor'] for r in rows}))
