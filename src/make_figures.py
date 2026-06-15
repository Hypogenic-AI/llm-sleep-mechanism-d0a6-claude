"""Generate figures from labeled.jsonl + analysis.json."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

sns.set_theme(style="whitegrid", context="talk")
LAB = os.path.join(C.RESULTS_DIR, "labeled.jsonl")
AN = os.path.join(C.RESULTS_DIR, "analysis.json")

PERSONA_ORDER = ["control", "age_child", "age_teen", "age_young", "age_middle", "age_senior",
                 "gender_woman", "gender_man", "gender_nb", "occ_student", "occ_nurse",
                 "occ_ceo", "occ_doctor", "occ_parent", "occ_unemp",
                 "vuln_depress", "vuln_pregnant", "vuln_ill", "vuln_athlete"]
SCEN_ORDER = ["cant_sleep", "anxious_night", "doomscroll", "tired_focus", "study_cram",
              "work_deadline", "one_more_task", "gaming_late"]


def load():
    df = pd.DataFrame(json.loads(l) for l in open(LAB) if l.strip())
    df = df[df["advises_sleep_judge"].notna()].copy()
    df["advises_sleep"] = df["advises_sleep_judge"].astype(int)
    df["strength"] = pd.to_numeric(df["strength"], errors="coerce")
    return df


def fig_scenario_persona_heatmap(df):
    piv = df.pivot_table(index="persona_id", columns="scenario", values="advises_sleep", aggfunc="mean")
    piv = piv.reindex(index=[p for p in PERSONA_ORDER if p in piv.index],
                      columns=[s for s in SCEN_ORDER if s in piv.columns])
    plt.figure(figsize=(13, 10))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1,
                cbar_kws={"label": "P(advises sleep)"}, linewidths=.4)
    plt.title("P(advises sleep): scenario (situation) x persona (person)\npooled over 5 models, 3 reps")
    plt.ylabel("Persona"); plt.xlabel("Scenario")
    plt.tight_layout(); plt.savefig(os.path.join(C.FIG_DIR, "heatmap_scenario_persona.png"), dpi=140)
    plt.close()


def fig_variance_decomp(an):
    vd = an["variance_decomposition"]
    label_map = {"C(scenario)": "Scenario\n(situation)", "C(level2)": "Persona factor\n(person)",
                 "C(model_label)": "Model", "C(scenario):C(level2)": "Scenario x Persona\n(interaction)"}
    terms = [t for t in label_map if t in vd]
    vals = [vd[t]["omega_sq"] for t in terms]
    plt.figure(figsize=(9, 6))
    colors = ["#2c7fb8", "#d95f0e", "#999999", "#7fcdbb"]
    bars = plt.bar([label_map[t] for t in terms], vals, color=colors[:len(terms)])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v*100:.1f}%", ha="center", fontsize=13)
    plt.ylabel(r"Variance explained ($\omega^2$)")
    plt.title("How unified? Variance in 'go to sleep' behavior\nattributable to situation vs person")
    plt.tight_layout(); plt.savefig(os.path.join(C.FIG_DIR, "variance_decomposition.png"), dpi=140)
    plt.close()


def fig_persona_effect(an):
    rd = an["risk_diff_vs_control"]
    items = [(lv, d["risk_diff_vs_control"], d["ci95"], d["factor"]) for lv, d in rd.items()]
    order = ["age_child", "age_teen", "age_young", "age_middle", "age_senior",
             "gender_woman", "gender_man", "gender_nb", "occ_student", "occ_nurse",
             "occ_ceo", "occ_doctor", "occ_parent", "occ_unemp",
             "vuln_depress", "vuln_pregnant", "vuln_ill", "vuln_athlete"]
    items = sorted(items, key=lambda x: order.index(x[0]) if x[0] in order else 99)
    labels = [i[0].replace("age_", "").replace("gender_", "").replace("occ_", "").replace("vuln_", "") for i in items]
    pts = [i[1] for i in items]
    lo = [i[1] - i[2][0] for i in items]; hi = [i[2][1] - i[1] for i in items]
    fac = [i[3] for i in items]
    cmap = {"age": "#2c7fb8", "gender": "#756bb1", "occupation": "#31a354", "vulnerability": "#d95f0e"}
    cols = [cmap.get(f, "#888") for f in fac]
    plt.figure(figsize=(12, 7))
    y = np.arange(len(items))
    plt.barh(y, pts, xerr=[lo, hi], color=cols, capsize=3)
    plt.axvline(0, color="k", lw=1)
    plt.yticks(y, labels); plt.gca().invert_yaxis()
    plt.xlabel("Risk difference in P(advises sleep) vs no-persona control\n(scenario-paired, 95% bootstrap CI)")
    plt.title("Person effect: which personas get more/less 'go to sleep'")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
    plt.legend(handles, cmap.keys(), title="factor", fontsize=11, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(C.FIG_DIR, "persona_effect_vs_control.png"), dpi=140)
    plt.close()


def fig_cross_model(df, an):
    cell = df.groupby(["model_label", "persona_id"])["advises_sleep"].mean().reset_index()
    piv = cell.pivot_table(index="persona_id", columns="model_label", values="advises_sleep")
    piv = piv.reindex([p for p in PERSONA_ORDER if p in piv.index])
    plt.figure(figsize=(11, 9))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="mako_r", vmin=0, vmax=1,
                cbar_kws={"label": "P(advises sleep)"}, linewidths=.4)
    plt.title("Per-persona P(advises sleep) by model\n(do models agree on who to tell to sleep?)")
    plt.ylabel("Persona"); plt.xlabel("Model")
    plt.tight_layout(); plt.savefig(os.path.join(C.FIG_DIR, "cross_model_persona.png"), dpi=140)
    plt.close()

    # scenario profile by model
    sc = df.groupby(["model_label", "scenario"])["advises_sleep"].mean().reset_index()
    pivs = sc.pivot_table(index="scenario", columns="model_label", values="advises_sleep")
    pivs = pivs.reindex([s for s in SCEN_ORDER if s in pivs.index])
    plt.figure(figsize=(11, 7))
    for m in pivs.columns:
        plt.plot(pivs.index, pivs[m], marker="o", label=m)
    plt.xticks(rotation=35, ha="right"); plt.ylabel("P(advises sleep)")
    plt.title("Situation effect is shared across models")
    plt.legend(fontsize=11); plt.tight_layout()
    plt.savefig(os.path.join(C.FIG_DIR, "scenario_profile_by_model.png"), dpi=140)
    plt.close()


def main():
    df = load(); an = json.load(open(AN))
    fig_scenario_persona_heatmap(df)
    fig_variance_decomp(an)
    fig_persona_effect(an)
    fig_cross_model(df, an)
    print("Figures written to", C.FIG_DIR)
    print(os.listdir(C.FIG_DIR))


if __name__ == "__main__":
    main()
