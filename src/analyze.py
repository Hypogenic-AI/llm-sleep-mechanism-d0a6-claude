"""E3/E4: Analysis of the go-to-sleep persona audit.

Consumes results/labeled.jsonl and produces:
  - results/analysis.json         (all numeric findings)
  - results/cell_means.csv        (per model x scenario x persona P(sleep))
  - figures/*.png                 (heatmaps, variance decomposition, etc.)

Key analyses:
  1. Labeler agreement (Cohen's kappa: regex vs LLM judge).
  2. Variance decomposition (ANOVA omega^2): scenario vs persona-factor vs model vs interaction.
  3. Situation effect (between-scenario) vs person effect (between-persona within scenario).
  4. Per-factor chi-square + Cramer's V; per-level risk difference vs control + bootstrap CI.
  5. Mixed/clustered logistic (GEE) person-effect estimates.
  6. Stereotype-direction contrast.
  7. Cross-model agreement of the conditioning pattern.
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CFG

rng = np.random.default_rng(42)
LAB_PATH = os.path.join(CFG.RESULTS_DIR, "labeled.jsonl")
OUT_JSON = os.path.join(CFG.RESULTS_DIR, "analysis.json")

# Stereotype expectation: which personas "should" get MORE protective rest-advice
# under a person-driven mechanism (vulnerable/young/old), vs LESS (high-status/robust).
MORE_REST = {"child", "teen", "senior", "depressed", "pregnant", "ill", "caregiver", "woman"}
LESS_REST = {"executive", "high_status_pro", "athlete", "man"}


def load():
    rows = [json.loads(l) for l in open(LAB_PATH) if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df["advises_sleep_judge"].notna()].copy()
    df["strength"] = pd.to_numeric(df["strength"], errors="coerce")
    # Four outcome definitions (see manual judge-validation in REPORT):
    #   strong_sleep (PRIMARY): graded strength >= 2 ("clearly recommends sleeping")
    #   insist_sleep (strict):  strength == 3 ("strongly/insistently urges sleep")
    #   any_sleep (lenient):    judge binary -- any rest/sleep mention as advice
    #   regex (lexical):        deterministic keyword match of explicit phrasing
    df["strong_sleep"] = (df["strength"] >= 2).astype(int)
    df["insist_sleep"] = (df["strength"] == 3).astype(int)
    df["any_sleep"] = df["advises_sleep_judge"].astype(int)
    df["regex"] = pd.to_numeric(df["advises_sleep_kw"], errors="coerce")
    df["advises_sleep"] = df["strong_sleep"]   # PRIMARY outcome
    return df


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = (~pd.isna(a)) & (~pd.isna(b))
    a, b = a[m].astype(int), b[m].astype(int)
    po = np.mean(a == b)
    cats = [0, 1]
    pe = sum((np.mean(a == c)) * (np.mean(b == c)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, int(m.sum())


def omega_sq_anova(df):
    """Variance decomposition via OLS ANOVA on the binary outcome.
    Returns omega^2 (unbiased) for each component."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = df.copy()
    d["scenario"] = d["scenario"].astype("category")
    d["level2"] = d["level"].astype("category")   # persona level (incl 'none')
    d["model_label"] = d["model_label"].astype("category")
    model = smf.ols("advises_sleep ~ C(scenario) + C(level2) + C(model_label) + C(scenario):C(level2)", data=d).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_total = aov["sum_sq"].sum()
    ms_res = aov.loc["Residual", "sum_sq"] / aov.loc["Residual", "df"]
    out = {}
    for term in aov.index:
        if term == "Residual":
            continue
        ss = aov.loc[term, "sum_sq"]; dfree = aov.loc[term, "df"]
        omega = (ss - dfree * ms_res) / (ss_total + ms_res)
        eta = ss / ss_total
        out[term] = {"eta_sq": float(eta), "omega_sq": float(max(omega, 0.0)),
                     "F": float(aov.loc[term, "F"]), "p": float(aov.loc[term, "PR(>F)"])}
    return out


def per_factor_chi2(df):
    res = {}
    for factor in ["age", "gender", "occupation", "vulnerability"]:
        sub = df[(df["factor"] == factor) | (df["is_control"])]
        ct = pd.crosstab(sub["level"], sub["advises_sleep"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        cramer = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        res[factor] = {"chi2": float(chi2), "p": float(p), "dof": int(dof),
                       "cramers_v": float(cramer), "n": int(n)}
    return res


def boot_risk_diff(df):
    """Per-level risk difference vs control, paired by scenario, bootstrap 95% CI."""
    out = {}
    ctrl = df[df["is_control"]].groupby("scenario")["advises_sleep"].mean()
    for level in sorted(df.loc[~df["is_control"], "level"].unique()):
        sub = df[df["level"] == level]
        by_scen = sub.groupby("scenario")["advises_sleep"].mean()
        common = by_scen.index.intersection(ctrl.index)
        diffs = (by_scen[common] - ctrl[common]).values
        if len(diffs) == 0:
            continue
        point = float(np.mean(diffs))
        boots = [float(np.mean(rng.choice(diffs, len(diffs), replace=True))) for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        factor = sub["factor"].iloc[0]
        out[level] = {"factor": factor, "risk_diff_vs_control": point,
                      "ci95": [float(lo), float(hi)], "n_scenarios": int(len(diffs))}
    return out


def gee_logistic(df):
    """Population-averaged logistic with scenario as cluster; person effect via levels."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = df.copy()
    d["level2"] = pd.Categorical(d["level"], categories=["none"] + sorted([x for x in d["level"].unique() if x != "none"]))
    try:
        mod = smf.gee("advises_sleep ~ C(level2)", groups="scenario", data=d,
                      family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
        r = mod.fit()
        coefs = {}
        for name, b, p, se in zip(r.params.index, r.params.values, r.pvalues.values, r.bse.values):
            coefs[name] = {"coef": float(b), "se": float(se), "p": float(p), "odds_ratio": float(np.exp(b))}
        return coefs
    except Exception as e:
        return {"error": str(e)[:200]}


def stereotype_contrast(df):
    """Mean P(sleep) for MORE_REST personas vs LESS_REST personas (within-scenario centered)."""
    ctrl = df[df["is_control"]].groupby("scenario")["advises_sleep"].mean()
    def centered(level):
        sub = df[df["level"] == level]
        by = sub.groupby("scenario")["advises_sleep"].mean()
        common = by.index.intersection(ctrl.index)
        return (by[common] - ctrl[common]).mean()
    more = {lv: centered(lv) for lv in MORE_REST if lv in df["level"].unique()}
    less = {lv: centered(lv) for lv in LESS_REST if lv in df["level"].unique()}
    more_vals = np.array(list(more.values())); less_vals = np.array(list(less.values()))
    t, p = stats.ttest_ind(more_vals, less_vals, equal_var=False)
    pooled = np.sqrt((more_vals.var(ddof=1) + less_vals.var(ddof=1)) / 2) if len(more_vals) > 1 and len(less_vals) > 1 else np.nan
    d = (more_vals.mean() - less_vals.mean()) / pooled if pooled and pooled > 0 else np.nan
    return {"more_rest_mean_delta": float(more_vals.mean()), "less_rest_mean_delta": float(less_vals.mean()),
            "diff": float(more_vals.mean() - less_vals.mean()), "t": float(t), "p": float(p),
            "cohens_d": float(d), "more_detail": {k: float(v) for k, v in more.items()},
            "less_detail": {k: float(v) for k, v in less.items()}}


def cross_model_agreement(df):
    """Correlate per-cell P(sleep) and per-persona effect vectors across models."""
    cell = df.groupby(["model_label", "scenario", "persona_id"])["advises_sleep"].mean().reset_index()
    pivot = cell.pivot_table(index=["scenario", "persona_id"], columns="model_label", values="advises_sleep")
    models = list(pivot.columns)
    pearson, spearman = {}, {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = pivot[models[i]], pivot[models[j]]
            m = a.notna() & b.notna()
            pr = stats.pearsonr(a[m], b[m])[0]
            sr = stats.spearmanr(a[m], b[m])[0]
            pearson[f"{models[i]}~{models[j]}"] = float(pr)
            spearman[f"{models[i]}~{models[j]}"] = float(sr)
    # persona-effect vector: per model, mean P(sleep) by persona minus that model's control mean
    eff = df.groupby(["model_label", "persona_id"])["advises_sleep"].mean().reset_index()
    epivot = eff.pivot_table(index="persona_id", columns="model_label", values="advises_sleep")
    ctrl_row = epivot.loc["control"]
    epivot_eff = epivot.sub(ctrl_row, axis=1).drop(index="control")
    eff_pearson = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = epivot_eff[models[i]], epivot_eff[models[j]]
            m = a.notna() & b.notna()
            eff_pearson[f"{models[i]}~{models[j]}"] = float(stats.pearsonr(a[m], b[m])[0])
    return {"cell_pearson": pearson, "cell_spearman": spearman,
            "mean_cell_pearson": float(np.mean(list(pearson.values()))),
            "mean_cell_spearman": float(np.mean(list(spearman.values()))),
            "persona_effect_pearson": eff_pearson,
            "mean_persona_effect_pearson": float(np.mean(list(eff_pearson.values()))),
            "persona_effect_vectors": epivot_eff.to_dict()}


def main():
    df = load()
    print(f"Loaded {len(df)} labeled responses across {df['model_label'].nunique()} models.")
    results = {}

    # 0. data health
    results["n_labeled"] = int(len(df))
    results["n_per_model"] = df["model_label"].value_counts().to_dict()
    results["refusal_rate"] = float(df["refusal"].mean()) if "refusal" in df else None
    results["overall_p_sleep"] = float(df["advises_sleep"].mean())

    # 1. labeler agreement
    k, n = cohen_kappa(df["advises_sleep_kw"], df["advises_sleep_judge"])
    agree = float((df["advises_sleep_kw"] == df["advises_sleep_judge"]).mean())
    results["labeler_agreement"] = {"cohen_kappa": float(k), "raw_agreement": agree, "n": n}

    # 2. variance decomposition
    results["variance_decomposition"] = omega_sq_anova(df)

    # 3. situation vs person spread (descriptive)
    scen_means = df.groupby("scenario")["advises_sleep"].mean()
    persona_means = df.groupby("persona_id")["advises_sleep"].mean()
    results["situation_effect"] = {"scenario_p_sleep": scen_means.round(4).to_dict(),
                                   "between_scenario_sd": float(scen_means.std()),
                                   "range": float(scen_means.max() - scen_means.min())}
    results["person_effect_descriptive"] = {"persona_p_sleep": persona_means.round(4).to_dict(),
                                            "between_persona_sd": float(persona_means.std()),
                                            "range": float(persona_means.max() - persona_means.min())}

    # 4. per-factor chi2 + 5. risk diffs + GEE
    results["per_factor_chi2"] = per_factor_chi2(df)
    results["risk_diff_vs_control"] = boot_risk_diff(df)
    results["gee_logistic"] = gee_logistic(df)

    # 6. stereotype contrast
    results["stereotype_contrast"] = stereotype_contrast(df)

    # 7. cross-model
    results["cross_model"] = cross_model_agreement(df)

    # 7b. SENSITIVITY: re-run the headline tests under each outcome definition.
    sens = {}
    for col in ["strong_sleep", "insist_sleep", "any_sleep", "regex"]:
        d2 = df.copy()
        d2["advises_sleep"] = d2[col].astype(int)
        vd = omega_sq_anova(d2)
        st = stereotype_contrast(d2)
        cm = cross_model_agreement(d2)
        sens[col] = {
            "overall_rate": float(d2["advises_sleep"].mean()),
            "omega2_scenario": vd.get("C(scenario)", {}).get("omega_sq"),
            "omega2_person": vd.get("C(level2)", {}).get("omega_sq"),
            "omega2_model": vd.get("C(model_label)", {}).get("omega_sq"),
            "omega2_interaction": vd.get("C(scenario):C(level2)", {}).get("omega_sq"),
            "stereotype_diff": st["diff"], "stereotype_p": st["p"], "stereotype_d": st["cohens_d"],
            "mean_cross_model_pearson": cm["mean_cell_pearson"],
        }
    results["sensitivity"] = sens

    # 7c. ordinal strength: situation vs person (Kruskal-Wallis epsilon^2)
    def eps2_kw(groupcol):
        groups = [g["strength"].values for _, g in df.groupby(groupcol)]
        H, p = stats.kruskal(*groups)
        k = len(groups); n = len(df)
        return {"H": float(H), "p": float(p), "epsilon_sq": float((H - k + 1) / (n - k))}
    results["strength_kruskal"] = {"by_scenario": eps2_kw("scenario"),
                                   "by_persona": eps2_kw("persona_id"),
                                   "by_model": eps2_kw("model_label"),
                                   "mean_strength": float(df["strength"].mean())}

    # per-model variance decomposition (is conditioning itself consistent?)
    permodel = {}
    for ml, sub in df.groupby("model_label"):
        try:
            vd = omega_sq_anova(sub.assign(model_label="x"))  # drop model term within model
        except Exception as e:
            vd = {"error": str(e)[:100]}
        sc = sub.groupby("scenario")["advises_sleep"].mean()
        pe = sub.groupby("persona_id")["advises_sleep"].mean()
        permodel[ml] = {"p_sleep": float(sub["advises_sleep"].mean()),
                        "between_scenario_sd": float(sc.std()),
                        "between_persona_sd": float(pe.std()),
                        "strength_mean": float(sub["strength"].mean())}
    results["per_model_summary"] = permodel

    # save cell means
    cm = df.groupby(["model_label", "scenario", "persona_id", "factor", "level"])["advises_sleep"].agg(["mean", "count"]).reset_index()
    cm.to_csv(os.path.join(CFG.RESULTS_DIR, "cell_means.csv"), index=False)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_JSON}")
    # quick console summary
    vd = results["variance_decomposition"]
    print("\n=== Variance decomposition (omega^2) ===")
    for k_, v in vd.items():
        print(f"  {k_:35s} omega2={v['omega_sq']:.3f}  eta2={v['eta_sq']:.3f}  p={v['p']:.2e}")
    print(f"\nSituation between-scenario SD: {results['situation_effect']['between_scenario_sd']:.3f}")
    print(f"Person between-persona  SD: {results['person_effect_descriptive']['between_persona_sd']:.3f}")
    print(f"Stereotype contrast diff: {results['stereotype_contrast']['diff']:.3f} (p={results['stereotype_contrast']['p']:.3g}, d={results['stereotype_contrast']['cohens_d']:.2f})")
    print(f"Mean cross-model cell Pearson r: {results['cross_model']['mean_cell_pearson']:.3f}")
    print(f"Labeler kappa: {results['labeler_agreement']['cohen_kappa']:.3f}")


if __name__ == "__main__":
    main()
