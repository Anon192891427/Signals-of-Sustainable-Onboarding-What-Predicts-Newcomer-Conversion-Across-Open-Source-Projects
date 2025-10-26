import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import zscore, norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_json", required=True)
    parser.add_argument("--scored_json", required=True)
    parser.add_argument("--tox_json", required=True)
    parser.add_argument("--exclude_list", default=None)
    parser.add_argument("--min_pr_denom", type=int, default=10,
                        help="Minimum denominator (per-outcome) to include a repo in that model.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min_effect_or", type=float, default=1.05)
    parser.add_argument("--cluster_col", type=str, default=None,
                        help="Column name for cluster-robust SE (e.g., 'org' or 'language'). If None, use HC1.")
    parser.add_argument("--use_firth", action="store_true",
                        help="Use Firth bias-reduced logistic regression (better for small samples).")
    parser.add_argument("--ridge_alpha", type=float, default=0.0,
                        help="Ridge penalty for regularization (0=none, higher=more shrinkage). Try 1.0 for retention.")
    parser.add_argument("--aggregate_retention", action="store_true",
                        help="Create aggregated 'any_retention' outcome (1m OR 3m OR 6m) for more power.")
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()

def load_json(path):
    with open(path) as f:
        return pd.json_normalize(json.load(f))

def load_exclude(path):
    if not path:
        return set()
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

def safe_merge(dfs, key="full_name"):
    """Outer join all DataFrames on key, preserving all repos"""
    all_keys = set().union(*[df[key].dropna() for df in dfs if key in df])
    base = pd.DataFrame({key: list(all_keys)})
    for df in dfs:
        if key in df:
            base = base.merge(df, on=key, how="left")
    return base

def prepare_data(df):
    """Type conversions, median imputation with missingness flags, log transforms"""
    outcome_pairs = [
        ("conversion_1m", "denom_conv_1m"),
        ("conversion_3m", "denom_conv_3m"),
        ("conversion_6m", "denom_conv_6m"),
        ("retention_1m", "denom_ret_1m"),
        ("retention_3m", "denom_ret_3m"),
        ("retention_6m", "denom_ret_6m"),
    ]
    for rate, denom in outcome_pairs:
        df[rate] = pd.to_numeric(df.get(rate), errors="coerce")
        df[denom] = pd.to_numeric(df.get(denom), errors="coerce").fillna(0).astype(int)

    flags = [
        "has_contributing", "has_coc", "has_issue_template", "has_pr_template",
        "has_codeowners", "has_maintainers", "has_governance", "has_workflows",
        "discussions_enabled"
    ]
    for flag in flags:
        df[flag] = df.get(flag, 0).fillna(0).astype(int)

    numerics = [
        "readme_words", "docs_md_count", "stars",
        "maintainer_first_reply_median_hrs_365", "maintainer_to_author_comment_ratio_365",
        "links_per_comment_365", "convo_broken_link_ratio_365", "R_convo_rule_365"
    ]
    for col in numerics:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
        is_miss = df[col].isna()
        if is_miss.any():
            df[col + "_missing"] = is_miss.astype(int)
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)
    
    for col in ["R", "A", "C", "score"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
        is_miss = df[col].isna()
        if is_miss.any():
            df[col + "_missing"] = is_miss.astype(int)
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)

    tox_ratio_temp = pd.to_numeric(df.get("tox_ratio"), errors="coerce").fillna(0)
    df["has_toxicity"] = (tox_ratio_temp > 0).astype(int)

    df["readme_words_log"] = np.log1p(df["readme_words"])
    df["log_reply_time"] = np.log1p(df["maintainer_first_reply_median_hrs_365"])
    df["log_mta_ratio"] = np.log1p(df["maintainer_to_author_comment_ratio_365"])
    df["star_quintile"] = pd.qcut(df["stars"], 5, labels=False, duplicates="drop")

    return df

def winsorize_and_zscore(df, cols, p):
    """Optional winsorization then z-score normalization for comparable ORs"""
    for col in cols:
        if col in df:
            if p and p > 0:
                low = df[col].quantile(p)
                high = df[col].quantile(1 - p)
                df[col] = np.clip(df[col], low, high)
            df[col] = zscore(df[col], nan_policy="omit")
    return df

def check_multicollinearity(df, predictors, outdir):
    """
    Calculate VIF for all predictors to detect multicollinearity.
    VIF > 10 indicates problematic multicollinearity.
    VIF 5-10 indicates moderate multicollinearity.
    """
    print("MULTICOLLINEARITY CHECK (Variance Inflation Factor)")
    
    X = df[predictors].copy()
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X = X.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    X = X.loc[:, X.nunique() > 1]
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
    
    vif_path = os.path.join(outdir, "vif_multicollinearity.csv")
    vif_data.to_csv(vif_path, index=False)
    print(f"\nVIF results saved to: {vif_path}\n")
    
    print(vif_data.to_string(index=False))
    print("\nInterpretation:")
    print("  VIF < 5:    Low multicollinearity")
    print("  VIF 5-10:   Moderate multicollinearity")
    print("  VIF > 10:   High multicollinearity")
    
    high_vif = vif_data[vif_data["VIF"] > 10]
    moderate_vif = vif_data[(vif_data["VIF"] >= 5) & (vif_data["VIF"] <= 10)]
    
    if not high_vif.empty:
        print(f"\nWARNING: {len(high_vif)} variable(s) with HIGH multicollinearity (VIF > 10):")
        for _, row in high_vif.iterrows():
            print(f"  - {row['Variable']}: VIF = {row['VIF']:.2f}")
        print("\nRecommendation: Consider removing one of the correlated variables.")
    
    if not moderate_vif.empty:
        print(f"\nCAUTION: {len(moderate_vif)} variable(s) with MODERATE multicollinearity (VIF 5-10):")
        for _, row in moderate_vif.iterrows():
            print(f"  - {row['Variable']}: VIF = {row['VIF']:.2f}")
    
    if high_vif.empty and moderate_vif.empty:
        print("\nAll predictors have VIF < 5 (low multicollinearity)")
    
    if not high_vif.empty:
        print("\n" + "-"*60)
        print("CORRELATION MATRIX (for variables with VIF > 10)")
        print("-"*60)
        high_vif_vars = high_vif["Variable"].tolist()
        corr_matrix = X[high_vif_vars].corr()
        corr_path = os.path.join(outdir, "correlation_matrix_high_vif.csv")
        corr_matrix.to_csv(corr_path)
        print(f"Saved to: {corr_path}\n")
        print(corr_matrix.to_string())
        
        print("\nHighly correlated pairs (|r| > 0.7):")
        for i in range(len(high_vif_vars)):
            for j in range(i+1, len(high_vif_vars)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    print(f"  - {high_vif_vars[i]} <-> {high_vif_vars[j]}: r = {corr_val:.3f}")
    
    return vif_data

def run_glm(df, outcome, denom, predictors, alpha, outdir, min_denom=1, cluster_col=None):
    """Binomial GLM with var_weights, rare-feature pruning, cluster-robust SE, ridge fallback"""
    subset = df[(pd.to_numeric(df[denom], errors="coerce") >= min_denom) & df[outcome].notna()].copy()
    if len(subset) < 10:
        return None, len(subset), 0, f"Too few rows for {outcome} (after {denom}>={min_denom})"

    X = subset[predictors].copy()
    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X = X.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    for col in X.columns:
        if X[col].nunique() == 2:
            prevalence = X[col].mean()
            if prevalence < 0.01 or prevalence > 0.99:
                print(f"  Dropping rare binary '{col}' (prevalence={prevalence:.3f}) from {outcome}")
                X = X.drop(columns=[col])
    
    X = X.loc[:, X.nunique() > 1]
    X = sm.add_constant(X, has_constant="add")

    y = subset[outcome].astype(float)
    weights = subset[denom].astype(int)
    
    assert ((y >= 0) & (y <= 1)).all(), f"{outcome} must be in [0,1]"
    
    perfect_zeros = ((y == 0) & (weights <= 5)).sum()
    perfect_ones = ((y == 1) & (weights <= 5)).sum()
    if perfect_zeros + perfect_ones > 0:
        print(f"  Warning: {outcome} has {perfect_zeros} y=0 and {perfect_ones} y=1 with denom≤5 (separation risk)")

    use_regularized = False
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=weights)
        
        if cluster_col and cluster_col in subset.columns:
            result = model.fit(cov_type="cluster", cov_kwds={"groups": subset[cluster_col]})
            print(f"  Using cluster-robust SE grouped by '{cluster_col}'")
        else:
            result = model.fit(cov_type="HC1")
    except Exception as e:
        try:
            print(f"  Standard GLM failed for {outcome}, trying regularized fit (SE/p-values unreliable)...")
            model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=weights)
            result = model.fit_regularized(alpha=1.0, L1_wt=0.0)
            use_regularized = True
        except Exception as e2:
            subset.to_csv(os.path.join(outdir, f"debug_{outcome}.csv"), index=False)
            return None, len(subset), 0, f"Both standard and regularized GLM failed: {e2}"

    if use_regularized:
        tbl = pd.DataFrame({
            "Coef.": result.params,
            "Std.Err.": getattr(result, "bse", np.full(len(result.params), np.nan)),
            "term": result.params.index
        })
        tbl["z"] = np.nan
        tbl["P>|z|"] = np.nan
        tbl["p"] = np.nan
        print(f"  Regularized estimates: SE/p/q values are NaN (unreliable inference)")
    else:
        tbl = result.summary2().tables[1].copy()
        tbl["term"] = tbl.index
        tbl["p"] = tbl["P>|z|"]
    
    tbl["OR"] = np.exp(tbl["Coef."])
    tbl["CI_low"] = np.exp(tbl["Coef."] - 1.96 * tbl["Std.Err."])
    tbl["CI_high"] = np.exp(tbl["Coef."] + 1.96 * tbl["Std.Err."])
    
    tbl["CI_width"] = tbl["CI_high"] - tbl["CI_low"]
    unstable = (tbl["CI_width"] > 50) | (tbl["OR"] > 50) | (tbl["OR"] < 0.01)
    if unstable.sum() > 0:
        excluded_terms = tbl.loc[unstable & (tbl["term"] != "const"), "term"].tolist()
        if excluded_terms:
            print(f"  Excluding {len(excluded_terms)} unstable estimates: {excluded_terms}")
        tbl = tbl[~unstable].copy()
    
    mask = tbl["term"] != "const"
    tbl["q"] = np.nan
    if mask.sum() > 0 and not use_regularized:
        _, qvals, _, _ = multipletests(tbl.loc[mask, "p"], method="fdr_bh")
        tbl.loc[mask, "q"] = qvals
    
    tbl.reset_index(drop=True)
    tbl["N_rows"] = len(subset)
    tbl["N_trials"] = int(weights.sum())
    
    print(f"  N_rows={len(subset)}, N_trials={int(weights.sum())}")
    if hasattr(result, 'deviance') and hasattr(result, 'null_deviance'):
        pseudo_r2 = 1 - (result.deviance / result.null_deviance)
        print(f"  Pseudo-R²={pseudo_r2:.4f}, Deviance={result.deviance:.2f}, Null deviance={result.null_deviance:.2f}")
    if hasattr(result, 'converged'):
        print(f"  Converged: {result.converged}")

    return tbl, len(subset), int(weights.sum()), None

def forest_plot(df, outcome, outdir):
    """Forest plot with clean labels, OR/CI annotations, adaptive margins"""
    if df is None or "CI_low" not in df or df.empty:
        return None

    d = df[df["term"] != "const"].copy()
    d = d.sort_values("OR").reset_index(drop=True)

    clean_names = {
        "R": "Readability", "A": "Actuality", "C": "Completeness",
        "has_contributing": "Contributing Guide", "has_coc": "Code of Conduct",
        "has_issue_template": "Issue Template", "has_pr_template": "PR Template",
        "has_codeowners": "CODEOWNERS", "has_maintainers": "Maintainers File",
        "has_governance": "Governance Docs", "has_workflows": "CI/CD Workflows",
        "discussions_enabled": "Discussions", "readme_words_log": "README Words (log)",
        "docs_md_count": "Doc Files", "links_per_comment_365": "Links/Comment",
        "convo_broken_link_ratio_365": "Broken Links", "R_convo_rule_365": "Conversation Score",
        "log_reply_time": "Reply Time (log SD)", "log_mta_ratio": "Maintainer/Author Ratio (log SD)",
        "has_toxicity": "Toxicity Present"
    }

    d["clean_term"] = d["term"].map(clean_names).fillna(d["term"])
    left_labels = [f'{row["clean_term"]} (q={row["q"]:.3f})' for _, row in d.iterrows()]

    n = len(d)
    fig_h = max(6.0, 0.55 * n)
    fig_w = 9.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=False)

    y = np.arange(n)
    x = d["OR"].values
    err_left = x - d["CI_low"].values
    err_right = d["CI_high"].values - x

    ax.errorbar(x, y, xerr=[err_left, err_right], fmt="o", capsize=6, elinewidth=2.0, markeredgewidth=1.0, markersize=6)
    ax.axvline(1, color="0.6", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(left_labels, fontsize=9)

    for yi, row in enumerate(d.itertuples()):
        ci_text = f"[{row.CI_low:.2f}, {row.CI_high:.2f}]"
        ax.text(row.CI_high + 0.02, yi, ci_text, va="center", ha="left", fontsize=8, color="#333333")

    x_max = np.nanmax(d["CI_high"].values) * 1.5
    ax.set_xlim(0, x_max)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", labelsize=9)
    ax.margins(y=0.05)
    ax.set_xlabel("Odds Ratio", fontsize=10, labelpad=6)
    ax.set_title(f"Effect Sizes: {outcome}", pad=10, fontsize=12, fontweight="bold")

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=8, 
                   markeredgewidth=1.0, markeredgecolor='#1f77b4', label='Odds Ratio'),
        plt.Line2D([0], [0], color='#1f77b4', linewidth=2, label='95% Confidence Interval')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, 
              shadow=True, fontsize=8, framealpha=0.95)

    max_left_len = max(len(s) for s in left_labels) if left_labels else 12
    left_margin = min(0.75, 0.20 + 0.007 * max_left_len)
    fig.subplots_adjust(left=left_margin, right=0.88, top=0.93, bottom=0.06)

    path = os.path.join(outdir, f"forest_{outcome}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path

def forest_plot_combined(results, outcome_group, outdir):
    """Combined forest plot for multiple time windows (conversion or retention 1m/3m/6m)"""
    clean_names = {
        "R": "Readability", "A": "Actuality", "C": "Completeness",
        "has_contributing": "Contributing Guide", "has_coc": "Code of Conduct",
        "has_issue_template": "Issue Template", "has_pr_template": "PR Template",
        "has_codeowners": "CODEOWNERS", "has_maintainers": "Maintainers File",
        "has_governance": "Governance Docs", "has_workflows": "CI/CD Workflows",
        "discussions_enabled": "Discussions", "readme_words_log": "README Words (log)",
        "docs_md_count": "Doc Files", "links_per_comment_365": "Links/Comment",
        "convo_broken_link_ratio_365": "Broken Links", "R_convo_rule_365": "Conversation Score",
        "log_reply_time": "Reply Time (log SD)", "log_mta_ratio": "Maintainer/Author Ratio (log SD)",
        "has_toxicity": "Toxicity Present"
    }
    
    if outcome_group == "conversion":
        outcomes = ["conversion_1m", "conversion_3m", "conversion_6m"]
        titles = ["1 Month", "3 Months", "6 Months"]
        main_title = "Conversion Outcomes: Effect Sizes Across Time Windows"
        filename = "forest_conversion_combined.png"
    else:
        outcomes = ["retention_1m", "retention_3m", "retention_6m"]
        titles = ["1 Month", "3 Months", "6 Months"]
        main_title = "Retention Outcomes: Effect Sizes Across Time Windows"
        filename = "forest_retention_combined.png"
    
    dfs = []
    for outcome in outcomes:
        df = results.get(outcome)
        if df is not None and not df.empty:
            d = df[df["term"] != "const"].copy()
            d["clean_term"] = d["term"].map(clean_names).fillna(d["term"])
            dfs.append(d)
        else:
            dfs.append(None)
    
    if all(df is None for df in dfs):
        return None
    
    all_terms = set()
    for df in dfs:
        if df is not None:
            all_terms.update(df["clean_term"].tolist())
    all_terms = sorted(all_terms)
    n_terms = len(all_terms)
    
    if n_terms == 0:
        return None
    
    fig_h = max(8.0, 0.45 * n_terms)
    fig_w = 16
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), sharey=True)
    
    for idx, (ax, df, title) in enumerate(zip(axes, dfs, titles)):
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue
        
        term_to_data = {row["clean_term"]: row for _, row in df.iterrows()}
        
        y_positions = []
        x_values = []
        err_left_values = []
        err_right_values = []
        colors = []
        ci_labels = []
        
        for i, term in enumerate(all_terms):
            if term in term_to_data:
                row = term_to_data[term]
                y_positions.append(i)
                x_values.append(row["OR"])
                err_left_values.append(row["OR"] - row["CI_low"])
                err_right_values.append(row["CI_high"] - row["OR"])
                ci_labels.append(f"[{row['CI_low']:.2f}, {row['CI_high']:.2f}]")
                
                if row["q"] < 0.05:
                    colors.append("#2166ac" if row["OR"] > 1 else "#b2182b")
                else:
                    colors.append("#999999")
        
        if y_positions:
            for y, x, el, er, color in zip(y_positions, x_values, err_left_values, err_right_values, colors):
                ax.errorbar(x, y, xerr=[[el], [er]], fmt="o", capsize=4, 
                           elinewidth=1.5, markeredgewidth=1.0, markersize=5, 
                           color=color, ecolor=color, alpha=0.8)
            
            for y, x_high, ci_text in zip(y_positions, [x_values[i] + err_right_values[i] for i in range(len(y_positions))], ci_labels):
                ax.text(x_high * 1.02, y, ci_text, va="center", ha="left", fontsize=7, color="#333333")
        
        ax.axvline(1, color="0.4", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_yticks(range(n_terms))
        
        if idx == 0:
            ax.set_yticklabels(all_terms, fontsize=9)
        
        x_min = min([d["CI_low"].min() for d in dfs if d is not None and not d.empty])
        x_max = max([d["CI_high"].max() for d in dfs if d is not None and not d.empty])
        x_range = x_max - x_min
        ax.set_xlim(max(0, x_min - 0.1*x_range), x_max + 0.35*x_range)
        
        ax.grid(axis="x", linestyle=":", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel("Odds Ratio", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    
    fig.suptitle(main_title, fontsize=13, fontweight="bold", y=0.995)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166ac', markersize=8, label='Positive (q<0.05)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#b2182b', markersize=8, label='Negative (q<0.05)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', markersize=8, label='Not significant'),
        plt.Line2D([0], [0], color='#333333', linewidth=0, marker='$CI$', markersize=8, label='[CI_low, CI_high]')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=True, fancybox=True, 
              shadow=True, fontsize=8, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    path = os.path.join(outdir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[combined-plot] saved {filename}")
    return path

def generate_insights(results, _min_or):
    """Ranked predictors by avg OR + detailed per-outcome results"""
    clean_names = {
        "R": "Readability", "A": "Actuality", "C": "Completeness",
        "has_contributing": "Contributing Guide", "has_coc": "Code of Conduct",
        "has_issue_template": "Issue Template", "has_pr_template": "PR Template",
        "has_codeowners": "CODEOWNERS", "has_maintainers": "Maintainers File",
        "has_governance": "Governance Docs", "has_workflows": "CI/CD Workflows",
        "discussions_enabled": "Discussions", "readme_words_log": "README Words (log)",
        "docs_md_count": "Doc Files", "links_per_comment_365": "Links/Comment",
        "convo_broken_link_ratio_365": "Broken Links", "R_convo_rule_365": "Conversation Score",
        "log_reply_time": "Reply Time (log SD)", "log_mta_ratio": "Maintainer/Author Ratio (log SD)",
        "has_toxicity": "Toxicity Present"
    }
    
    titles = {
        "conversion_1m": "Conversion · 1 Month", "conversion_3m": "Conversion · 3 Months",
        "conversion_6m": "Conversion · 6 Months", "retention_1m":  "Retention · 1 Month",
        "retention_3m":  "Retention · 3 Months", "retention_6m":  "Retention · 6 Months",
    }
    order = ["conversion_1m","conversion_3m","conversion_6m","retention_1m","retention_3m","retention_6m"]
    conversion_outcomes = ["conversion_1m", "conversion_3m", "conversion_6m"]
    retention_outcomes = ["retention_1m", "retention_3m", "retention_6m"]
    
    def compute_avg_or(outcome_list):
        predictor_ors = {}
        for outcome in outcome_list:
            df = results.get(outcome)
            if df is not None and not df.empty:
                for _, row in df[df["term"] != "const"].iterrows():
                    term = row["term"]
                    if term not in predictor_ors:
                        predictor_ors[term] = []
                    predictor_ors[term].append(row["OR"])
        avg_ors = []
        for term, ors in predictor_ors.items():
            avg_ors.append({"term": term, "clean_term": clean_names.get(term, term), "avg_OR": np.mean(ors), "n_models": len(ors)})
        return pd.DataFrame(avg_ors).sort_values("avg_OR", ascending=False)
    
    all_avg_or = compute_avg_or(order)
    conversion_avg_or = compute_avg_or(conversion_outcomes)
    retention_avg_or = compute_avg_or(retention_outcomes)

    header = (
        "# Key Insights (Full, mirrors forest plots)\n\n"
        "For each outcome, every modeled term from the forest plot is listed below in the same "
        "order (sorted by Odds Ratio) and with the same numeric labels.\n\n"
        "- **OR (Odds Ratio)**: >1 = positive, <1 = negative.\n"
        "- **Continuous predictors**: ORs are per 1 SD change (z-scored).\n"
        "- **Binary predictors**: ORs are for feature present vs absent.\n"
        "- **q**: FDR-adjusted p-value (shown for reference).\n\n"
    )
    
    header += "## Predictor Rankings by Average Odds Ratio\n\n"
    if not all_avg_or.empty:
        header += "### Overall (All 6 Outcomes)\n\n"
        for _, row in all_avg_or.iterrows():
            header += f"- **{row['clean_term']}**: Avg OR = {row['avg_OR']:.3f} (n={row['n_models']})\n"
        header += "\n"
    if not conversion_avg_or.empty:
        header += "### Conversion Outcomes (1m, 3m, 6m)\n\n"
        for _, row in conversion_avg_or.iterrows():
            header += f"- **{row['clean_term']}**: Avg OR = {row['avg_OR']:.3f} (n={row['n_models']})\n"
        header += "\n"
    if not retention_avg_or.empty:
        header += "### Retention Outcomes (1m, 3m, 6m)\n\n"
        for _, row in retention_avg_or.iterrows():
            header += f"- **{row['clean_term']}**: Avg OR = {row['avg_OR']:.3f} (n={row['n_models']})\n"
        header += "\n"
    header += "---\n\n"
    
    lines = [header]
    for outcome in order:
        df = results.get(outcome)
        lines.append(f"## {titles.get(outcome, outcome)}")
        if df is None or df.empty:
            lines.append("- No model results available.\n")
            continue
        d = df[df["term"] != "const"].copy().sort_values("OR").reset_index(drop=True)
        for _, r in d.iterrows():
            clean_term = clean_names.get(r['term'], r['term'])
            lines.append(f"- **{clean_term}** — 95% CI [{r['CI_low']:.2f}, {r['CI_high']:.2f}] · OR {r['OR']:.2f} · q={r['q']:.3f}")
        lines.append("")
    return "\n".join(lines)

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    import sys
    import scipy
    session_info = f"""Session Info
=============
Date: {pd.Timestamp.now()}
Python: {sys.version}
NumPy: {np.__version__}
Pandas: {pd.__version__}
Statsmodels: {sm.__version__}
SciPy: {scipy.__version__}
Matplotlib: {plt.matplotlib.__version__}

Command-line arguments:
{vars(args)}
"""
    with open(os.path.join(args.outdir, "SESSION_INFO.txt"), "w") as f:
        f.write(session_info)
    
    print("Loading...")
    df = safe_merge([
        load_json(args.features_json),
        load_json(args.scored_json),
        load_json(args.tox_json).rename(columns={"repo": "full_name"})
    ])
    print(f"Merged {len(df)} repos.")

    print("Preprocessing...")
    df = prepare_data(df)

    if args.exclude_list:
        exclude = load_exclude(args.exclude_list)
        df = df[~df["full_name"].isin(exclude)]

    df = winsorize_and_zscore(df, [
        "readme_words", "readme_words_log", "docs_md_count", "R", "A", "C", "stars",
        "maintainer_first_reply_median_hrs_365", "maintainer_to_author_comment_ratio_365",
        "links_per_comment_365", "convo_broken_link_ratio_365", "R_convo_rule_365",
        "log_reply_time", "log_mta_ratio"
    ], 0.01)

    df.to_csv(os.path.join(args.outdir, "merged_analysis_input.csv"), index=False)

    predictors = [
        "has_contributing", "has_coc", "has_issue_template", "has_pr_template",
        "has_codeowners", "has_maintainers", "has_governance", "has_workflows", "discussions_enabled",
        "readme_words_log", "docs_md_count", "R", "A", "C",
        "links_per_comment_365", "convo_broken_link_ratio_365", "R_convo_rule_365",
        "log_reply_time", "log_mta_ratio", "has_toxicity"
    ]
    
    assert not ({"score"} & set(predictors) and {"R","A","C"} & set(predictors)), \
        "Cannot include both 'score' and its components (R, A, C) in the same model"

    outcomes = [
        ("conversion_1m", "denom_conv_1m"), ("conversion_3m", "denom_conv_3m"),
        ("conversion_6m", "denom_conv_6m"), ("retention_1m", "denom_ret_1m"),
        ("retention_3m", "denom_ret_3m"), ("retention_6m", "denom_ret_6m"),
    ]

    check_multicollinearity(df, predictors, args.outdir)

    results = {}
    for outcome, denom in outcomes:
        print(f"Modeling {outcome}...")
        model_df, n, trials, err = run_glm(
            df, outcome, denom, predictors, args.alpha, args.outdir, min_denom=args.min_pr_denom, cluster_col=args.cluster_col
        )
        if err:
            print(f"Note: {err}")
        results[outcome] = model_df
        if model_df is not None:
            model_df.to_csv(os.path.join(args.outdir, f"model_{outcome}.csv"), index=False)
            forest_plot(model_df, outcome, args.outdir)

    forest_plot_combined(results, "conversion", args.outdir)
    forest_plot_combined(results, "retention", args.outdir)

    with open(os.path.join(args.outdir, "insights.md"), "w") as f:
        f.write(generate_insights(results, args.min_effect_or))

    print("Done!")

if __name__ == "__main__":
    main()