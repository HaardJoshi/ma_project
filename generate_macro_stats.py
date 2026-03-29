"""
generate_macro_stats.py
========================
Extracts global metrics from all results files, SHAP values, and hypothesis test
data into a single macro_stats.json file for the M&A Deal Intelligence Terminal.

Usage:
    python generate_macro_stats.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine

# Fix paths
sys.path.insert(0, str(Path(os.getcwd()) / "scripts"))
from training_utils import (
    load_and_prepare_data, get_feature_configs,
    SEED, N_FOLDS, FINANCIAL_COLS, TEXT_COLS, GRAPH_COLS
)

np.random.seed(SEED)

RESULTS_DIR = "results"


def extract_model_performance():
    """Parse all result CSVs into structured model performance data."""
    print("\n[1/5] Extracting model performance metrics...")

    perf = {}

    # ── Classifier results (AUC-based) ──────────────────────────
    clf_file = os.path.join(RESULTS_DIR, "classifier_results_20260311_150342.csv")
    clf = pd.read_csv(clf_file)

    for _, row in clf.iterrows():
        key = f"{row['config']}_{row['model']}"
        perf[key] = {
            "config": row["config"],
            "config_name": row["config_name"],
            "model": row["model"],
            "n_features": int(row["n_features"]),
            "auc_mean": round(float(row["AUC_ROC_mean"]), 4),
            "auc_std": round(float(row["AUC_ROC_std"]), 4),
            "accuracy_mean": round(float(row["Accuracy_mean"]), 4),
            "f1_mean": round(float(row["F1_mean"]), 4),
            "precision_mean": round(float(row["Precision_mean"]), 4),
            "recall_mean": round(float(row["Recall_mean"]), 4),
        }

    # ── Regression results (R², RMSE) ───────────────────────────
    reg_files = {
        "linear_baselines": os.path.join(RESULTS_DIR, "linear_baselines_results_20260311_140128.csv"),
        "xgboost_reg": os.path.join(RESULTS_DIR, "xgboost_results_20260311_140905.csv"),
        "mlp_reg": os.path.join(RESULTS_DIR, "mlp_results_20260311_141249.csv"),
    }

    regression = {}
    for label, fpath in reg_files.items():
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            key = f"{row['config']}_{row['model']}_reg"
            regression[key] = {
                "config": row["config"],
                "model": row["model"],
                "n_features": int(row["n_features"]),
                "r2_mean": round(float(row["R2_mean"]), 4),
                "r2_std": round(float(row["R2_std"]), 4),
                "rmse_mean": round(float(row["RMSE_mean"]), 4),
                "rmse_std": round(float(row["RMSE_std"]), 4),
                "mae_mean": round(float(row["MAE_mean"]), 4),
            }

    # ── Tuned XGBoost ───────────────────────────────────────────
    tuned_file = os.path.join(RESULTS_DIR, "tuned_xgboost_results_20260311_151704.csv")
    tuned = pd.read_csv(tuned_file)

    tuned_results = {}
    for _, row in tuned.iterrows():
        tuned_results[row["config"]] = {
            "config_name": row["config_name"],
            "n_features": int(row["n_features"]),
            "auc_baseline": round(float(row["AUC_baseline"]), 4),
            "auc_tuned_mean": round(float(row["AUC_tuned_mean"]), 4),
            "auc_tuned_std": round(float(row["AUC_tuned_std"]), 4),
            "auc_delta": round(float(row["AUC_delta"]), 4),
            "accuracy_mean": round(float(row["Acc_mean"]), 4),
            "f1_mean": round(float(row["F1_mean"]), 4),
        }

    # ── Enhanced classifier ─────────────────────────────────────
    enh_file = os.path.join(RESULTS_DIR, "enhanced_classifier_results_20260311_152647.csv")
    enh = pd.read_csv(enh_file)

    enhanced = {}
    for _, row in enh.iterrows():
        enhanced[row["config"]] = {
            "config_desc": row["config_desc"],
            "n_features": int(row["n_features"]),
            "auc_mean": round(float(row["AUC_mean"]), 4),
            "auc_std": round(float(row["AUC_std"]), 4),
            "accuracy_mean": round(float(row["Acc_mean"]), 4),
            "f1_mean": round(float(row["F1_mean"]), 4),
        }

    # ── Build the ablation wall matrix ──────────────────────────
    ablation_wall = []
    for config in ["M1", "M2", "M3"]:
        for model in ["logreg", "xgboost", "mlp"]:
            key = f"{config}_{model}"
            if key in perf:
                p = perf[key]
                ablation_wall.append({
                    "config": config,
                    "config_name": p["config_name"],
                    "model": model,
                    "auc": p["auc_mean"],
                    "auc_std": p["auc_std"],
                    "accuracy": p["accuracy_mean"],
                    "f1": p["f1_mean"],
                    "n_features": p["n_features"],
                })

    # ── Best performers ─────────────────────────────────────────
    best_auc = max(ablation_wall, key=lambda x: x["auc"])

    print(f"    Loaded {len(perf)} classifier configs, {len(regression)} regression configs")
    print(f"    Best AUC: {best_auc['config']}_{best_auc['model']} = {best_auc['auc']:.4f}")

    return {
        "classifier_performance": perf,
        "regression_performance": regression,
        "tuned_xgboost": tuned_results,
        "enhanced_classifier": enhanced,
        "ablation_wall": ablation_wall,
        "best_classifier": best_auc,
    }


def extract_shap_distributions():
    """Compute global SHAP distributions from shap_values_M3.csv."""
    print("\n[2/5] Computing global SHAP distributions...")

    shap_df = pd.read_csv(os.path.join(RESULTS_DIR, "shap_values_M3.csv"))
    n_samples = len(shap_df)

    # Mean absolute SHAP per feature
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

    # Top 20 features by mean |SHAP|
    top20 = []
    for feat, val in mean_abs_shap.head(20).items():
        col_vals = shap_df[feat].values
        top20.append({
            "feature": feat,
            "mean_abs_shap": round(float(val), 6),
            "mean_shap": round(float(col_vals.mean()), 6),
            "std_shap": round(float(col_vals.std()), 6),
            "min_shap": round(float(col_vals.min()), 6),
            "max_shap": round(float(col_vals.max()), 6),
            "pct_positive": round(float((col_vals > 0).mean() * 100), 1),
        })

    # Feature category breakdown
    fin_cols = [c for c in shap_df.columns if not c.startswith(("mda_pca_", "rf_pca_", "graph_emb_"))]
    mda_cols = [c for c in shap_df.columns if c.startswith("mda_pca_")]
    rf_cols = [c for c in shap_df.columns if c.startswith("rf_pca_")]
    graph_cols = [c for c in shap_df.columns if c.startswith("graph_emb_")]

    category_impact = {
        "financial": round(float(shap_df[fin_cols].abs().mean().mean()), 6),
        "mda_text": round(float(shap_df[mda_cols].abs().mean().mean()), 6) if mda_cols else 0,
        "rf_text": round(float(shap_df[rf_cols].abs().mean().mean()), 6) if rf_cols else 0,
        "graph": round(float(shap_df[graph_cols].abs().mean().mean()), 6) if graph_cols else 0,
    }

    # Beeswarm data: for each of top 20 features, sample values for plotting
    beeswarm_data = []
    for feat_info in top20:
        feat = feat_info["feature"]
        vals = shap_df[feat].values
        # Subsample for plotting (max 200 points)
        if len(vals) > 200:
            idx = np.random.choice(len(vals), 200, replace=False)
            sampled = vals[idx].tolist()
        else:
            sampled = vals.tolist()
        beeswarm_data.append({
            "feature": feat,
            "values": [round(v, 6) for v in sampled],
        })

    print(f"    {n_samples} samples, {len(shap_df.columns)} features")
    print(f"    Top feature: {top20[0]['feature']} (|SHAP|={top20[0]['mean_abs_shap']:.6f})")
    print(f"    Category impact: Financial={category_impact['financial']:.6f}, "
          f"MD&A={category_impact['mda_text']:.6f}, RF={category_impact['rf_text']:.6f}, "
          f"Graph={category_impact['graph']:.6f}")

    return {
        "n_samples": n_samples,
        "n_features": len(shap_df.columns),
        "top20_features": top20,
        "category_impact": category_impact,
        "beeswarm_data": beeswarm_data,
    }


def run_hypothesis_tests():
    """Run all three hypothesis tests and capture structured results."""
    print("\n[3/5] Running hypothesis tests...")

    subset, y_cont = load_and_prepare_data()
    configs = get_feature_configs(subset)
    y_binary = (y_cont > 0).astype(int)

    h_results = {}

    # ═══ H1: Topological Alpha - Sector-Segmented Analysis ═══
    print("    Running H1: Topological Alpha...")

    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    sic_col = "Current Acquirer SIC Code"
    h1_data = {"supported": False, "sector_results": {}}

    if sic_col in subset.columns:
        sic_2digit = subset[sic_col].astype(str).str[:2]
        sc_dependent = [str(i) for i in range(20, 50)]
        asset_light = [str(i) for i in range(60, 68)] + ["70", "73", "78", "79"]

        sector_groups = {"supply_chain": sc_dependent, "asset_light": asset_light}

        for group_name, sic_list in sector_groups.items():
            mask = sic_2digit.isin(sic_list)
            n_deals = int(mask.sum())
            y_group = y_binary[mask.values]
            n_pos = int((y_group == 1).sum())

            if n_deals < 50:
                h1_data["sector_results"][group_name] = {
                    "n_deals": n_deals, "skipped": True
                }
                continue

            n_folds = min(N_FOLDS, max(2, n_deals // 30))
            sector_aucs = {}

            for config_name in ["M1", "M3"]:
                X = subset.loc[mask, configs[config_name]["cols"]].values
                neg, pos = (y_group == 0).sum(), (y_group == 1).sum()
                spw = neg / max(pos, 1)
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
                fold_aucs = []

                for train_idx, test_idx in skf.split(X, y_group):
                    model = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("model", XGBClassifier(
                            n_estimators=300, max_depth=5, learning_rate=0.05,
                            min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=1.0,
                            objective="binary:logistic", scale_pos_weight=spw,
                            random_state=SEED, n_jobs=-1, verbosity=0,
                        )),
                    ])
                    model.fit(X[train_idx], y_group[train_idx])
                    y_prob = model.predict_proba(X[test_idx])[:, 1]
                    fold_aucs.append(float(roc_auc_score(y_group[test_idx], y_prob)))

                avg_auc = float(np.mean(fold_aucs))
                sector_aucs[config_name] = {"auc": round(avg_auc, 4), "folds": fold_aucs}

            delta = sector_aucs["M3"]["auc"] - sector_aucs["M1"]["auc"]
            t_stat, p_val = stats.ttest_rel(sector_aucs["M3"]["folds"], sector_aucs["M1"]["folds"])

            h1_data["sector_results"][group_name] = {
                "n_deals": n_deals,
                "n_positive": n_pos,
                "m1_auc": sector_aucs["M1"]["auc"],
                "m3_auc": sector_aucs["M3"]["auc"],
                "delta": round(float(delta), 4),
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_val), 4),
                "significant": bool(p_val < 0.05),
                "skipped": False,
            }

        sc_d = h1_data["sector_results"].get("supply_chain", {}).get("delta", 0)
        al_d = h1_data["sector_results"].get("asset_light", {}).get("delta", 0)
        h1_data["supported"] = sc_d > al_d
        h1_data["sc_delta"] = round(sc_d, 4)
        h1_data["al_delta"] = round(al_d, 4)
        print(f"      SC Δ={sc_d:+.4f}, AL Δ={al_d:+.4f} → {'SUPPORTED' if h1_data['supported'] else 'NOT SUPPORTED'}")

    h_results["H1"] = h1_data

    # ═══ H2: Semantic Divergence ═══
    print("    Running H2: Semantic Divergence...")

    mda_cols_avail = sorted([c for c in subset.columns if c.startswith("mda_pca_")])
    rf_cols_avail = sorted([c for c in subset.columns if c.startswith("rf_pca_")])

    h2_data = {"supported": False}

    if mda_cols_avail and rf_cols_avail:
        has_mda = subset[mda_cols_avail].abs().sum(axis=1) > 0
        has_rf = subset[rf_cols_avail].abs().sum(axis=1) > 0
        has_both = has_mda & has_rf
        text_subset = subset[has_both].copy()
        y_text = y_cont[has_both.values]

        mda_emb = text_subset[mda_cols_avail].values
        rf_emb = text_subset[rf_cols_avail].values
        mda_mean = mda_emb.mean(axis=0)
        rf_mean = rf_emb.mean(axis=0)

        mda_sim = np.array([1 - cosine(mda_emb[i], mda_mean) for i in range(len(mda_emb))])
        rf_sim = np.array([1 - cosine(rf_emb[i], rf_mean) for i in range(len(rf_emb))])

        valid = np.isfinite(mda_sim) & np.isfinite(rf_sim)
        mda_sim, rf_sim, y_text = mda_sim[valid], rf_sim[valid], y_text[valid]

        from sklearn.linear_model import LinearRegression
        X_sim = np.column_stack([mda_sim, rf_sim])
        reg = LinearRegression().fit(X_sim, y_text)
        beta_mda, beta_rf = float(reg.coef_[0]), float(reg.coef_[1])
        r2 = float(reg.score(X_sim, y_text))

        r_mda, p_mda = stats.pearsonr(mda_sim, y_text)
        r_rf, p_rf = stats.pearsonr(rf_sim, y_text)

        # Quartile analysis
        quartile_data = {}
        for label, sim_vals in [("mda", mda_sim), ("rf", rf_sim)]:
            q_cuts = np.percentile(sim_vals, [25, 50, 75])
            q1_vals = y_text[sim_vals <= q_cuts[0]]
            q4_vals = y_text[sim_vals > q_cuts[2]]
            delta_q = float(q4_vals.mean() - q1_vals.mean())
            t_q, p_q = stats.ttest_ind(q4_vals, q1_vals)

            quartile_data[label] = {
                "q1_mean": round(float(q1_vals.mean()), 4),
                "q4_mean": round(float(q4_vals.mean()), 4),
                "q4_q1_delta": round(delta_q, 4),
                "t_stat": round(float(t_q), 4),
                "p_value": round(float(p_q), 4),
            }

        h2_supported = beta_mda > 0 and beta_rf < 0
        h2_data = {
            "supported": h2_supported,
            "n_deals": int(has_both.sum()),
            "n_mda": int(has_mda.sum()),
            "n_rf": int(has_rf.sum()),
            "beta_mda": round(beta_mda, 4),
            "beta_rf": round(beta_rf, 4),
            "intercept": round(float(reg.intercept_), 4),
            "r_squared": round(r2, 6),
            "corr_mda_car": {"r": round(float(r_mda), 4), "p": round(float(p_mda), 4)},
            "corr_rf_car": {"r": round(float(r_rf), 4), "p": round(float(p_rf), 4)},
            "mda_sim_stats": {
                "mean": round(float(mda_sim.mean()), 4),
                "std": round(float(mda_sim.std()), 4),
            },
            "rf_sim_stats": {
                "mean": round(float(rf_sim.mean()), 4),
                "std": round(float(rf_sim.std()), 4),
            },
            "mda_sim_values": [round(float(v), 4) for v in mda_sim.tolist()],
            "rf_sim_values": [round(float(v), 4) for v in rf_sim.tolist()],
            "car_values": [round(float(v), 4) for v in y_text.tolist()],
            "quartiles": quartile_data,
        }
        print(f"      β_MDA={beta_mda:+.4f}, β_RF={beta_rf:+.4f}, R²={r2:.6f} → "
              f"{'SUPPORTED' if h2_supported else 'NOT SUPPORTED'}")

    h_results["H2"] = h2_data

    # ═══ H3: Topological Arbitrage - Centrality vs CAR Variance ═══
    print("    Running H3: Topological Arbitrage...")

    import torch
    import networkx as nx

    graph_file = "data/interim/hetero_supply_chain_graph.pt"
    meta_file = "data/interim/hetero_graph_metadata.json"

    h3_data = {"supported": False}

    if os.path.exists(graph_file) and os.path.exists(meta_file):
        data_graph = torch.load(graph_file, weights_only=False)
        with open(meta_file, "r") as f:
            meta = json.load(f)

        ticker_to_id = meta["ticker_to_id"]
        deal_to_acq_ticker = meta["deal_to_acq_ticker"]

        G = nx.DiGraph()
        G.add_nodes_from(range(data_graph["company"].num_nodes))
        for edge_type in [("company", "supplies", "company"), ("company", "buys_from", "company")]:
            ei = data_graph[edge_type].edge_index
            for i in range(ei.size(1)):
                G.add_edge(ei[0, i].item(), ei[1, i].item())

        betweenness = nx.betweenness_centrality(G)
        clustering = nx.clustering(G.to_undirected())
        degree = nx.degree_centrality(G)

        records = []
        for deal_id_str, acq_ticker in deal_to_acq_ticker.items():
            deal_id = int(deal_id_str)
            node_id = ticker_to_id.get(acq_ticker)
            if node_id is not None:
                records.append({
                    "deal_id": deal_id,
                    "betweenness": betweenness.get(node_id, 0),
                    "clustering": clustering.get(node_id, 0),
                    "degree": degree.get(node_id, 0),
                })

        centrality_df = pd.DataFrame(records)
        subset_copy = subset.copy()
        subset_copy["deal_id"] = subset_copy.index
        subset_copy["car"] = y_cont
        merged = subset_copy.merge(centrality_df, on="deal_id", how="inner")
        merged["abs_car"] = merged["car"].abs()

        # Correlations
        r_bet_car, p_bet_car = stats.pearsonr(merged["betweenness"], merged["car"])
        r_bet_abs, p_bet_abs = stats.pearsonr(merged["betweenness"], merged["abs_car"])
        r_clu_car, p_clu_car = stats.pearsonr(merged["clustering"], merged["car"])
        r_clu_abs, p_clu_abs = stats.pearsonr(merged["clustering"], merged["abs_car"])

        # Quartile analysis for betweenness
        ranks = merged["betweenness"].rank(method="first")
        q_labels = pd.qcut(ranks, q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        bet_quartiles = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            m = q_labels == q
            car_q = merged.loc[m, "car"]
            abs_car_q = merged.loc[m, "abs_car"]
            bet_quartiles[q] = {
                "n": int(m.sum()),
                "car_mean": round(float(car_q.mean()), 5),
                "car_std": round(float(car_q.std()), 5),
                "car_var": round(float(car_q.var()), 6),
                "abs_car_mean": round(float(abs_car_q.mean()), 5),
            }

        # Levene test (Q4 vs Q1)
        lev_stat, lev_p = stats.levene(
            merged.loc[q_labels == "Q1", "car"],
            merged.loc[q_labels == "Q4", "car"],
        )

        h3_bet = float(r_bet_abs) > 0
        h3_clu = float(r_clu_abs) < 0
        h3_supported = h3_bet and h3_clu

        h3_data = {
            "supported": h3_supported,
            "partially_supported": h3_bet or h3_clu,
            "n_merged": len(merged),
            "n_graph_nodes": G.number_of_nodes(),
            "n_graph_edges": G.number_of_edges(),
            "correlations": {
                "betweenness_car": {"r": round(float(r_bet_car), 4), "p": round(float(p_bet_car), 4)},
                "betweenness_abs_car": {"r": round(float(r_bet_abs), 4), "p": round(float(p_bet_abs), 4)},
                "clustering_car": {"r": round(float(r_clu_car), 4), "p": round(float(p_clu_car), 4)},
                "clustering_abs_car": {"r": round(float(r_clu_abs), 4), "p": round(float(p_clu_abs), 4)},
            },
            "betweenness_quartiles": bet_quartiles,
            "levene_test": {
                "statistic": round(float(lev_stat), 4),
                "p_value": round(float(lev_p), 4),
                "significant": bool(lev_p < 0.05),
            },
            "centrality_stats": {
                "betweenness": {
                    "mean": round(float(merged["betweenness"].mean()), 6),
                    "std": round(float(merged["betweenness"].std()), 6),
                    "nonzero_pct": round(float((merged["betweenness"] > 0).mean() * 100), 1),
                },
                "clustering": {
                    "mean": round(float(merged["clustering"].mean()), 6),
                    "std": round(float(merged["clustering"].std()), 6),
                    "nonzero_pct": round(float((merged["clustering"] > 0).mean() * 100), 1),
                },
                "degree": {
                    "mean": round(float(merged["degree"].mean()), 6),
                    "std": round(float(merged["degree"].std()), 6),
                    "nonzero_pct": round(float((merged["degree"] > 0).mean() * 100), 1),
                },
            },
            # Store scatter data for H3 visualization
            "scatter_betweenness": [round(float(v), 6) for v in merged["betweenness"].tolist()],
            "scatter_car": [round(float(v), 4) for v in merged["car"].tolist()],
            "scatter_abs_car": [round(float(v), 4) for v in merged["abs_car"].tolist()],
            "scatter_clustering": [round(float(v), 6) for v in merged["clustering"].tolist()],
        }
        print(f"      Betweenness→|CAR|: r={r_bet_abs:+.4f}, Clustering→|CAR|: r={r_clu_abs:+.4f} → "
              f"{'SUPPORTED' if h3_supported else 'PARTIALLY' if h3_bet or h3_clu else 'NOT SUPPORTED'}")

    h_results["H3"] = h3_data

    return h_results


def extract_dataset_stats():
    """Get dataset-level statistics."""
    print("\n[4/5] Extracting dataset statistics...")

    subset, y_cont = load_and_prepare_data()

    # Class balance
    y_binary = (y_cont > 0).astype(int)
    n_pos = int(y_binary.sum())
    n_neg = int(len(y_binary) - n_pos)

    # Sector distribution
    sic_col = "Current Acquirer SIC Code"
    sector_dist = {}
    if sic_col in subset.columns:
        sic_2 = subset[sic_col].astype(str).str[:2]
        # Map 2-digit SIC to sector names
        sic_sectors = {
            "01-09": "Agriculture", "10-14": "Mining", "15-17": "Construction",
            "20-39": "Manufacturing", "40-49": "Transport/Utilities",
            "50-51": "Wholesale", "52-59": "Retail", "60-67": "Finance/Insurance/RE",
            "70-89": "Services", "91-99": "Public Admin"
        }
        for label, name in sic_sectors.items():
            lo, hi = label.split("-")
            codes = [str(i).zfill(2) for i in range(int(lo), int(hi) + 1)]
            count = int(sic_2.isin(codes).sum())
            if count > 0:
                sector_dist[name] = count

    stats_data = {
        "total_deals": len(subset),
        "n_positive_car": n_pos,
        "n_negative_car": n_neg,
        "class_ratio": round(n_pos / max(n_neg, 1), 4),
        "car_mean": round(float(y_cont.mean()), 4),
        "car_std": round(float(y_cont.std()), 4),
        "car_median": round(float(np.median(y_cont)), 4),
        "car_min": round(float(y_cont.min()), 4),
        "car_max": round(float(y_cont.max()), 4),
        "n_financial_features": len([c for c in FINANCIAL_COLS if c in subset.columns]),
        "n_text_features": len([c for c in TEXT_COLS if c in subset.columns]),
        "n_graph_features": len([c for c in GRAPH_COLS if c in subset.columns]),
        "total_features_m3": len([c for c in FINANCIAL_COLS + TEXT_COLS + GRAPH_COLS if c in subset.columns]),
        "sector_distribution": sector_dist,
    }

    print(f"    {stats_data['total_deals']} deals, {n_pos}/{n_neg} pos/neg")
    print(f"    Features: {stats_data['n_financial_features']} fin + "
          f"{stats_data['n_text_features']} text + {stats_data['n_graph_features']} graph = "
          f"{stats_data['total_features_m3']} total")

    return stats_data


def build_pipeline_dag():
    """Define the script execution flow for the DAG visualization."""
    print("\n[5/5] Building pipeline DAG metadata...")

    dag = {
        "stages": [
            {
                "name": "Data Acquisition",
                "scripts": [
                    {"name": "pull_car_data.py", "desc": "Pulls CAR event study data from Bloomberg API"},
                    {"name": "generate_bbg_excel.py", "desc": "Generates Bloomberg Excel queries for financial data"},
                    {"name": "generate_splc_excel.py", "desc": "Generates Bloomberg supply-chain (SPLC) queries"},
                    {"name": "run_edgar_fetch.py", "desc": "Fetches SEC EDGAR 10-K filings (MD&A + Risk Factors)"},
                ],
            },
            {
                "name": "Data Cleaning",
                "scripts": [
                    {"name": "fix_dates.py", "desc": "Standardizes date formats across Bloomberg downloads"},
                    {"name": "retry_failed_tickers.py", "desc": "Re-attempts failed ticker lookups with fuzzy matching"},
                    {"name": "merge_bbg_data.py", "desc": "Merges multiple Bloomberg financial data files"},
                    {"name": "merge_splc_data.py", "desc": "Merges and deduplicates supply chain relationship data"},
                    {"name": "run_cleaning.py", "desc": "Master cleaning script: outliers, NaN, type casting"},
                ],
            },
            {
                "name": "Feature Engineering",
                "scripts": [
                    {"name": "compute_car.py", "desc": "Computes Cumulative Abnormal Returns [-5,+5] window"},
                    {"name": "run_text_features.py", "desc": "FinBERT encoding → PCA (64 dims) for MD&A and Risk Factors"},
                    {"name": "build_hetero_graph.py", "desc": "Constructs heterogeneous supply-chain graph (PyG HeteroData)"},
                    {"name": "train_hetero_graph.py", "desc": "Trains 2-layer HeteroGraphSAGE → 64-dim node embeddings"},
                    {"name": "merge_hetero_embeddings.py", "desc": "Merges graph embeddings back into the deal-level dataset"},
                    {"name": "build_combined_dataset.py", "desc": "Concatenates M1/M2/M3 feature matrices"},
                ],
            },
            {
                "name": "Model Training",
                "scripts": [
                    {"name": "train_ridge.py", "desc": "Ridge/ElasticNet regression baselines (5-fold CV)"},
                    {"name": "train_xgboost.py", "desc": "XGBoost regression (M1/M2/M3, 5-fold CV)"},
                    {"name": "train_mlp.py", "desc": "2-layer MLP regressor (M1/M2/M3, 5-fold CV)"},
                    {"name": "train_classifier.py", "desc": "Classification pipeline: LogReg/XGBoost/MLP with AUC"},
                    {"name": "tune_xgboost.py", "desc": "Bayesian hyperparameter tuning (Optuna) for XGBoost classifier"},
                ],
            },
            {
                "name": "Evaluation & Hypothesis Testing",
                "scripts": [
                    {"name": "test_h1.py", "desc": "H1: Graph embeddings help more in supply-chain sectors"},
                    {"name": "test_h2.py", "desc": "H2: MD&A alignment → +CAR, Risk Factor concentration → −CAR"},
                    {"name": "test_h3.py", "desc": "H3: High betweenness centrality → higher CAR variance"},
                    {"name": "verify_car.py", "desc": "Cross-validates CAR calculations against Bloomberg benchmarks"},
                ],
            },
        ],
    }

    print(f"    {len(dag['stages'])} pipeline stages, "
          f"{sum(len(s['scripts']) for s in dag['stages'])} scripts total")

    return dag


def main():
    print("=" * 70)
    print("  GENERATING macro_stats.json FOR M&A DEAL INTELLIGENCE TERMINAL")
    print("=" * 70)

    macro = {}

    # 1. Model performance
    macro["model_performance"] = extract_model_performance()

    # 2. SHAP distributions
    macro["shap_distributions"] = extract_shap_distributions()

    # 3. Hypothesis test results
    macro["hypothesis_tests"] = run_hypothesis_tests()

    # 4. Dataset statistics
    macro["dataset_stats"] = extract_dataset_stats()

    # 5. Pipeline DAG
    macro["pipeline_dag"] = build_pipeline_dag()

    # Write out
    out_path = "macro_stats.json"
    with open(out_path, "w") as f:
        json.dump(macro, f, indent=2)

    file_size = os.path.getsize(out_path)
    print(f"\n{'=' * 70}")
    print(f"  ✅ macro_stats.json written ({file_size:,} bytes)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
