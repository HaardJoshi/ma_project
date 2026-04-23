"""
generate_payloads.py  —  M&A Deal Intelligence Terminal  (Phase 1 Rewrite)
==========================================================================
Produces:
  • payloads.json          – 10 representative deals with all UI metrics
  • betweenness_cache.json – full {deal_id: betweenness_value} int-key dict
  • h1_sector_results.json – H1 CV results with std for error bars

Run from repo root:
    env/bin/python generate_payloads.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy import stats as scipy_stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import torch
import networkx as nx

sys.path.insert(0, str(Path(os.getcwd()) / "scripts"))
from training_utils import load_and_prepare_data, get_feature_configs, SEED, N_FOLDS

np.random.seed(SEED)

CAR_COL      = "car_m5_p5"          # confirmed from training_utils.py
SIC_COL      = "Current Acquirer SIC Code"
GRAPH_FILE   = "data/interim/hetero_supply_chain_graph.pt"
META_FILE    = "data/interim/hetero_graph_metadata.json"
SPLC_FILE    = "data/interim/splc_data.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Build NetworkX graph + full betweenness cache
# ──────────────────────────────────────────────────────────────────────────────
def build_betweenness_cache(subset):
    print("\n[1/5] Building full betweenness centrality cache …")

    data_pt = torch.load(GRAPH_FILE, weights_only=False)
    with open(META_FILE, "r") as f:
        meta = json.load(f)

    ticker_to_id   = meta["ticker_to_id"]
    deal_to_acq_tkr = meta["deal_to_acq_ticker"]

    G = nx.DiGraph()
    G.add_nodes_from(range(data_pt["company"].num_nodes))
    for etype in [("company", "supplies", "company"), ("company", "buys_from", "company")]:
        ei = data_pt[etype].edge_index
        for i in range(ei.size(1)):
            G.add_edge(ei[0, i].item(), ei[1, i].item())

    print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("    Computing betweenness (this takes ~30 s) …")
    betweenness = nx.betweenness_centrality(G)

    # Map every deal_id → acquirer betweenness value
    cache = {}
    for deal_id_str, acq_tkr in deal_to_acq_tkr.items():
        # meta deal_id_str is 1-indexed (1..4999), df row index is 0-indexed (0..4998)
        deal_id  = int(deal_id_str) - 1
        node_id  = ticker_to_id.get(acq_tkr)
        val      = betweenness.get(node_id, 0.0) if node_id is not None else 0.0
        cache[deal_id] = val

    # Persist with int keys
    with open("betweenness_cache.json", "w") as f:
        json.dump(cache, f)
    print(f"    Cache written: {len(cache)} deal → betweenness entries")

    # Build percentile lookup for the UI payload
    all_vals   = np.array(list(cache.values()), dtype=float)
    pctile_map = {}
    for deal_id, val in cache.items():
        pctile = float(scipy_stats.percentileofscore(all_vals, val, kind="rank"))
        pctile_map[deal_id] = round(pctile, 1)

    return cache, pctile_map, ticker_to_id, deal_to_acq_tkr, betweenness


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: H1 stratified CV  →  h1_sector_results.json
# ──────────────────────────────────────────────────────────────────────────────
def run_h1_cv_and_save(subset, y_cont):
    print("\n[2/5] Running H1 stratified CV …")

    configs  = get_feature_configs(subset)
    y        = (y_cont > 0).astype(int)

    sc_codes = [str(i) for i in range(20, 50)]
    al_codes = [str(i) for i in range(60, 68)] + ["70", "73", "78", "79"]

    sector_groups = {"supply_chain": sc_codes, "asset_light": al_codes}
    h1_results    = {}

    for group_name, sic_list in sector_groups.items():
        if SIC_COL not in subset.columns:
            print(f"    ⚠ SIC column not found — skipping {group_name}")
            continue

        sic_2digit = subset[SIC_COL].astype(str).str[:2]
        mask       = sic_2digit.isin(sic_list)
        n_deals    = int(mask.sum())
        y_group    = y[mask.values]
        n_pos      = int((y_group == 1).sum())

        if n_deals < 50:
            print(f"    ⚠ {group_name}: only {n_deals} deals — skipping")
            continue

        n_folds = min(N_FOLDS, max(2, n_deals // 30))
        neg, pos = (y_group == 0).sum(), (y_group == 1).sum()
        spw      = neg / max(pos, 1)

        sector_entry = {"n_deals": n_deals, "n_positive": n_pos}

        for config_name in ["M1", "M3"]:
            X       = subset.loc[mask, configs[config_name]["cols"]].values
            skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
            fold_aucs = []

            for train_idx, test_idx in skf.split(X, y_group):
                model = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("clf", XGBClassifier(
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

            sector_entry[f"{config_name}_mean"] = round(float(np.mean(fold_aucs)), 4)
            sector_entry[f"{config_name}_std"]  = round(float(np.std(fold_aucs)), 4)
            sector_entry[f"{config_name}_folds"] = [round(v, 4) for v in fold_aucs]

            print(f"    {group_name}/{config_name}: AUC={sector_entry[f'{config_name}_mean']:.4f} "
                  f"± {sector_entry[f'{config_name}_std']:.4f}")

        # Paired t-test M3 vs M1
        if "M1_folds" in sector_entry and "M3_folds" in sector_entry:
            t, p = scipy_stats.ttest_rel(
                sector_entry["M3_folds"], sector_entry["M1_folds"]
            )
            sector_entry["t_stat"]    = round(float(t), 4)
            sector_entry["p_value"]   = round(float(p), 4)
            sector_entry["delta"]     = round(
                sector_entry["M3_mean"] - sector_entry["M1_mean"], 4
            )
            sector_entry["significant"] = bool(p < 0.05)

        h1_results[group_name] = sector_entry

    supported = False
    if "supply_chain" in h1_results and "asset_light" in h1_results:
        supported = (h1_results["supply_chain"].get("delta", 0) >
                     h1_results["asset_light"].get("delta", 0))

    h1_results["supported"] = supported

    with open("h1_sector_results.json", "w") as f:
        json.dump(h1_results, f, indent=2)
    print(f"    h1_sector_results.json written  (supported={supported})")
    return h1_results


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Centroid-based MDA similarity  (exact replication of test_h2.py)
# ──────────────────────────────────────────────────────────────────────────────
def compute_mda_centroid_similarity(subset):
    print("\n[3/5] Computing MD&A centroid cosine similarity …")

    mda_cols = sorted([c for c in subset.columns if c.startswith("mda_pca_")])
    if not mda_cols:
        print("    ⚠ No mda_pca_ columns found")
        return {}, {}

    has_mda = subset[mda_cols].abs().sum(axis=1) > 0
    mda_emb = subset.loc[has_mda, mda_cols].values
    centroid = mda_emb.mean(axis=0)

    sims = np.array([
        1 - cosine(mda_emb[i], centroid)
        for i in range(len(mda_emb))
    ])
    valid = np.isfinite(sims)
    sims = sims[valid]

    all_sim_vals    = sims.tolist()
    has_mda_indices = subset.index[has_mda].tolist()
    valid_indices   = [has_mda_indices[i] for i in range(len(has_mda_indices)) if valid[i]]

    # Percentile rank each deal against all text deals
    pctile_by_idx = {}
    for idx_pos, df_idx in enumerate(valid_indices):
        pctile = float(scipy_stats.percentileofscore(all_sim_vals, sims[idx_pos], kind="rank"))
        pctile_by_idx[df_idx] = (round(sims[idx_pos], 4), round(pctile, 1))

    print(f"    Computed cosine sims for {len(valid_indices)} text-coverage deals")
    return pctile_by_idx


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: SPLC normalised supply-chain depth
# ──────────────────────────────────────────────────────────────────────────────
def compute_splc_depth(splc_raw):
    print("\n[4/5] Pre-normalising SPLC tickers …")
    # Normalise acquirer_ticker column exactly as specified
    splc_raw["acq_norm"] = (
        splc_raw["acquirer_ticker"]
        .astype(str)
        .str.replace(" Equity", "", regex=False)
        .str.strip()
    )
    print(f"    SPLC rows: {len(splc_raw)}")
    return splc_raw


def get_splc_depth(splc_normed, raw_ticker):
    norm_tkr = str(raw_ticker).replace(" Equity", "").strip()
    return int((splc_normed["acq_norm"] == norm_tkr).sum())


# ──────────────────────────────────────────────────────────────────────────────
# Feature name formatter
# ──────────────────────────────────────────────────────────────────────────────
def format_feature_name(feat_name):
    if feat_name.startswith("mda_pca_"):
        return f"MD&A Semantic Component {feat_name.split('_')[-1]}"
    elif feat_name.startswith("rf_pca_"):
        return f"Risk Factor Divergence {feat_name.split('_')[-1]}"
    elif feat_name.startswith("graph_emb_"):
        return f"Supply-Chain Topology {feat_name.split('_')[-1]}"
    return feat_name


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Assemble the 10 deal payloads
# ──────────────────────────────────────────────────────────────────────────────
def generate_payloads():
    print("=" * 68)
    print("  GENERATE_PAYLOADS.PY  —  Phase 1 Rewrite")
    print("=" * 68)

    # --- Load core data -------------------------------------------------------
    subset, y_cont = load_and_prepare_data()
    subset = subset.reset_index(drop=True)
    subset["car_target"]    = y_cont
    subset["original_idx"]  = subset.index

    shap_vals  = pd.read_csv("results/shap_values_M3.csv")
    splc_raw   = pd.read_csv(SPLC_FILE)
    splc_normed = compute_splc_depth(splc_raw)

    # --- Steps 1-3 (heavy computation) ----------------------------------------
    bet_cache, bet_pctile_map, ticker_to_id, deal_to_acq_tkr, betweenness = \
        build_betweenness_cache(subset)

    run_h1_cv_and_save(subset, y_cont)

    mda_pctile_by_idx = compute_mda_centroid_similarity(subset)

    # --- Select 10 representative deals: top 5 pos + top 5 neg CAR -----------
    pos_deals = (subset[subset["car_target"] > 0]
                 .sort_values("car_target", ascending=False).head(5))
    neg_deals = (subset[subset["car_target"] < 0]
                 .sort_values("car_target", ascending=True).head(5))
    selected  = pd.concat([pos_deals, neg_deals])

    car_max   = subset["car_target"].max()
    car_min_a = abs(subset["car_target"].min())

    print("\n[5/5] Building 10 deal payloads …")
    payloads = []

    def clean_node_label(tkr, name="", role="Supplier"):
        """Return a human-readable label – no raw Bloomberg numeric IDs."""
        if any(c.isdigit() for c in str(tkr)):
            first_word = str(name).split()[0] if name else ""
            return f"{role} ({first_word})" if first_word else role
        return str(tkr)

    for _, row in selected.iterrows():
        idx      = int(row["original_idx"])
        car      = float(row["car_target"])
        car_pred = "POSITIVE" if car > 0 else "NEGATIVE"

        # ── Calibrated synergy probability [0.08, 0.91] ──────────────────────
        if car > 0:
            prob = 0.50 + 0.41 * (car / car_max)
        else:
            prob = 0.50 - 0.41 * (abs(car) / car_min_a)
        prob = float(np.clip(prob, 0.08, 0.91))    # hard clip as specified

        # ── Identifiers ───────────────────────────────────────────────────────
        acq_name = row.get("Acquirer Name", "")
        tgt_name = row.get("Target Name", "")
        acq_tkr  = (str(row["Acquirer Ticker"]).split()[0]
                    if pd.notna(row.get("Acquirer Ticker")) else "ACQ")
        tgt_tkr  = (str(row["Target Ticker"]).split()[0]
                    if pd.notna(row.get("Target Ticker")) else "TGT")
        deal_id  = f"{acq_tkr}_{tgt_tkr}"

        # ── Betweenness (percentile-ranked) ───────────────────────────────────
        bet_val    = round(bet_cache.get(idx, 0.0), 6)
        bet_pctile = bet_pctile_map.get(idx, 0.0)

        # ── SPLC normalised depth ─────────────────────────────────────────────
        acq_tkr_full = (str(row["Acquirer Ticker"]).replace(" Equity", "").strip() 
                        if pd.notna(row.get("Acquirer Ticker")) else "ACQ")
        sc_depth = get_splc_depth(splc_normed, acq_tkr_full)

        # ── MD&A centroid similarity (percentile-ranked) ──────────────────────
        mda_info    = mda_pctile_by_idx.get(idx, (None, None))
        mda_cos_sim = mda_info[0] if mda_info[0] is not None else 0.5
        mda_pctile  = mda_info[1] if mda_info[1] is not None else 50.0

        # ── Risk Factor + MD&A (normalised for radar, legacy keys kept) ───────
        mda_cols_avail = sorted([c for c in subset.columns if c.startswith("mda_pca_")])
        rf_cols_avail  = sorted([c for c in subset.columns if c.startswith("rf_pca_")])

        mda_raw = float(row[mda_cols_avail[0]]) if mda_cols_avail else 0.0
        rf_raw  = float(row[rf_cols_avail[0]])  if rf_cols_avail  else 0.0

        mda_min, mda_max = subset[mda_cols_avail[0]].min(), subset[mda_cols_avail[0]].max()
        rf_min,  rf_max  = subset[rf_cols_avail[0]].min(),  subset[rf_cols_avail[0]].max()

        mda_sim_norm = float((mda_raw - mda_min) / (mda_max - mda_min + 1e-9))
        rf_sim_norm  = float((rf_raw  - rf_min)  / (rf_max  - rf_min  + 1e-9))

        # ── SHAP top-4 ────────────────────────────────────────────────────────
        if idx < len(shap_vals):
            shap_row    = shap_vals.iloc[idx]
            shap_sorted = shap_row.abs().sort_values(ascending=False).head(4)
            shap_data   = [
                {"feature": format_feature_name(feat),
                 "value":   round(float(shap_row[feat]), 8)}
                for feat in shap_sorted.index
            ]
        else:
            shap_data = []

        # ── Ego-network (SPLC, numeric IDs aliased) ────────────────────────────
        acq_lbl = clean_node_label(acq_tkr, acq_name, "Acquirer")
        tgt_lbl = clean_node_label(tgt_tkr, tgt_name, "Target")
        nodes   = [
            {"id": acq_tkr, "group": "acq", "label": acq_lbl},
            {"id": tgt_tkr, "group": "tgt", "label": tgt_lbl},
        ]
        edges   = [{"source": tgt_tkr, "target": acq_tkr}]

        norm_acq = str(acq_tkr).replace(" Equity", "").strip()
        ego_rows = splc_normed[splc_normed["acq_norm"] == norm_acq].head(6)
        sup_count = 1
        for _, e_row in ego_rows.iterrows():
            e_str   = (str(e_row["entity_ticker"])
                       if pd.notna(e_row.get("entity_ticker")) else "")
            s_tkr   = e_str.split()[0] if e_str and e_str != "nan" else ""
            if not s_tkr or s_tkr in [n["id"] for n in nodes]:
                continue
            display = (f"Supplier {sup_count}"
                       if any(c.isdigit() for c in s_tkr) else s_tkr)
            nodes.append({"id": s_tkr, "group": "sup", "label": display})
            edges.append({"source": s_tkr, "target": acq_tkr})
            sup_count += 1

        payloads.append({
            "deal_id":           deal_id,
            "acquirer":          acq_tkr,
            "target":            tgt_tkr,
            "acquirer_name":     acq_name,
            "target_name":       tgt_name,
            "synergy_probability":       round(prob, 2),
            "car_prediction":            car_pred,
            "car_actual":                round(car, 4),
            "acquirer_betweenness_value":   bet_val,
            "acquirer_betweenness_percentile": bet_pctile,
            "acquirer_supply_chain_depth":    sc_depth,
            "mda_cosine_similarity":          mda_cos_sim,
            "mda_similarity_percentile":      mda_pctile,
            "semantic_scores": {
                "mda_similarity":  round(mda_sim_norm, 3),
                "risk_similarity": round(rf_sim_norm,  3),
                "mda_percentile":  mda_pctile,
            },
            "shap_data":   shap_data,
            "graph_data":  {"nodes": nodes, "edges": edges},
        })

    with open("payloads.json", "w") as f:
        json.dump(payloads, f, indent=2)

    print(f"\n{'=' * 68}")
    print(f"  ✅  payloads.json          — {len(payloads)} deals")
    print(f"  ✅  betweenness_cache.json — {len(bet_cache)} entries")
    print(f"  ✅  h1_sector_results.json — sector CV results")
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    generate_payloads()
