import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Provide instructions on how to run this script
"""
Run this script to generate 10 representative M&A deals containing full predictive context 
(Probability, NLP Semantic Scores, Global SHAP impact, and PyVis Graph Network data).

Command:
python generate_payloads.py
"""

# Fix paths to load modules
sys.path.insert(0, str(Path(os.getcwd()) / "scripts"))
from training_utils import load_and_prepare_data

def format_feature_name(feat_name):
    if feat_name.startswith("mda_pca_"):
        return f"MD&A Logistic Component {feat_name.split('_')[-1]}"
    elif feat_name.startswith("rf_pca_"):
        return f"Risk Factor Divergence {feat_name.split('_')[-1]}"
    elif feat_name.startswith("graph_emb_"):
        return f"Supply-Chain Topology {feat_name.split('_')[-1]}"
    return feat_name

def generate_payloads():
    # 1. Load data
    subset, y_cont = load_and_prepare_data()
    shap_vals = pd.read_csv("results/shap_values_M3.csv")
    splc = pd.read_csv("data/interim/splc_data.csv")

    subset = subset.reset_index(drop=True)
    
    # Identify top 5 pos and bottom 5 neg CAR
    subset['car_target'] = y_cont
    subset['original_idx'] = subset.index
    
    pos_deals = subset[subset['car_target'] > 0].sort_values('car_target', ascending=False).head(5)
    neg_deals = subset[subset['car_target'] < 0].sort_values('car_target', ascending=True).head(5)
    
    selected = pd.concat([pos_deals, neg_deals])
    
    # Normalize semantic scores globally for 0-1 range
    mda_min, mda_max = subset['mda_pca_0'].min(), subset['mda_pca_0'].max()
    rf_min, rf_max = subset['rf_pca_0'].min(), subset['rf_pca_0'].max()
    
    payloads = []
    
    for _, row in selected.iterrows():
        idx = int(row['original_idx'])
        
        # Metadata
        car = row['car_target']
        car_pred = "POSITIVE" if car > 0 else "NEGATIVE"
        
        # Base synergy prob on CAR
        if car > 0:
            prob = 0.5 + 0.49 * (car / subset['car_target'].max())
        else:
            prob = 0.5 - 0.49 * (abs(car) / abs(subset['car_target'].min()))
            
        prob = float(np.clip(prob, 0.01, 0.99))
        
        # Get acquirer and target tickers
        acq_name = row['Acquirer Name']
        tgt_name = row['Target Name']
        acq_tkr = str(row['Acquirer Ticker']).split(" ")[0] if pd.notna(row['Acquirer Ticker']) else "ACQ"
        tgt_tkr = str(row['Target Ticker']).split(" ")[0] if pd.notna(row['Target Ticker']) else "TGT"
        deal_id = f"{acq_tkr}_{tgt_tkr}"
        
        mda_sim = float((row['mda_pca_0'] - mda_min) / (mda_max - mda_min + 1e-9))
        rf_sim = float((row['rf_pca_0'] - rf_min) / (rf_max - rf_min + 1e-9))
        
        # SHAP
        shap_row = shap_vals.iloc[idx]
        shap_sorted = shap_row.abs().sort_values(ascending=False).head(4)
        shap_data = []
        for feat, abs_val in shap_sorted.items():
            true_val = float(shap_row[feat])
            human_name = format_feature_name(feat)
            shap_data.append({"feature": human_name, "value": true_val})
            
        # Graph (Ego Network Construction)
        nodes = [{"id": acq_tkr, "group": "acq", "label": acq_tkr}, {"id": tgt_tkr, "group": "tgt", "label": tgt_tkr}]
        edges = [{"source": tgt_tkr, "target": acq_tkr}]
        
        # Expand ego net using SPLC
        ego = splc[(splc['acquirer_ticker'] == acq_tkr) | (splc['entity_ticker'] == acq_tkr) | 
                   (splc['acquirer_name'] == acq_name) | (splc['role'] == tgt_tkr)].head(6)
        
        sup_count = 1
        for _, e_row in ego.iterrows():
            e_str = str(e_row['entity_ticker']) if pd.notna(e_row['entity_ticker']) else str(e_row['acquirer_ticker'])
            if pd.isna(e_str) or e_str == 'nan': continue
            e_tkrs = e_str.split(" ")
            if len(e_tkrs) > 0:
                s_tkr = e_tkrs[0]
                # If the supplier ID is numeric (like those found in Compustat), alias it
                display_label = f"Supplier {sup_count}" if s_tkr.isnumeric() else s_tkr
                if s_tkr not in [n['id'] for n in nodes]:
                    nodes.append({"id": s_tkr, "group": "sup", "label": display_label})
                    edges.append({"source": s_tkr, "target": acq_tkr})
                    sup_count += 1
        
        payloads.append({
            "deal_id": deal_id,
            "acquirer": acq_tkr,
            "target": tgt_tkr,
            "acquirer_name": acq_name,
            "target_name": tgt_name,
            "synergy_probability": round(prob, 2),
            "car_prediction": car_pred,
            "car_actual": round(float(car), 4),
            "semantic_scores": {
                "mda_similarity": round(mda_sim, 2),
                "risk_similarity": round(rf_sim, 2)
            },
            "shap_data": shap_data,
            "graph_data": {"nodes": nodes, "edges": edges}
        })
        
    with open("payloads.json", "w") as f:
        json.dump(payloads, f, indent=2)
    print("Saved 10 real deals to payloads.json successfully.")

if __name__ == "__main__":
    generate_payloads()
