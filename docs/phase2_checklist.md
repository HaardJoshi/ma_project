# Phase 2: Classification & Improvements — Checklist

## Step 1: Classification Reframe
- [ ] Create `train_classifier.py` with binary target (CAR > 0 vs ≤ 0)
- [ ] Baselines: Logistic Regression (M1, M2, M3)
- [ ] Primary: XGBoostClassifier (M1, M2, M3)
- [ ] Secondary: MLP with sigmoid (M1, M2, M3)
- [ ] Metrics: Accuracy, AUC-ROC, F1, Precision, Recall
- [ ] Statistical significance: McNemar's test (M1 vs M3)

## Step 2: Hyperparameter Tuning
- [ ] Install Optuna
- [ ] Tune XGBoostClassifier via Bayesian optimisation (100 trials)
- [ ] Re-run best config across M1/M2/M3

## Step 3: Feature Engineering
- [ ] Add interaction features (acquirer/target ratios)
- [ ] Add sector indicators (if SIC codes available)
- [ ] Re-run best model with engineered features

## Step 4: Analysis & Documentation
- [ ] SHAP values for best classifier
- [ ] Dissertation notes: classification findings
- [ ] Update `phase2_results_and_findings.md`
- [ ] Commit and push
