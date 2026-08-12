# Day 4 — Feature Engineering & Hyperparameter Tuning

Welcome to Day 4 of Week 4. This session combines two of the most effective ways to improve a model's performance: **Feature Engineering** — deriving new, domain-relevant features that expose stronger predictive signals — and **Hyperparameter Tuning** — systematically searching for the best configuration of a Random Forest with `GridSearchCV` under strict 5-fold Stratified Cross-Validation, and benchmarking the tuned model against the untuned Week 3 baseline.

---

## 🎯 Objective

Improve model performance without touching the test set, by:

- constructing at least **two new domain-relevant engineered features** and justifying their mathematical/domain rationale
- defining a **structured hyperparameter grid** for a `RandomForestClassifier`
- running **`GridSearchCV`** with 5-Fold Stratified Cross-Validation (`StratifiedKFold`) to find the optimal combination
- **benchmarking** the tuned model's cross-validated score against the untuned baseline model from Week 3
- analyzing **feature importances** and **hyperparameter impact** to interpret what drove the improvement

---

## 📓 Notebook

- [Feature_Engineering_and_Tuning.ipynb](./Feature_Engineering_and_Tuning.ipynb)

---

## ✅ Key Tasks & Accomplishments

- Generated a synthetic customer-behavior classification dataset (1,000 samples, 6 features) simulating **Telecom Churn / Credit Risk** with `make_classification` (4 informative, 1 redundant, `n_clusters_per_class=2`, 65% / 35% class weights) and a fixed `SEED = 42`. Renamed the raw columns into domain-meaningful features — `Monthly_Charges`, `Total_Tenure_Months`, `Support_Calls`, `Usage_GB`, `Payment_Delay_Days`, `Age` — and scaled values into realistic positive ranges. Target: **Churn**.
- **Step 1 — Feature Engineering & Justification (6 → 8 features):**
  - **`Total_Estimated_Spend`** (`Monthly_Charges × Total_Tenure_Months`) — monthly charge or tenure alone doesn't reflect the customer's lifetime financial relationship; multiplying them creates a cumulative **Customer Lifetime Value (CLV)** metric that lets the model distinguish high-value long-term clients.
  - **`Support_Calls_Per_Tenure_Month`** (`Support_Calls / Total_Tenure_Months`) — 5 support calls in 2 months is a much stronger churn signal than 5 calls over 50 months; normalizing by tenure creates a **"Frustration Density Index"**.
  - Performed a stratified **80/20 Train/Test split** (800 train / 200 test samples) on the engineered feature set.
- **Steps 2 & 3 — Hyperparameter Grid & GridSearchCV:**
  - Established the **Week 3 baseline**: an untuned default `RandomForestClassifier(random_state=42)` evaluated on the **original raw features** via 5-Fold Stratified CV → **Mean F1-Score 0.8830 ± 0.0366**.
  - Defined a grid over the key Random Forest hyperparameters: `n_estimators: [50, 100, 200]`, `max_depth: [None, 5, 10]`, `min_samples_split: [2, 5, 10]`, `max_features: ['sqrt', 'log2']` → **54 candidates × 5 folds = 270 fits**.
  - Ran `GridSearchCV` on the **engineered data** with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, `scoring='f1'`, and `n_jobs=-1`.
  - **Best Parameters:** `{'max_depth': 10, 'max_features': 'log2', 'min_samples_split': 2, 'n_estimators': 50}` → **Best Cross-Validated F1-Score 0.8941**.
- **Step 4 — Model Comparison & Final Test Evaluation:**

  | Model Variant | 5-Fold CV F1-Score | Test Set F1-Score | F1 Improvement |
  | :--- | :---: | :---: | :---: |
  | Untuned Baseline RF (Raw Data) | 0.8830 ± 0.0366 | 0.8372 | 0.00% |
  | GridSearchCV Tuned RF (Engineered) | 0.8941 | 0.8636 | **+3.16%** |

  Visualized the comparison (untuned vs. tuned) across CV and test scores with Matplotlib.
- **Step 5 — Findings & Diagnostic Documentation:**
  - **Most influential engineered feature:** `Total_Estimated_Spend` ranked **5th overall** with a Gini importance of **11.11%**, versus `Support_Calls_Per_Tenure_Month` at **2.82%** (7th) — the cumulative spending signal carried far more predictive information than the support-call ratio.
  - **Strongest predictors overall:** Age (**23.83%**), Monthly_Charges (**22.04%**), Total_Tenure_Months (**19.34%**), Usage_GB (**13.41%**).
  - **Hyperparameter impact:** `max_depth: 10` + `max_features: 'log2'` restricted tree depth and feature subspace, controlling complexity and reducing overfitting vs. the default baseline — test F1 rose from **0.8372 to 0.8636 (+3.16%)**.
  - **Conclusion:** domain-driven feature engineering combined with GridSearchCV tuning delivered a **+3.16% Test F1 improvement** over the raw baseline, with all tuning performed strictly under cross-validation to prevent data leakage.

---

## 🛠️ Skills Covered

- **Domain-driven feature engineering** — deriving features from business/domain rationale (CLV, frustration density)
- **Mathematical justification of features** — multiplication and normalization transformations
- **Hyperparameter grid definition** — `n_estimators`, `max_depth`, `min_samples_split`, `max_features`
- **`GridSearchCV`** — exhaustive search over 54 candidates (270 fits) with `n_jobs=-1`
- **5-Fold Stratified Cross-Validation** — `StratifiedKFold` for the imbalanced 65/35 target
- **Leak-free tuning** — test set untouched until the final single evaluation
- **Model comparison** — untuned baseline vs. tuned model on both CV and test F1
- **Feature importance interpretation** — Gini importances to explain which features mattered
- **Hyperparameter impact analysis** — connecting `max_depth` / `max_features` to overfitting control
- **Result documentation** — comparison table, importance breakdown, and final conclusions

---

## 🔗 Related

- [Week 4 Overview](../README.md)
- [Root Repository README](../../README.md)
