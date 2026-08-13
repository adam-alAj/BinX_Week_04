# Day 5 — Tuned End-to-End Leakage-Free ML Pipeline Mini-Project

Welcome to Day 5 of Week 4 — the mini-project that brings the whole week together! This session builds a complete, **end-to-end, leakage-free Machine Learning pipeline** that predicts customer churn, combining everything learned in Days 1–4: three-way splitting, cross-validation, the bias–variance tradeoff, feature engineering, and hyperparameter tuning — all wrapped into a single `scikit-learn` `Pipeline` and tuned with `GridSearchCV` under strict 5-Fold Stratified Cross-Validation.

---

## 🎯 Objective

Build a full data-to-model pipeline that:

- encapsulates **preprocessing** (`StandardScaler` + `OneHotEncoder`) inside a `ColumnTransformer`
- integrates **domain-specific engineered features** directly into the automated transformation workflow using `FunctionTransformer` (zero manual leakage risk)
- tunes the **entire end-to-end pipeline** with `GridSearchCV` under strict 5-Fold `StratifiedKFold` cross-validation
- evaluates the final tuned model **exactly once** on the untouched test set
- benchmarks the tuned pipeline against a **DummyClassifier (majority-class)** baseline and an **untuned default pipeline** baseline
- records the final results, key takeaways, and the GitHub commit workflow

---

## 📓 Notebook

- [Tuned_End_to_End_Pipeline.ipynb](./Tuned_End_to_End_Pipeline.ipynb)

---

## ✅ Key Tasks & Accomplishments

- Generated a synthetic **Churn / Retention dataset** (1,000 samples, 5 features) with `np.random.seed(42)` — numeric: `Total_Tenure_Months`, `Monthly_Charges`, `Support_Calls`; categorical: `Contract_Type` (Month-to-Month 50% / One-Year 30% / Two-Year 20%), `Payment_Method` (Electronic / Mailed Check / Bank Transfer). The `Churn` target was derived from realistic interactions (`0.35 × support-call density + 0.25 × Month-to-Month + 0.20 × charges > $80`, binarized at the 65th percentile → **65.2% Retained / 34.8% Churned**).
- Performed a **strict stratified 80/20 Train/Test split** (800 train / 200 test) with `random_state=42` **before any preprocessing** — the test set is never touched until the final evaluation.
- **Steps 1 & 2 — Pipeline Architecture & Feature Engineering:**
  - Wrapped a custom `add_engineered_features()` function in a `FunctionTransformer(validate=False)` that dynamically creates **two new features** inside the pipeline:
    - **`Support_Calls_Per_Tenure_Month`** — the "Frustration Density Index" (`Support_Calls / Total_Tenure_Months`).
    - **`High_Value_Customer`** — binary flag for `Monthly_Charges > $75`.
  - Built a `ColumnTransformer` preprocessor: **numeric** columns (raw + engineered) → `StandardScaler`; **categorical** columns (`Contract_Type`, `Payment_Method`) → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - Assembled the end-to-end pipeline: `feature_engineering → preprocessing → RandomForestClassifier(random_state=42)`.
- **Step 3 — End-to-End Hyperparameter Tuning (5-Fold CV):**
  - Defined the search space with the `classifier__` prefix: `n_estimators [50, 100, 200]`, `max_depth [3, 5, 10, None]`, `min_samples_split [2, 5, 10]` → **36 candidates × 5 folds = 180 fits**.
  - Ran `GridSearchCV` on the full pipeline with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, `scoring='f1'`, `n_jobs=-1` — every preprocessing step (scaling, encoding, feature creation) is fitted **strictly inside each fold's training split**, eliminating all structural data leakage.
  - **Best 5-Fold Cross-Validated F1-Score: 0.9945** with best hyperparameters `{'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 200}`.
- **Step 4 — Honest Test Set Evaluation & Baseline Comparison:**

  | Model Variant | Test Accuracy | Test F1-Score |
  | :--- | :---: | :---: |
  | Dummy Baseline (Majority Class) | 0.650 | 0.0000 |
  | Untuned Default Pipeline | 0.965 | 0.9504 |
  | **Tuned End-to-End Pipeline** | **0.975** | **0.9645** |

  - Printed the final classification report for the tuned pipeline — Retained (0): Precision 0.98 / Recall 0.98 / F1 0.98 (130 samples); Churned (1): Precision 0.96 / Recall 0.97 / F1 0.96 (70 samples); overall **Accuracy 0.97**.
  - Visualized the **Confusion Matrix heatmap** for the tuned pipeline on the test set.
- **Step 5 — Summary & Key Findings:**
  1. **Leakage-Free Guarantee:** encapsulating scaling, one-hot encoding, and feature creation inside the `Pipeline` ensured that no test-set or validation-set statistics ever infected model training.
  2. **Value-Add Demonstration:** the Tuned Pipeline significantly outperformed both the Dummy Baseline (F1 0.9645 vs 0.0000) and the Untuned Default Pipeline (F1 0.9645 vs 0.9504, Accuracy 0.975 vs 0.965).
  3. **Generalization:** controlling tree depth via `max_depth = 5` stabilized performance across CV folds and prevented overfitting.

---

## 🛠️ Skills Covered

- End-to-end pipeline design (`feature_engineering → preprocessing → classifier`) with `Pipeline`
- Dynamic in-pipeline feature engineering with `FunctionTransformer` (Frustration Density Index, High-Value Customer flag)
- `ColumnTransformer` for mixed-type preprocessing (`StandardScaler` + `OneHotEncoder(handle_unknown='ignore')`)
- Leakage-free hyperparameter tuning of the whole pipeline via `GridSearchCV` with `classifier__` prefixed parameters
- 5-Fold Stratified Cross-Validation (`StratifiedKFold`) with F1 scoring
- Honest, single-use test-set evaluation vs. `DummyClassifier` and untuned baselines
- Confusion Matrix + classification report interpretation (Precision, Recall, F1, Accuracy)
- Result documentation — comparison table, best-parameter selection, and key takeaways

---

## 🔗 Related

- [Week 4 Overview](../README.md)
- [Root Repository README](../../README.md)
