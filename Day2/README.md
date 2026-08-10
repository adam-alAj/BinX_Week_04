# Day 2 — Cross-Validating a Model (5-Fold Cross-Validation)

Welcome to Day 2 of Week 4. This session replaces the single train/validation split with **5-Fold Cross-Validation**, producing a more robust and unbiased performance estimate (`Mean ± Std`) for a classification model, and justifying why **Stratified K-Fold** is essential for imbalanced data.

---

## 🎯 Objective

Perform a reliable model evaluation that covers:

- training and evaluating a classification model using 5-Fold Cross-Validation (manual fold loop)
- reporting the **mean** and **standard deviation** of the validation scores across all 5 folds (`Mean ± Std`)
- comparing the Cross-Validated score against a single train/validation split score and analyzing the differences
- verifying that **Stratified K-Fold** is used for classification and justifying its importance in maintaining class proportion

---

## 📓 Notebook

- [Cross_Validation.ipynb](./Cross_Validation.ipynb)

---

## ✅ Key Tasks & Accomplishments

- Generated a synthetic customer-churn classification dataset (1,000 samples, 10 features, 6 informative, 2 redundant, 80% Non-Churn / 20% Churn) with `make_classification` and a fixed `SEED = 42`.
- **Step 1 & Step 2 — 5-Fold Cross-Validation (manual fold loop):** Initialized `StandardScaler()` and `LogisticRegression(random_state=SEED)` separately — no `Pipeline` — and ran a manual 5-fold loop over `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`: fit the scaler on each training fold only, scale both folds, train the model, and score with the weighted F1 metric.
- Reported every fold: **Fold F1-scores 0.8543, 0.8268, 0.8729, 0.8333, 0.8638** → **Mean F1 0.8502, Std ± 0.0176** → **Final Estimate: 0.8502 ± 0.0176**. The small standard deviation confirms the model is stable across all 5 data subsets.
- **Step 3 — CV vs. Single Split:** Computed a single stratified 80/20 train/validation split (Day 1 approach) — **Single-Split F1 0.8227** vs. **CV Mean 0.8502** (difference **−0.0275**). The single split was slightly pessimistic because that particular 20% hold-out happened to be harder than average; cross-validation uses **every sample in the validation set exactly once**, giving a far more trustworthy generalization estimate.
- Visualized the comparison with Matplotlib — a line plot of the 5 fold scores against the CV mean (green dashed) and the single-split score (red dotted).
- **Step 4 — Stratified K-Fold Justification:** Verified that `StratifiedKFold` is `scikit-learn`'s default cross-validation strategy for classification (used automatically when passing an integer like `cv=5`), and explicitly defined it in our code. Documented why stratification matters:
  - **Class Ratio Preservation** — the dataset is imbalanced (80% Class 0 / 20% Class 1); plain `KFold` could create folds with as little as 5% or as much as 35% minority samples.
  - **Elimination of Evaluation Bias** — every fold retains the exact 80/20 class proportions of the full dataset.
  - **Prevention of Training Failures** — without stratification a fold could miss the minority class entirely, making F1 computation invalid or extremely noisy.

---

## 🛠️ Skills Covered

- Manual K-Fold Cross-Validation loop over `StratifiedKFold`
- `StratifiedKFold` for imbalanced classification problems
- Leak-free scaling — fitting `StandardScaler` inside each training fold (no `Pipeline`)
- Performance reporting as `Mean ± Std` and interpreting standard deviation
- Comparing cross-validated vs. single-split estimates
- Class proportion preservation and bias elimination via stratification
- Data leakage prevention by fitting the scaler inside each fold

---

## 🔗 Related

- [Week 4 Overview](../README.md)
- [Root Repository README](../../README.md)
