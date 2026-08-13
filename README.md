# Week 4: Evaluation, Tuning & Pipelines

**Welcome to Week 4 of the BinX Tech AI & Machine Learning Internship Program.**
This week moves from building models to evaluating them properly: partitioning data into training, validation, and test sets, tuning hyperparameters without touching the test set, and laying the groundwork for robust evaluation, pipelines, and model selection.

---

## 📋 Table of Contents

| Day | Topic | Notebook | Status |
|:---:|-------|----------|:------:|
| 1 | Building a Three-Way Split (Train / Validation / Test) | [`Three_Way_Split.ipynb`](./Day1/Three_Way_Split.ipynb) | ✅ |
| 2 | Cross-Validating a Model (5-Fold Cross-Validation) | [`Cross_Validation.ipynb`](./Day2/Cross_Validation.ipynb) | ✅ |
| 3 | Diagnosing & Fixing Model Fit (Bias–Variance Tradeoff) | [`Diagnosing_Fixing_Model_Fit.ipynb`](./Day3/Diagnosing_Fixing_Model_Fit.ipynb) | ✅ |
| 4 | Feature Engineering & Hyperparameter Tuning | [`Feature_Engineering_and_Tuning.ipynb`](./Day4/Feature_Engineering_and_Tuning.ipynb) | ✅ |
| 5 | Tuned End-to-End Leakage-Free ML Pipeline (Mini-Project) | [`Tuned_End_to_End_Pipeline.ipynb`](./Day5-mini-project/Tuned_End_to_End_Pipeline.ipynb) | ✅ |

---

## 📖 Summary by Day

### [Day 1 — Building a Three-Way Split (Train / Validation / Test)](./Day1/README.md)
Built a 60/20/20 Train / Validation / Test split on a synthetic dataset (1,000 samples) using two chained, stratified `train_test_split` calls. Trained a `DecisionTreeClassifier` baseline on the Training Set and tuned `max_depth` — selecting **`max_depth = 15` (Validation Accuracy 0.8350)** — using ONLY the Validation Set. Retrained the final model on Train + Validation and evaluated it exactly once on the untouched Test Set (**Final Test Accuracy 0.7950**, Precision/Recall ~0.80, F1 ~0.79). Documented why tuning directly on the Test Set causes Data Leakage, optimization bias, and false optimism.

### [Day 2 — Cross-Validating a Model (5-Fold Cross-Validation)](./Day2/README.md)
Evaluated a `LogisticRegression` classifier with **leak-free 5-Fold Stratified Cross-Validation** on a synthetic customer-churn dataset (1,000 samples, 80/20 class split) via a manual fold loop — fitting `StandardScaler()` on each training fold only, then `LogisticRegression` with the weighted F1 metric — **Mean F1 0.8502 ± 0.0176**. Compared the CV estimate against a single stratified 80/20 split (**Single-Split F1 0.8227**, difference −0.0275) and explained why the single split was slightly pessimistic. Verified that `scikit-learn` auto-applies `StratifiedKFold` for classification and documented why stratification preserves the exact 80/20 class ratio in every fold, eliminating evaluation bias and preventing training failures.

### [Day 3 — Diagnosing & Fixing Model Fit (Bias–Variance Tradeoff)](./Day3/README.md)
Deliberately constructed three `DecisionTreeClassifier` states on a noisy synthetic dataset (1,000 samples, 20 features, `flip_y=0.15`, stratified 70/30 split) to make the **Bias–Variance Tradeoff** visible: an unconstrained tree (**100.00% train vs 69.33% val — gap 30.67%**) = overfitting (high variance); a Decision Stump with `max_depth=1` (**60.14% train vs 60.33% val**) = underfitting (high bias); and a regularized/pruned tree (`max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5`) that raised **validation accuracy from 69.33% to 72.33%** while shrinking the **gap from 30.67% to 10.52%**. Documented all three states in a summary comparison table with diagnoses and key takeaways.

### [Day 4 — Feature Engineering & Hyperparameter Tuning](./Day4/README.md)
Engineered **two domain-driven features** on a synthetic telecom-churn dataset (1,000 samples, 6 features, `SEED=42`): `Total_Estimated_Spend` (`Monthly_Charges × Total_Tenure_Months` — Customer Lifetime Value) and `Support_Calls_Per_Tenure_Month` (`Support_Calls / Total_Tenure_Months` — "Frustration Density Index"), expanding the feature space from 6 to 8 and splitting 80/20 stratified (800/200). Benchmarked an untuned `RandomForestClassifier` baseline on raw features (**Mean F1 0.8830 ± 0.0366**) against a **`GridSearchCV`**-tuned Random Forest on engineered data — grid of `n_estimators [50, 100, 200]`, `max_depth [None, 5, 10]`, `min_samples_split [2, 5, 10]`, `max_features ['sqrt', 'log2']` (54 candidates, 270 fits) over 5-Fold `StratifiedKFold` with F1 scoring — best params `{max_depth: 10, max_features: 'log2', min_samples_split: 2, n_estimators: 50}` (**CV F1 0.8941**, **Test F1 0.8636**, **+3.16%** vs baseline 0.8372). Analyzed feature importances (`Total_Estimated_Spend` 11.11% vs 2.82% for the support-call ratio; Age/Monthly_Charges/Tenure strongest overall) and documented why `max_depth=10` + `max_features='log2'` controlled overfitting.

### [Day 5 — Tuned End-to-End Leakage-Free ML Pipeline (Mini-Project)](./Day5-mini-project/README.md)
Built the **Week 4 mini-project**: a complete end-to-end, leakage-free churn-prediction pipeline on a synthetic dataset (1,000 samples, 5 features, stratified 80/20 split → 800/200) that encapsulates everything from Days 1–4. Wrapped custom feature engineering (`Support_Calls_Per_Tenure_Month` "Frustration Density Index" and `High_Value_Customer` flag) in a `FunctionTransformer` **inside** the `Pipeline`, preprocessed mixed types with a `ColumnTransformer` (`StandardScaler` + `OneHotEncoder(handle_unknown='ignore')`), and tuned the **entire pipeline end-to-end** with `GridSearchCV` (grid `classifier__n_estimators [50, 100, 200]`, `classifier__max_depth [3, 5, 10, None]`, `classifier__min_samples_split [2, 5, 10]` — 36 candidates × 5 folds = 180 fits) under 5-Fold `StratifiedKFold` with F1 scoring — **best CV F1 0.9945** at `{max_depth: 5, min_samples_split: 5, n_estimators: 200}`. Evaluated once on the untouched test set: **Test Accuracy 0.975 / Test F1 0.9645** vs. Dummy Baseline (0.650 / 0.0000) and Untuned Default Pipeline (0.965 / 0.9504), with a confusion-matrix heatmap and classification report. Documented the leakage-free guarantee (all preprocessing fitted per-fold inside the pipeline), the value-add over baselines, and how `max_depth=5` prevented overfitting.

---

## 🛠️ Skills & Tools Covered

| Skill | Day | Application |
|-------|:---:|-------------|
| **Three-Way Data Splitting** | 1 | 60/20/20 Train/Validation/Test split with `train_test_split` |
| **Stratified Splitting** | 1 | Preserving class balance across all three sets |
| **Baseline Model Training** | 1 | `DecisionTreeClassifier` on the Training Set |
| **Hyperparameter Tuning** | 1 | Sweeping `max_depth` and evaluating on the Validation Set only |
| **Data Leakage Prevention** | 1 | Test Set kept untouched until the final evaluation |
| **Final Model Evaluation** | 1 | One-time test evaluation (Accuracy, Precision, Recall, F1) |
| **Model Complexity Trade-offs** | 1 | Preferring the smaller `max_depth` on ties |
| **K-Fold Cross-Validation** | 2 | 5-fold manual loop over `StratifiedKFold` (weighted F1) |
| **Stratified K-Fold** | 2 | `StratifiedKFold(n_splits=5, shuffle=True)` for imbalanced classes |
| **Leak-Free Scaling in CV** | 2 | Per-fold `StandardScaler` + `LogisticRegression` (no `Pipeline`) |
| **Mean ± Std Reporting** | 2 | Final estimate 0.8502 ± 0.0176 across 5 folds |
| **CV vs. Single-Split Comparison** | 2 | Single split (0.8227) vs. CV mean (0.8502) |
| **Stratification Justification** | 2 | Preserving 80/20 class proportions in every fold |
| **Bias–Variance Tradeoff** | 3 | Diagnosing high-variance (overfit) vs high-bias (underfit) models |
| **Deliberate Overfitting** | 3 | Unconstrained tree (`max_depth=None`) — 100% train vs 69.33% val |
| **Deliberate Underfitting** | 3 | Decision Stump (`max_depth=1`) — 60.14% train / 60.33% val |
| **Hyperparameter Regularization** | 3 | `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5` |
| **Train-vs-Val Gap Diagnosis** | 3 | Gap shrunk from 30.67% to 10.52% after regularization |
| **Model Fit Documentation** | 3 | Summary comparison table + key takeaways |
| **Feature Engineering** | 4 | `Total_Estimated_Spend` (CLV) + `Support_Calls_Per_Tenure_Month` (Frustration Density) |
| **Feature Justification** | 4 | Mathematical/domain rationale for each engineered feature |
| **Hyperparameter Grid Definition** | 4 | `n_estimators`, `max_depth`, `min_samples_split`, `max_features` |
| **GridSearchCV** | 4 | 54 candidates × 5 folds = 270 fits over `StratifiedKFold` |
| **Tuned vs. Untuned Benchmarking** | 4 | Baseline (0.8830 ± 0.0366) vs. tuned (0.8941 CV, 0.8636 Test) |
| **Feature Importance Analysis** | 4 | Gini importances — engineered vs. raw feature rankings |
| **Hyperparameter Impact Analysis** | 4 | `max_depth` / `max_features` controlling overfitting |
| **Leak-Free Tuning** | 4 | Test set untouched until final evaluation |
| **End-to-End Pipeline** | 5 | `feature_engineering → preprocessing → classifier` in one `Pipeline` |
| **In-Pipeline Feature Engineering** | 5 | `FunctionTransformer` creating features inside the pipeline (leak-free) |
| **ColumnTransformer Preprocessing** | 5 | `StandardScaler` (numeric) + `OneHotEncoder(handle_unknown='ignore')` (categorical) |
| **End-to-End GridSearchCV** | 5 | `classifier__` prefixed grid — 36 candidates × 5 folds = 180 fits |
| **Baseline Comparison** | 5 | Dummy (0.650 / 0.0000) vs. Untuned (0.965 / 0.9504) vs. Tuned (0.975 / 0.9645) |
| **Confusion Matrix Reporting** | 5 | Heatmap + classification report for the tuned pipeline |

---

## 📁 Folder Structure

```
BinX_Week_04/
├── Day1/
│   ├── Three_Way_Split.ipynb
│   └── README.md
├── Day2/
│   ├── Cross_Validation.ipynb
│   └── README.md
├── Day3/
│   ├── Diagnosing_Fixing_Model_Fit.ipynb
│   └── README.md
├── Day4/
│   ├── Feature_Engineering_and_Tuning.ipynb
│   └── README.md
├── Day5-mini-project/
│   ├── Tuned_End_to_End_Pipeline.ipynb
│   └── README.md
└── README.md          ← You are here
```

---

## 🚀 How to Run

1. **Clone the parent repository:**
   ```bash
   git clone --recurse-submodules https://github.com/adam-alAj/BinX-ML-Internship.git
   ```

2. **Navigate to Week 4:**
   ```bash
   cd BinX_ML_Internship/BinX_Week_04
   ```

3. **Activate the virtual environment** (located at the parent root):
   ```bash
   ..\.venv\Scripts\activate        # Windows
   source ../.venv/bin/activate     # Linux / macOS
   ```

4. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

5. **Launch Jupyter:**
   ```bash
   python -m jupyter notebook
   ```

---

## 🔗 Related

- [Root Repository README](../README.md) — Full internship overview and progress tracker
- [Week 3: Machine Learning Workflow Foundations](../BinX_Week_03/README.md) — Previous module with workflow design, linear/logistic regression, and an end-to-end pipeline
