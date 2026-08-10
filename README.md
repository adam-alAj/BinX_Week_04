# Week 4: Evaluation, Tuning & Pipelines

**Welcome to Week 4 of the BinX Tech AI & Machine Learning Internship Program.**
This week moves from building models to evaluating them properly: partitioning data into training, validation, and test sets, tuning hyperparameters without touching the test set, and laying the groundwork for robust evaluation, pipelines, and model selection.

---

## 📋 Table of Contents

| Day | Topic | Notebook | Status |
|:---:|-------|----------|:------:|
| 1 | Building a Three-Way Split (Train / Validation / Test) | [`Three_Way_Split.ipynb`](./Day1/Three_Way_Split.ipynb) | ✅ |
| 2 | Cross-Validating a Model (5-Fold Cross-Validation) | [`Cross_Validation.ipynb`](./Day2/Cross_Validation.ipynb) | ✅ |

---

## 📖 Summary by Day

### [Day 1 — Building a Three-Way Split (Train / Validation / Test)](./Day1/README.md)
Built a 60/20/20 Train / Validation / Test split on a synthetic dataset (1,000 samples) using two chained, stratified `train_test_split` calls. Trained a `DecisionTreeClassifier` baseline on the Training Set and tuned `max_depth` — selecting **`max_depth = 15` (Validation Accuracy 0.8350)** — using ONLY the Validation Set. Retrained the final model on Train + Validation and evaluated it exactly once on the untouched Test Set (**Final Test Accuracy 0.7950**, Precision/Recall ~0.80, F1 ~0.79). Documented why tuning directly on the Test Set causes Data Leakage, optimization bias, and false optimism.

### [Day 2 — Cross-Validating a Model (5-Fold Cross-Validation)](./Day2/README.md)
Evaluated a `LogisticRegression` classifier with **leak-free 5-Fold Stratified Cross-Validation** on a synthetic customer-churn dataset (1,000 samples, 80/20 class split) using `cross_val_score` with the `f1_weighted` metric — **Mean F1 0.8502 ± 0.0176**. Compared the CV estimate against a single stratified 80/20 split (**Single-Split F1 0.8227**, difference −0.0275) and explained why the single split was slightly pessimistic. Verified that `scikit-learn` auto-applies `StratifiedKFold` for classification and documented why stratification preserves the exact 80/20 class ratio in every fold, eliminating evaluation bias and preventing training failures.

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
| **K-Fold Cross-Validation** | 2 | 5-Fold `cross_val_score` with the `f1_weighted` metric |
| **Stratified K-Fold** | 2 | `StratifiedKFold(n_splits=5, shuffle=True)` for imbalanced classes |
| **Leak-Free Pipelines in CV** | 2 | `make_pipeline(StandardScaler, LogisticRegression)` |
| **Mean ± Std Reporting** | 2 | Final estimate 0.8502 ± 0.0176 across 5 folds |
| **CV vs. Single-Split Comparison** | 2 | Single split (0.8227) vs. CV mean (0.8502) |
| **Stratification Justification** | 2 | Preserving 80/20 class proportions in every fold |

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
