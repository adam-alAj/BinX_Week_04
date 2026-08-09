# Day 1 — Building a Three-Way Split (Train / Validation / Test)

Welcome to Day 1 of Week 4. This session introduces the three-way split — Training, Validation, and Test sets — and demonstrates why tuning hyperparameters on a reserved validation set (never the test set) leads to honest, unbiased model evaluation.

---

## 🎯 Objective

Build a robust evaluation workflow that covers:

- performing a 60% Train / 20% Validation / 20% Test split with a fixed `random_state`
- training a baseline model on the Training Set only
- tuning a hyperparameter (`max_depth`) using ONLY the Validation Set
- performing a final, honest evaluation on the untouched Test Set exactly once
- documenting why tuning directly on the Test Set causes Data Leakage

---

## 📓 Notebook

- [Three_Way_Split.ipynb](./Three_Way_Split.ipynb)

---

## ✅ Key Tasks & Accomplishments

- Generated a synthetic classification dataset (1,000 samples, 10 features, 8 informative, 2 redundant) with `make_classification` and a fixed `SEED = 42`.
- **Step 1 — Three-Way Split:** Built the 60/20/20 split with two chained `train_test_split` calls — first Train+Val (80%) vs Test (20%), then Train (60%) vs Validation (20%) — all stratified: 600 train, 200 validation, 200 test samples.
- **Step 2 — Baseline & Hyperparameter Tuning:** Trained a `DecisionTreeClassifier` on the Training Set and swept `max_depth` in `[1, 2, 3, 5, 7, 10, 15, None]`, evaluating each configuration **only on the Validation Set**. Selected **`max_depth = 15` (Validation Accuracy 0.8350)**; ties were broken toward the smaller depth to reduce model complexity.
- **Step 3 — Final Evaluation on the Unseen Test Set:** Retrained the final model on Train + Validation and evaluated it **exactly once** on the untouched Test Set — **Final Test Accuracy 0.7950** (Precision ~0.80, Recall ~0.80, F1-score ~0.79).
- **Step 4 — Why Tuning on the Test Set is Dangerous:** Documented the consequences:
  - **Optimization Bias & Test Contamination** — the Test Set loses its independence, and the hyperparameter selection implicitly learns the test set's random noise.
  - **False Optimism** — test scores become misleadingly optimistic and cannot be replicated on real-world production data.
  - **The Role of Each Split** — Train (60%) learns model weights; Validation (20%) selects hyperparameters; Test (20%) stays in the "vault" and is used only once to estimate true performance.

---

## 🛠️ Skills Covered

- Three-way 60/20/20 data splitting with `train_test_split`
- Stratified splitting to preserve class balance
- Baseline `DecisionTreeClassifier` training
- Hyperparameter tuning on a validation set only
- Final one-time model evaluation (Accuracy, Precision, Recall, F1)
- Data leakage prevention and honest generalization assessment
- Model complexity trade-off analysis

---

## 🔗 Related

- [Week 4 Overview](../README.md)
- [Root Repository README](../../README.md)
