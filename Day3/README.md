# Day 3 — Diagnosing & Fixing Model Fit (Bias–Variance Tradeoff)

Welcome to Day 3 of Week 4. This session explores the core trade-off in machine learning: **Bias vs. Variance**. Instead of just reading about it, we deliberately construct three model states — overfitting, underfitting, and a regularized (balanced) model — and use the **Train-vs-Validation performance gap** as the diagnostic tool to identify and fix model fit problems.

---

## 🎯 Objective

Diagnose and fix model fit by:

- deliberately **overfitting** a model (high variance) — a tree that memorizes the training set but fails to generalize
- deliberately **underfitting** a model (high bias) — an overly simplistic model that misses underlying patterns
- applying **regularization / pruning** to reduce variance and shrink the train-validation gap
- documenting the diagnosis with a score comparison table and key takeaways

---

## 📓 Notebook

- [Diagnosing_Fixing_Model_Fit.ipynb](./Diagnosing_Fixing_Model_Fit.ipynb)

---

## ✅ Key Tasks & Accomplishments

- Generated a synthetic classification dataset (1,000 samples, 20 features, 8 informative, 4 redundant, `n_clusters_per_class=2`) with `make_classification`, adding noise via `flip_y=0.15` — the noisy labels are what make overfitting visible. Used a fixed `SEED = 42` and a stratified **70/30 Train/Validation split** (700 train / 300 validation samples).
- **Step 1 — Deliberately Overfit (High Variance):** Trained an unconstrained `DecisionTreeClassifier(max_depth=None, min_samples_split=2)`. Because the tree can grow without depth limits, it memorized noise in the training set: **Training Accuracy 100.00% vs. Validation Accuracy 69.33% → gap 30.67%** — textbook evidence of overfitting.
- **Step 2 — Deliberately Underfit (High Bias):** Trained a Decision Stump, `DecisionTreeClassifier(max_depth=1)`. With capacity limited to a single split, the model failed to capture feature patterns: **Training Accuracy 60.14% vs. Validation Accuracy 60.33% → gap −0.19%** — low scores on *both* sets with a negligible gap is the classic signature of underfitting.
- **Step 3 — Apply Regularization & Pruning to Fix Overfitting:** Retrained with complexity controls — `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5` — preventing the tree from learning noisy outliers and forcing it to learn generalized rules: **Training Accuracy 82.86% vs. Validation Accuracy 72.33% → gap 10.52%**.
- **Step 4 — Diagnosis Documentation & Score Evidence:** Built a **Summary Comparison Table** of all three model states (configuration, train accuracy, validation accuracy, train-val gap, diagnosis):
  | Model State | Configuration / Parameters | Train Acc | Val Acc | Gap | Diagnosis |
  | :--- | :--- | :---: | :---: | :---: | :--- |
  | **Overfit (High Variance)** | `max_depth=None` (Unconstrained) | 100.00% | 69.33% | 30.67% | Large gap = memorization of noise |
  | **Underfit (High Bias)** | `max_depth=1` (Decision Stump) | 60.14% | 60.33% | −0.19% | Both scores low = cannot learn features |
  | **Balanced (Regularized)** | `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=5` | 82.86% | 72.33% | 10.52% | Gap shrunk, validation improved |
- **The Result:** Restricting tree depth and requiring minimum samples per leaf constrained the model's capacity — validation accuracy rose from **69.33% to 72.33%** while the train-validation gap contracted from **30.67% to 10.52%**, demonstrating effective variance reduction and better generalization. Visualized the three model states with Matplotlib for side-by-side comparison.

---

## 🛠️ Skills Covered

- Bias–Variance Tradeoff — recognizing high-bias (underfit) and high-variance (overfit) behavior
- Deliberate overfitting with an unconstrained `DecisionTreeClassifier`
- Deliberate underfitting with a Decision Stump (`max_depth=1`)
- Hyperparameter regularization & pruning (`max_depth`, `min_samples_split`, `min_samples_leaf`)
- Using the **Train-vs-Validation gap** as a model-fit diagnostic
- Reading & comparing accuracy scores across model states
- Documenting diagnoses with comparison tables and actionable takeaways

---

## 🔗 Related

- [Week 4 Overview](../README.md)
- [Root Repository README](../../README.md)
