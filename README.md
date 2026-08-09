# Week 4: Evaluation, Tuning & Pipelines

**Welcome to Week 4 of the BinX Tech AI & Machine Learning Internship Program.**
This week moves from building models to evaluating them properly: partitioning data into training, validation, and test sets, tuning hyperparameters without touching the test set, and laying the groundwork for robust evaluation, pipelines, and model selection.

---

## 📋 Table of Contents

| Day | Topic | Notebook | Status |
|:---:|-------|----------|:------:|
| 1 | Building a Three-Way Split (Train / Validation / Test) | [`Three_Way_Split.ipynb`](./Day1/Three_Way_Split.ipynb) | ✅ |

---

## 📖 Summary by Day

### [Day 1 — Building a Three-Way Split (Train / Validation / Test)](./Day1/README.md)
Built a 60/20/20 Train / Validation / Test split on a synthetic dataset (1,000 samples) using two chained, stratified `train_test_split` calls. Trained a `DecisionTreeClassifier` baseline on the Training Set and tuned `max_depth` — selecting **`max_depth = 15` (Validation Accuracy 0.8350)** — using ONLY the Validation Set. Retrained the final model on Train + Validation and evaluated it exactly once on the untouched Test Set (**Final Test Accuracy 0.7950**, Precision/Recall ~0.80, F1 ~0.79). Documented why tuning directly on the Test Set causes Data Leakage, optimization bias, and false optimism.

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

---

## 📁 Folder Structure

```
BinX_Week_04/
├── Day1/
│   ├── Three_Way_Split.ipynb
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
