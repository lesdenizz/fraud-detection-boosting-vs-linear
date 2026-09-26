# Fraud Detection — Model Comparison & Selection

An end-to-end fraud detection pipeline that trains, tunes, and compares 7 classification algorithms (Logistic Regression, KNN, Random Forest, XGBoost, LightGBM, and CatBoost in two configurations) on a transaction dataset, then narrows down to a single production-ready model using feature importance and SHAP.

## Overview

Financial fraud is a classic **rare-event, imbalanced classification** problem — in this dataset, fraudulent transactions make up only **~1.2%** of all records. The goal of this project is to build a model that reliably separates fraudulent from legitimate transactions while being honest about the overfitting risk that comes with such a skewed target.

The notebook walks through the full workflow a data scientist would follow in practice:
1. Clean and prepare the data
2. Build model-specific preprocessing pipelines (WoE for linear models, scaling/encoding for distance-based models, minimal encoding for tree ensembles)
3. Train and benchmark 7 models
4. Tune the top candidates with Optuna (Bayesian hyperparameter search)
5. Select the most *consistent* model — not just the highest-scoring one — using the Train/Test Gini gap as an overfitting signal
6. Explain the winning model with feature importance + SHAP, and trim any features with no real signal
7. Rebuild and re-tune on the final feature set

## Dataset

`fraud_data.xlsx` — 10,125 transactions with the following raw columns:

| Column | Description |
|---|---|
| `type` | Transaction type (`PAYMENT`, `CASH_OUT`, `CASH_IN`, `TRANSFER`, `DEBIT`) |
| `branch` | Country/branch where the transaction originated |
| `amount` | Transaction amount |
| `nameOrig` / `nameDest` | Sender / receiver account IDs |
| `oldbalanceOrg` / `newbalanceOrig` | Sender's balance before / after the transaction |
| `unusuallogin` | Count of unusual login signals associated with the transaction |
| `isFlaggedFraud` | A pre-existing system rule flag |
| `Acct type` | `Savings` or `Current` |
| `Date of transaction` / `Time of day` | When the transaction occurred |
| `isFraud` | **Target** — 1 if the transaction is fraudulent (1.16% of rows) |

### Columns dropped before modeling

Five columns were dropped as part of preprocessing, each for a specific, defensible reason:

| Column | Why it was dropped |
|---|---|
| `nameOrig` | 10,119 / 10,125 unique values — behaves like a row ID, no generalizable signal |
| `nameDest` | 6,494 / 10,125 unique values — same issue |
| `isFlaggedFraud` | Constant (single value across all 10,125 rows) — zero variance, no information |
| `Date of transaction` | Raw date string, largely redundant with `Time of day` |
| `branch` | 135 distinct categories — too sparse/high-cardinality to encode reliably |

## Preprocessing pipelines (per model family)

Different model families need different treatment, so four parallel pipelines were built from the same cleaned data:

| Pipeline | Steps |
|---|---|
| **Logistic Regression** | Weight-of-Evidence (WoE) transformation on every feature → drop WoE features with >70% pairwise correlation → keep only `_woe` columns |
| **KNN** | IQR outlier capping → 70% correlation filter → VIF pruning (target VIF ≤ 5) → one-hot encode categoricals → standard-scale everything |
| **CatBoost (categorical-native)** | Missing-value imputation only — categorical columns are left untouched and passed via `cat_features` |
| **XGBoost / LightGBM / CatBoost / Random Forest** | Label encoding of categorical columns |

All transformations (WoE bins, correlation drops, VIF, scaler, encoders) are **fit on the training split only** and applied to test — no leakage.

## Modeling & Hyperparameter Tuning

7 models were trained with default parameters first, then 5 of them (KNN, Random Forest, XGBoost, LightGBM, CatBoost) were re-tuned with **Optuna** (Tree-structured Parzen Estimator, seeded for reproducibility) using 3-fold cross-validated ROC-AUC as the objective.

### Full results (sorted by Test Gini, then by Train–Test Gini gap)

| Model | Train Gini | Test Gini | Gini Gap |
|---|---|---|---|
| RandomForest | 1.000 | 0.659 | 0.341 |
| RandomForest Optuna | 0.994 | 0.634 | 0.360 |
| LightGBM | 1.000 | 0.603 | 0.397 |
| CatBoost_Custom | 0.963 | 0.600 | 0.363 |
| LightGBM Optuna | 0.999 | 0.591 | 0.408 |
| XGBoost Optuna | 0.960 | 0.573 | 0.387 |
| CatBoost | 0.991 | 0.552 | 0.439 |
| XGBoost | 1.000 | 0.493 | 0.507 |
| **CatBoost Optuna** | **0.713** | **0.473** | **0.239** ✅ |
| LogisticRegression (WoE) | 0.614 | 0.433 | 0.181 |
| KNN Optuna | 1.000 | 0.284 | 0.716 |
| KNN | 0.971 | 0.141 | 0.830 |

### Key takeaways

- Nearly every tree ensemble hits a **Train Gini near 1.0** while Test Gini sits far lower — classic overfitting on a small, heavily imbalanced dataset.
- **KNN is the most overfit model of all**, despite tuning — a symptom of distance-based learning on a high-dimensional one-hot-encoded feature space.
- **Logistic Regression on WoE features** has the smallest gap among non-tree models and a respectable Test Gini — the most *stable* model overall, even though it doesn't have the top raw score.
- Among tree-based models, **CatBoost tuned by Optuna has by far the smallest Train–Test Gini gap** — Optuna's search settled on a much shallower, more regularized configuration than the untuned defaults, trading some raw Test Gini for meaningfully better generalization. That's why it was selected over Random Forest (which has the single highest raw Test Gini but a much larger overfitting gap).

## Feature Selection: Importance + SHAP

The winning model (**CatBoost Optuna**) was explained using both native feature importance and SHAP values:

1. Kept features with **native importance > 1%**
2. Cross-checked with mean absolute SHAP value to confirm each feature has real, non-negligible impact on predictions
3. Final feature set: `newbalanceOrig`, `oldbalanceOrg`, `type`, `unusuallogin`, `amount`, `Acct type` (`Time of day` was dropped — negligible importance and SHAP contribution)

The model was then **re-tuned with Optuna on this reduced feature set** to produce the final production candidate.

## Final Model

| Metric | Value |
|---|---|
| Model | CatBoost (Optuna-tuned, final feature set) |
| Train Gini | 81.67% |
| Test Gini | 52.62% |
| Test ROC-AUC | 0.763 |
| Test PR-AUC | 0.439 |

### Visuals

> Save the PNGs the notebook generates (`roc_curve_final_model.png`, `pr_curve_final_model.png`, `confusion_matrix_final_model.png`, `model_comparison_test_gini.png`) into an `assets/` folder in this repo, then these will render below.

**ROC Curve**

![ROC Curve](assets/roc_curve_final_model.png)

**Precision-Recall Curve** — more informative than ROC given the ~1.2% fraud rate

![Precision-Recall Curve](assets/pr_curve_final_model.png)

**Confusion Matrix (Test set)**

![Confusion Matrix](assets/confusion_matrix_final_model.png)

**Model Comparison — Test Gini (winning model highlighted)**

![Model Comparison](assets/model_comparison_test_gini.png)

## Tech Stack

- **Python**, **pandas**, **numpy**
- **scikit-learn** — preprocessing, Logistic Regression, KNN, Random Forest, metrics
- **XGBoost**, **LightGBM**, **CatBoost** — gradient boosting models
- **Optuna** — Bayesian hyperparameter optimization
- **SHAP** — model explainability
- **statsmodels** — VIF calculation
- **matplotlib**, **seaborn** — visualization

## How to Run

1. Clone this repo and place `fraud_data.xlsx` in the project root (or update the path in the notebook's first data cell).
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna shap statsmodels matplotlib seaborn openpyxl
   ```
3. Open `Fraud_Model_all_models_completed.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

## Project Structure

```
.
├── fraud_data.xlsx
├── Fraud_Model_all_models_completed.ipynb
├── assets/
│   ├── roc_curve_final_model.png
│   ├── pr_curve_final_model.png
│   ├── confusion_matrix_final_model.png
│   └── model_comparison_test_gini.png
└── README.md
```

## Notes & Limitations

- The dataset is small (10,125 rows) with very few positive (fraud) cases, which limits how much the Train–Test gap can be closed — more data or techniques like SMOTE/class-weighting would be natural next steps.
- Hyperparameter search used a modest number of Optuna trials per model for runtime reasons; increasing `n_trials` may find better configurations.
- Model selection prioritized **generalization (smallest Gini gap)** over raw Test Gini — a deliberate choice appropriate for a fraud system where an overfit model performing well only in backtesting is a real business risk.
