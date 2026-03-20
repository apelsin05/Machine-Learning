# Machine Learning Experiments

Hands-on machine learning notebooks developed during the **Data Mining & ML Algorithms** course (Erasmus+ exchange semester, 2024–2025) at "Dunărea de Jos" University of Galați, Faculty FACIEE.

Each notebook is a self-contained case study combining theory, implementation, and experimental analysis on real-world datasets.

---

## Notebooks

### kNN Comparative Study — Efficient Nearest Neighbour Search
`kNN_comparative_study/`

**Dataset:** Diabetes, Hypertension and Stroke Prediction (Prosper Chucks) — 70,692 rows, 18 features including demographic, lifestyle, and health indicators. Target variable: Diabetes (binary).

**What this covers:**
- Comparative performance analysis between **standard kNN (brute-force)** and **KD-Tree optimised kNN**
- KD-Tree organises data into a hierarchical spatial structure, drastically reducing the number of comparisons at inference time
- Benchmarking execution time and prediction accuracy across both implementations on the same dataset
- Demonstrates that KD-Tree indexing significantly accelerates search while maintaining accuracy — relevant for real-time applications like recommendation systems and NLP

**Key concepts:** k-Nearest Neighbours, KD-Tree, nearest neighbour search, classification, bias-variance tradeoff, scikit-learn

---

### Ridge vs Lasso Regression — Regularisation Comparative Study
`RidgeVsLassoRegression/`

**Dataset:** Ames Housing dataset — rich mix of numerical and categorical features describing residential properties.

**What this covers:**
- Side-by-side comparison of **Ridge (L2)** and **Lasso (L1)** regularisation for combating overfitting in linear regression
- Full preprocessing pipeline: missing value imputation, numerical standardisation, one-hot encoding for categorical variables
- Optimal λ (regularisation strength) selection via **k-fold cross-validation** on the training set
- Model evaluation on a held-out test set using **R²** and **RMSE**
- Key finding: Lasso achieves the best test error and performs automatic feature selection; Ridge is more stable when predictors are correlated

**Key concepts:** Linear regression, OLS, Ridge L2, Lasso L1, overfitting, regularisation, k-fold CV, feature selection, scikit-learn

---

### Association Rule Mining — FAO Food Consumption Dataset
`association_food/`

**Dataset:** FAO (Food and Agriculture Organization of the United Nations) food consumption data for **Romania** and **Nigeria** — over 10,000 food transactions, standardised using the FoodEx2 classification system.

**What this covers:**
- Comparison of **ECLAT** (vertical format) vs **Apriori** (classic) algorithms for association rule mining
- Full preprocessing pipeline: noise removal, description standardisation via FoodEx2
- Evaluation using **Support**, **Confidence**, and **Lift** metrics to filter statistically significant rules
- Key finding: ECLAT was ~30x faster than Apriori on dense datasets
- Qualitative insight: Romanian diet patterns centre on raw cooking ingredients (oil, onion, carrot); Nigerian patterns show structured meal compositions (starch + soup)

**Key concepts:** Association rule mining, Apriori, ECLAT, support/confidence/lift, market basket analysis, mlxtend, pandas, cross-cultural data analysis

---

## Stack

```
Python · Jupyter Notebook · scikit-learn · pandas · numpy · matplotlib · mlxtend
```

## Context

These notebooks were produced as graded coursework during the last year of University in the Computer Science Bachelor's, alongside a parallel theoretical study of linear regression formulas and regularisation mathematics. Each notebook has an accompanying written report with full methodology, results, and analysis in Romanian.
