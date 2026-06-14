# DM2 Project — Child Mind Institute: Problematic Internet Use

**Course:** Data Mining 2 : Advanced Topics and Applications  | **Academic Year:** 2025/2026

---

## Folder Structure

```
DM2_project/
│
├── data/
│   ├── cmi_internet.csv                  ← tabular dataset (download from Kaggle)
│   ├── data_dictionary.csv               ← feature descriptions
│   └── CMI_timeseries_dataset_pkl.gz     ← accelerometer time series
│
├── notebooks/
│   ├── 00_data_understanding.ipynb       ← Module 0: EDA + preprocessing
│   ├── 01_outlier_imbalance.ipynb        ← Module 1: Outlier detection + imbalanced learning
│   ├── 02_classification_regression_explainability.ipynb  ← Module 2: All classifiers + regression + SHAP
│   └── 03_time_series.ipynb              ← Module 3: TS clustering + classification
│
├── outputs/                              ← auto-created; all CSVs, figures, models saved here
├── DM2_Report.docx
└── README.md
```

---


## Installation

Install all required packages in one step:

```bash
pip install -r requirements.txt
```

Or using conda:
```bash
conda install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost shap imbalanced-learn pyod stumpy tslearn sktime
```

**Python version:** 3.9 or higher recommended.

---

## Notebook Summary

### `00_data_understanding.ipynb` — Module 0
- Load and inspect the raw CMI dataset
- Drop PCIAT leakage columns
- Explore target distribution, missingness, correlations
- Stratified 75/25 train/test split (before any cleaning)
- Build preprocessing pipeline (missingness flags + one-hot encoding + median imputation)
- Save: `X_train_processed.csv`, `X_test_processed.csv`, `y_train.csv`, `y_test.csv`

### `01_outlier_imbalance.ipynb` — Module 1
**Part A — Outlier Detection**
- LOF (n_neighbors=50), ABOD (fast, n_neighbors=50), Isolation Forest (200 estimators)
- Majority-vote ensemble (≥2/3): flags 7.66% of training set
- PCA visualisation + per-class outlier rates
- Saves: `X_train_clean.csv`, `y_train_clean.csv`

**Part B — Imbalanced Learning**
- Binary problem: sii=3 (Severe) vs rest (~1% positive)
- Decision Tree with SMOTE / ENN / SMOTEENN (k sensitivity analysis)
- All AUC ≈ 0.5 → motivates class-weighting strategy in Module 2

### `02_classification_regression_explainability.ipynb` — Module 2
**Part A — Advanced Classification (6 model families)**
1. Logistic Regression (ordinal cumulative binary models)
2. SVM (linear kernel, C=1, class_weight='balanced')
3. Neural Network / MLP (hidden_layer_sizes=(64,), early stopping)
4. Bagging (50 DT estimators, max_samples=0.7)
5. Random Forest (200 est, max_depth=40, max_features=0.7) ← best model (macro-F1=0.341)
6. AdaBoost (50 est, lr=1.0, base max_depth=3)
7. XGBoost (ordinal, scale_pos_weight per threshold)

**Part B — Advanced Regression**
- Target: CGAS-CGAS_Score (0–100 psychological functioning scale)
- XGBoost Regressor + Random Forest Regressor, both with RandomizedSearchCV
- Both achieve R² ≈ 0.03 (regression-to-mean, expected for this type of target)

**Part C — Explainability (SHAP)**
- TreeExplainer on XGBoost P(sii≥1) model
- Top predictors: internet hours/day, CGAS, sleep disturbance (SDS), height, age

### `03_time_series.ipynb` — Module 3
- Load 4,437 accelerometer series → drop 179 non-wear → 4,258 series from 326 children
- Focus signal: **enmo** (activity intensity, univariate)
- Train/test split by child ID (no leakage)
- **Motifs/Discords**: Matrix Profile (STUMP, window=10 steps = 5 hours)
- **Clustering**: SAX+KMeans (cluster 1: 51.2% problematic) + Ward on normalised profiles (cluster 2: 53.8%)
- **Classification**: k-NN Euclidean (F1=0.507), k-NN DTW (F1=0.505), ROCKET (F1=0.504), Shapelets (F1=0.522, best)

---

## Key Findings

| Module | Finding |
|--------|---------|
| M0 | 69.7% / 18.4% / 10.8% / 1.0% class split; 107 features after preprocessing |
| M1 | 7.66% outliers (ABOD-dominated); resampling cannot fix sii=3 scarcity |
| M2 | Random Forest best (macro-F1=0.341); SHAP highlights internet hours, CGAS, sleep |
| M2 Reg | R²≈0.03 for CGAS; regression-to-mean expected with this feature set |
| M3 | All TS classifiers macro-F1≈0.50–0.52, above baseline ≈0.40; Shapelets best |
