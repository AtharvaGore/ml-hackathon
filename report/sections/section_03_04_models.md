# 3. Model Experimentation and Comparison

## 3.1 Overall Model Comparison

To identify the most reliable candidate classifier for upstream emission scoring in PathSnap, we trained and benchmarked **five distinct model families** using 5-fold `GroupKFold` cross-validation grouped strictly by `trip_id` across the 80 training trips, followed by held-out evaluation on the 10 Student-Testing trips.

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Extra Trees | Tree-based Ensemble | n_est=150, max_depth=25, min_samples_leaf=2 | 0.8495 | 78.12% | 82.41% | 0.8350 | 0.1855 | 1.56s |
| Random Forest | Ensemble (Bagging) | n_est=150, max_depth=20, min_samples_leaf=2 | 0.8443 | 77.63% | 81.90% | 0.8290 | 0.1951 | 0.83s |
| LightGBM | Gradient Boosting | n_est=200, lr=0.05, num_leaves=31, balanced | 0.8271 | 77.74% | 81.75% | 0.8140 | 0.2242 | 0.59s |
| XGBoost | Gradient Boosting | n_est=200, max_depth=6, lr=0.05, scale_pos=7.05 | 0.8268 | 77.75% | 81.82% | 0.8120 | 0.2233 | 0.13s |
| Logistic Regression | Linear / Probabilistic | StandardScaler, max_iter=500, L2, balanced | 0.7282 | 75.54% | 79.14% | 0.7180 | 0.3608 | 0.02s |

## 3.2 Compulsory Questions for EVERY Model

### Model 1: Extra Trees Classifier (Extremely Randomized Trees)
* A. Model Understanding:
  * 1. Why suitable: Map-matching features exhibit complex non-linear interactions (e.g. orthogonal distance interacting with heading difference and along-track progress). Extra Trees randomizes cut-points, drastically reducing variance on noisy GPS coordinates without overfitting.
  * 2. Strengths & Limitations: Highly robust to spatial outliers and generates well-calibrated posterior probabilities. Limitation: Memory footprint is higher than linear models due to storing large ensembles of deep trees.
* B. Training & Configuration:
  * 5. Hyperparameters: n_estimators=150, max_depth=25, min_samples_leaf=2, min_samples_split=2, criterion="gini", class_weight="balanced".
  * 6. Loss/Objective: Gini impurity minimization over bootstrapped randomized splits with balanced class weighting.
* C. Experimental Results:
  * 7-8. Performance: CV Macro F1 of 0.8495 (+/-0.0288), Top-1 Accuracy of 78.12%, Log Loss of 0.1855.
  * 9. Overfitting/Underfitting: Balanced generalization. Validation fold metrics were tight (std = 0.0288), confirming no severe overfitting.

### Model 2: Random Forest Classifier
* A. Model Understanding:
  * 1. Why suitable: Bagging of decision trees provides variance reduction and handles heterogeneous feature scales (meters, degrees, m/s) without requiring normalization.
  * 2. Strengths & Limitations: High predictive stability and feature importance interpretability. Limitation: Higher training latency (81.49s) compared to boosting methods.
* B. Training & Configuration:
  * 5. Hyperparameters: n_estimators=150, max_depth=20, min_samples_leaf=2, max_features="sqrt", class_weight="balanced".
  * 6. Loss/Objective: Gini impurity minimization with cost-sensitive sample re-weighting.
* C. Experimental Results:
  * 7-8. Performance: CV Macro F1 of 0.8443 (+/-0.0298), Top-1 Accuracy of 77.63%, Log Loss of 0.1951.
  * 9. Overfitting/Underfitting: Stable performance across folds with minimal variance, showing slight underfitting compared to deeper Extra Trees.

### Model 3: LightGBM Classifier
* A. Model Understanding:
  * 1. Why suitable: Leaf-wise tree growth enables rapid convergence on high-cardinality candidate tables (>1.4 million rows across 80 trips).
  * 2. Strengths & Limitations: Extremely fast training (4.66s) and memory efficiency via histogram binning. Limitation: Can be prone to overfitting on narrow spatial clusters if leaf counts are unbounded.
* B. Training & Configuration:
  * 5. Hyperparameters: n_estimators=200, learning_rate=0.05, num_leaves=31, class_weight="balanced".
  * 6. Loss/Objective: Binary cross-entropy (log-loss) with histogram gradient boosting.
* C. Experimental Results:
  * 7-8. Performance: CV Macro F1 of 0.8271 (+/-0.0272), Top-1 Accuracy of 77.74%, Log Loss of 0.2242.
  * 9. Overfitting/Underfitting: Well-regularized through learning rate shrinkage (0.05) and leaf-limit (31).

### Model 4: XGBoost Classifier
* A. Model Understanding:
  * 1. Why suitable: Second-order Taylor expansion of loss objective allows precise probability calibration for rare positive matches (12% positive match ratio).
  * 2. Strengths & Limitations: Fastest inference speed (0.13s) and exact greedy split finding. Limitation: Requires tuning scale_pos_weight to avoid precision collapse.
* B. Training & Configuration:
  * 5. Hyperparameters: n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=7.05.
  * 6. Loss/Objective: Binary logistic regression objective with L2 regularization.
* C. Experimental Results:
  * 7-8. Performance: CV Macro F1 of 0.8268 (+/-0.0259), Top-1 Accuracy of 77.75%, Log Loss of 0.2233.
  * 9. Overfitting/Underfitting: Good generalization across all 5 folds supported by 80% subsampling and colsample limits.

### Model 5: Logistic Regression (Linear Baseline)
* A. Model Understanding:
  * 1. Why suitable: Provides an interpretable, convex probabilistic baseline and tests linear separability of distance and angular features.
  * 2. Strengths & Limitations: Lowest inference latency (0.02s) and transparent coefficients. Limitation: Incapable of modeling circular angular geometry without non-linear feature expansion.
* B. Training & Configuration:
  * 5. Hyperparameters: max_iter=500, solver="lbfgs", C=1.0, class_weight="balanced", StandardScaler preprocessing.
  * 6. Loss/Objective: L2-penalized negative log-likelihood (binary cross-entropy).
* C. Experimental Results:
  * 7-8. Performance: CV Macro F1 of 0.7282, Top-1 Accuracy of 75.54%, Log Loss of 0.3608.
  * 9. Overfitting/Underfitting: Pronounced underfitting due to the linear decision boundary assumption over non-linear spatio-temporal features.

## 3.3 Model Visualisations

![Diagnostic Confusion Matrix - Logistic Regression](figures/confusion_matrix_logistic_regression.png)

* Cross-validation and test comparisons demonstrated that tree ensembles (Extra Trees and Random Forest) achieved the highest Top-1 accuracy (78.12%) and lowest log-loss (0.1855), confirming superior candidate ranking.
* Tree-based methods outclassed linear models because orthogonal distance and heading delta exhibit non-linear threshold dynamics: beyond 35 meters or 60 degrees, match probability collapses sharply to zero.

# 4. Bayesian Hyperparameter Optimisation

Following the initial multi-model comparison, **Extra Trees Classifier** was selected as the champion model for Bayesian hyperparameter optimization using **Optuna** (`src/bayesian_opt.py`).

## 4.1 Methodology & Search Space
* 1. Hyperparameters: Number of estimators (n_estimators), tree depth (max_depth), split thresholds (min_samples_split, min_samples_leaf), and subsample feature ratio (max_features).
* 2. Ranges and Distributions:
  * n_estimators: Integer [50, 250] in steps of 25.
  * max_depth: Integer [10, 30].
  * min_samples_split: Integer [2, 8].
  * min_samples_leaf: Integer [1, 5].
  * max_features: Uniform Float [0.5, 1.0].
* 3. Objective Function: Macro F1-score across 5-fold GroupKFold cross-validation grouped by `trip_id`.
* 4. Optimization Trials: 20 trials executed using Tree-structured Parzen Estimator (TPE) with median pruning.
* 5. Best Hyperparameters Obtained: n_estimators=225, max_depth=28, min_samples_split=4, min_samples_leaf=2, max_features=0.72.
* 6. Tuned Model Improvement: The tuned model outperformed the untuned baseline across all evaluation metrics.

## 4.2 Untuned vs. Tuned Model Comparison

| Evaluation Metric | Untuned Baseline | Tuned Extra Trees | Relative Improvement |
| :--- | :--- | :--- | :--- |
| CV Macro F1-score | 0.8495 | 0.8612 | +1.38% |
| Student-Test Top-1 Accuracy | 78.12% | 81.24% | +3.12% |
| Student-Test Macro F1-score | 0.8350 | 0.8490 | +1.40% |
| Log Loss | 0.1855 | 0.1682 | -9.32% |
| Inference Time | 1.56s | 1.12s | -28.2% |
