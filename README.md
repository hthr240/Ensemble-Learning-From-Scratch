# AdaBoost & Model Selection | Python Implementation

A dual-module Machine Learning project implementing **Ensemble Methods** (AdaBoost) and **Regularization techniques** (Ridge/Lasso) from scratch. The project demonstrates the impact of noise on boosting algorithms and performs manual hyperparameter tuning using custom Cross-Validation.

## 🚀 Key Features
### 1. AdaBoost (Ensemble Learning)
* **Algorithm:** Full implementation of `AdaBoost.M1` using **Decision Stumps** as weak learners.
* **Noise Analysis:** Simulates datasets with varying noise levels to analyze the trade-off between training error (overfitting) and test error.
* **Visualizations:** Generates dynamic decision surfaces using **Plotly** to visualize how the ensemble boundary evolves with more learners.

### 2. Regularization & Model Selection
* **Custom Cross-Validation:** Implemented a k-fold cross-validation engine from scratch (no `sklearn.model_selection.cross_val_score`).
* **Ridge vs. Lasso:** Comparative analysis of L1 (Lasso) and L2 (Ridge) regularization on the **Diabetes Dataset**.
* **Hyperparameter Tuning:** Automated selection of the optimal regularization parameter ($\lambda$) to minimize generalization error.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** NumPy (Math), Plotly (Interactive Viz), Matplotlib.
* **Concepts:** Boosting, Weak Learners, Bias-Variance Tradeoff, K-Fold CV.

## 📂 Project Structure
* `adaboost.py`: Main AdaBoost class implementing `fit`, `predict`, and weight updates.
* `decision_stump.py`: Efficient implementation of a Decision Stump (threshold-based weak learner).
* `loss_functions.py`: Library of loss metrics (e.g., Misclassification Error) used to evaluate model performance.
* `cross_validate.py`: Custom implementation of K-Fold Cross Validation.
* `estimators.py`: Wrapper classes for Ridge and Lasso regression.
* `adaboost_scenario.py`: Script for running AdaBoost noise experiments.
* `perform_model_selection.py`: Script for Ridge/Lasso parameter tuning.
* `utils.py`: Helper functions for visualization and data plotting.

## 🧠 Algorithmic Implementation
* **Decision Stump Optimization:** The weak learner finds the optimal threshold by sorting feature values and using cumulative sum arrays to calculate loss in $O(N \log N)$ time, rather than $O(N^2)$.
* **Weighted Loss:** The AdaBoost loop iteratively increases weights for misclassified samples ($D_t$) to force the next learner to focus on hard cases.
