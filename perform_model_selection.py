import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from cross_validate import cross_validate
from estimators import RidgeRegression, Lasso, LinearRegression
from sklearn.datasets import load_diabetes

# helpers
def load_diabetes_data(n_samples: int):
    data = load_diabetes()
    X, y = data.data, data.target
    return X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

def cross_validate_model_over_lambdas(model_type: str, lambdas: np.ndarray, X: np.ndarray, y: np.ndarray):

    train_errors, val_errors = [], []

    for lam in lambdas:
        if model_type == "ridge":
            model = RidgeRegression(lam=lam, include_intercept=True)
        elif model_type == "lasso":
            model = Lasso(alpha=lam, include_intercept=True)
        else:
            raise ValueError("model_type must be 'ridge' or 'lasso'")

        train_loss, val_loss = cross_validate(model, X, y, cv=5)
        train_errors.append(train_loss)
        val_errors.append(val_loss)

    return train_errors, val_errors

def plot_cv_errors(lambdas, train_errors, val_errors, title):
    # plot the graphs
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, train_errors, label="Train Error", color="blue")
    plt.plot(lambdas, val_errors, label="Validation Error", color="orange")
    plt.xscale('linear')
    plt.xlabel("λ (Regularization Strength)")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_test_models(train_X, train_y, test_X, test_y, best_ridge_lambda, best_lasso_lambda):
    # Train Ridge
    ridge_model = RidgeRegression(lam=best_ridge_lambda, include_intercept=True)
    ridge_model.fit(train_X, train_y)
    ridge_test_error = ridge_model.loss(test_X, test_y)

    # Train Lasso
    lasso_model = Lasso(alpha=best_lasso_lambda, include_intercept=True)
    lasso_model.fit(train_X, train_y)
    lasso_test_error = lasso_model.loss(test_X, test_y)

    # Train Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(train_X, train_y)
    linear_test_error = linear_model.loss(test_X, test_y)

    # Print errors
    print("\n--- Test Errors ---")
    print(f"Ridge (λ={best_ridge_lambda:.5f}): {ridge_test_error:.2f}")
    print(f"Lasso (λ={best_lasso_lambda:.5f}): {lasso_test_error:.2f}")
    print(f"Linear Regression: {linear_test_error:.2f}")

    # Plot comparison
    models = ["Ridge", "Lasso", "Linear"]
    errors = [ridge_test_error, lasso_test_error, linear_test_error]

    plt.figure(figsize=(8, 5))
    plt.bar(models, errors, color=["blue", "green", "orange"])
    plt.title("Test Error Comparison")
    plt.ylabel("Mean Squared Error")
    plt.grid(axis='y')
    plt.show()

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    # Load diabetes dataset
    train_X, train_y, test_X, test_y = load_diabetes_data(n_samples)

    # Question 2 - Perform Cross Validation for different values of the regularization parameter for Ridge and
    # Lasso regressions
    ridge_lambdas = np.linspace(0.001, 10, n_evaluations)
    lasso_lambdas = np.linspace(0.001, 10, n_evaluations)
    ridge_train, ridge_val = cross_validate_model_over_lambdas("ridge", ridge_lambdas, train_X, train_y)
    lasso_train, lasso_val = cross_validate_model_over_lambdas("lasso", lasso_lambdas, train_X, train_y)
    plot_cv_errors(ridge_lambdas, ridge_train, ridge_val, "Ridge Regression CV")
    plot_cv_errors(lasso_lambdas, lasso_train, lasso_val, "Lasso Regression CV")
    

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lambda = ridge_lambdas[np.argmin(ridge_val)]
    best_lasso_lambda = lasso_lambdas[np.argmin(lasso_val)]
    compare_test_models(train_X, train_y, test_X, test_y, best_ridge_lambda, best_lasso_lambda)


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
