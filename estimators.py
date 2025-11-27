from abc import ABC, abstractmethod
from typing import NoReturn
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Lasso as SklearnLasso
import numpy as np
from base_estimator import BaseEstimator


class LinearRegression(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.model = SklearnLR()

    def _fit(self, X, y):
        self.model.fit(X, y)

    def _predict(self, X):
        return self.model.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_pred = self._predict(X)
        return np.mean((y - y_pred) ** 2)


class Lasso(BaseEstimator):
    def __init__(self, alpha: float = 1.0, include_intercept: bool = True):
        super().__init__()
        self.include_intercept_ = include_intercept
        self.model = SklearnLasso(alpha=alpha, fit_intercept=include_intercept, max_iter=10000)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_pred = self._predict(X)
        return np.mean((y - y_pred) ** 2)


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True):
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            # Add intercept column of ones
            X = np.c_[np.ones(X.shape[0]), X]
            identity = np.eye(X.shape[1]) # identity matrix
            identity[0, 0] = 0  # Don't regularize the intercept
        else:
            # identity matrix
            identity = np.eye(X.shape[1])

        # Ridge formula (X^T X + λI)^(-1) X^T y
        XtX = X.T @ X
        reg = self.lam_ * identity
        XtX_reg = XtX + reg
        Xty = X.T @ y

        self.coefs_ = np.linalg.pinv(XtX_reg) @ Xty  

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_pred = self._predict(X)
        return np.mean((y - y_pred) ** 2)


