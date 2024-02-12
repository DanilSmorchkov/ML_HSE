import numpy as np
from sklearn.preprocessing import StandardScaler
# Task 1


def mse(y_true: np.ndarray, y_predicted: np.ndarray):
    return np.sum((y_predicted-y_true)**2) / y_true.shape[0]


def r2(y_true: np.ndarray, y_predicted: np.ndarray):
    return 1 - mse(y_true, y_predicted) / (1/y_true.shape[0] * np.sum((y_true - np.mean(y_true))**2))

# Task 2


class NormalLR:
    def __init__(self):
        self.weights = None  # Save weights here
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.weights
    
# Task 3


class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.006):
        self.weights = None  # Save weights here
        self.iterations = iterations
        self.alpha = alpha
        self.l = l
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)
        y = y.reshape((y.shape[0], 1))
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.weights = np.random.random((X.shape[1], 1))
        for iter_ in range(self.iterations):
            grad = (2 * X.T @ (X @ self.weights - y)) / X.shape[0] + self.l * np.sign(self.weights)
            self.weights -= self.alpha * grad

    def predict(self, X: np.ndarray):
        X = self.scaler.transform(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = X @ self.weights
        return y_pred.flatten()

# Task 4


def get_feature_importance(linear_regression):
    weights = linear_regression.weights[1:,].flatten()
    importance = np.abs(weights)
    return importance


def get_most_important_features(linear_regression):
    weights = linear_regression.weights[1:,].flatten()
    most_important = np.argsort(np.abs(weights))[::-1]
    return most_important
