import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False

# Task 1


class LinearSVM:
    def __init__(self, C: float):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        
        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        # Выразим оптимизационную задачу
        P = matrix(np.outer(y, y) * (X @ X.T))
        q = matrix(-np.ones(y.shape[0], dtype='float'))
        G = matrix(np.concatenate((-np.eye(y.shape[0], dtype='float'), np.eye(y.shape[0], dtype='float')),
                                  axis=0))
        h = matrix(np.concatenate((np.zeros(y.shape[0], dtype='float'), np.full(y.shape[0], self.C)), axis=0))
        A = matrix(y.astype('float'), size=(1, y.shape[0]))
        b = matrix(0.0)

        # Решение двойственной задачи
        alpha = np.ravel(solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)['x'])

        # Найдем опорные вектора
        support = alpha > 1e-3
        self.support = np.where((alpha > 1e-3) * (alpha < self.C - 1e-5))[0]
        alpha = alpha[support]
        X_suppport = X[support]
        support_y = y[support]


        # Веса и смещение
        self.w = np.sum(np.expand_dims(alpha, axis=1) * np.expand_dims(support_y, axis=1) * X_suppport, axis=0)
        self.b = self.w @ X[self.support[0]] - y[self.support[0]]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """
        return X @ self.w - self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))
    
# Task 2


def get_polynomial_kernel(c=1, power=2):
    """Возвращает полиномиальное ядро с заданной константой и степенью"""
    def polynomial_kernel(X, y):
        return (c + X @ y.T) ** power
    return polynomial_kernel


def get_gaussian_kernel(sigma=1.):
    """Возвращает ядро Гаусса с заданным коэффицинтом сигма"""
    def gaussian_kernel(X, y):
        return np.exp(-sigma * np.linalg.norm(X - y, axis=1) ** 2)
    return gaussian_kernel

# Task 3


class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """
        
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.
        
        """
        self.C = C
        self.kernel = kernel
        self.support = None
        self.b = None
        self.all_support = None
        self.alpha_support = None
        self.y_support = None
        self.X_all_support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp
        
        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X 
            (можно считать, что равны -1 или 1). 
        
        """
        # Решаем оптимизационную задачу, но с ядром вместо скалярного произведения
        K = np.zeros((y.shape[0], y.shape[0]))
        for i, x_i in enumerate(X):
            K[i, :] = self.kernel(X, x_i)
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(y.shape[0], dtype='float'))
        G = matrix(np.concatenate((-np.eye(y.shape[0], dtype='float'), np.eye(y.shape[0], dtype='float')),
                                  axis=0))
        h = matrix(np.concatenate((np.zeros(y.shape[0], dtype='float'), np.full(y.shape[0], self.C)), axis=0))
        A = matrix(y.astype('float'), size=(1, y.shape[0]))
        b = matrix(0.0)

        # Решение двойственной задачи
        alpha = np.ravel(solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)['x'])

        # Опорные вектора
        self.all_support = alpha > 1e-5
        self.support = np.where((alpha > 1e-5) * (alpha < self.C - 1e-5))[0]
        self.alpha_support = alpha[self.all_support]
        self.X_all_support = X[self.all_support]
        self.y_support = y[self.all_support]

        # Смещение

        self.b = np.sum(self.alpha_support * self.y_support * self.kernel(self.X_all_support,
                                                                                  X[self.support[0]])) - y[self.support[0]]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).     
        
        """
        a = np.zeros(X.shape[0])
        for i, support in enumerate(self.X_all_support):
            a += self.alpha_support[i] * self.y_support[i] * self.kernel(X, support)
        a -= self.b
        return a

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.
        
        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.   
        
        """
        return np.sign(self.decision_function(X))