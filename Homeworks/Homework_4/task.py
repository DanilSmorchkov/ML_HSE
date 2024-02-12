from typing import NoReturn
import numpy as np


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.w = None
        self.iters = iterations
        self.unique_labels = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        self.unique_labels = np.unique(y)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.w = np.zeros(X.shape[1])
        y = y.reshape((X.shape[0], 1))
        y[y == self.unique_labels[0]] = -1
        y[y == self.unique_labels[1]] = 1
        for i in range(self.iters):
            H = np.sign(X @ self.w.reshape(X.shape[1], 1))
            diff = np.where(y - H != 0)[0]
            self.w += (X[diff].T @ y[diff]).reshape(X.shape[1])
        y[y == 1] = self.unique_labels[1]
        y[y == -1] = self.unique_labels[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        ans = np.sign(X @ self.w.reshape((X.shape[1], 1))).flatten()
        ans = ans.astype(np.int8)
        ans[ans == 1] = self.unique_labels[1]
        ans[ans == -1] = self.unique_labels[0]
        return ans

# Task 2


class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """
        self.iters = iterations
        self.w = None
        self.unique_labels = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        self.unique_labels = np.unique(y)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        w = np.zeros(X.shape[1])
        best_diff = np.inf
        y = y.reshape((X.shape[0], 1))
        y[y == self.unique_labels[0]] = -1
        y[y == self.unique_labels[1]] = 1
        for i in range(self.iters):
            H = np.sign(X @ w.reshape(X.shape[1], 1))
            diff = np.where(y - H != 0)[0]
            if best_diff > diff.shape[0]:
                best_diff = diff.shape[0]
                self.w = w.copy()
            w += (X[diff].T @ y[diff]).reshape(X.shape[1])
        y[y == 1] = self.unique_labels[1]
        y[y == -1] = self.unique_labels[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        ans = np.sign(X @ self.w.reshape((X.shape[1], 1))).flatten()
        ans = ans.astype(np.int8)
        ans[ans == 1] = self.unique_labels[1]
        ans[ans == -1] = self.unique_labels[0]
        return ans


# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    feature_1 = np.sum(np.sum(images, axis=2), axis=1).reshape((images.shape[0], 1))
    image_rev = np.flip(images, axis=1)
    feature_2 = np.sum(np.sum(np.abs(images - image_rev), axis=2), axis=1).reshape((images.shape[0], 1))
    return np.concatenate((feature_1, feature_2), axis=1)
