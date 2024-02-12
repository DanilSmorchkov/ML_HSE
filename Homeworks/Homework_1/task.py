import numpy as np
import random
import copy
import pandas as pd
from typing import NoReturn, Tuple, List


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
        0 --- злокачественной (B).


    """
    dataframe = pd.read_csv(path_to_csv, index_col=False).replace({'label': {'M': 1, 'B': 0}}).sample(frac=1).to_numpy()
    X = dataframe[:, 1:]
    y = dataframe[:, 0]
    return X, y


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток,
        1 если сообщение содержит спам, 0 если не содержит.

    """
    dataframe = pd.read_csv(path_to_csv, index_col=False).sample(frac=1).to_numpy()
    X = dataframe[:, :-1]
    y = dataframe[:, -1]
    return X, y

# Task 2


def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    X_train, X_test = np.split(X, [int(ratio * len(X))])
    y_train, y_test = np.split(y, [int(ratio * len(y))])
    return X_train, y_train, X_test, y_test


# Task 3


def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    labels = np.unique(y_true)
    accuracy = np.mean(y_true == y_pred)   # Среднее от маски из 0 и 1 и есть accuracy
    precision = np.array([])
    recall = np.array([])
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        tn = np.sum((y_true != label) & (y_pred != label))   # Зато написал!
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        precision = np.append(precision, prec)
        recall = np.append(recall, rec)

    return precision, recall, accuracy


# Task 4


class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40, depth=0, init_indices=None, parent=None):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.parent = parent
        if depth == 0:
            self.init_X = X
            initial_indices = np.arange(X.shape[0])
        else:
            initial_indices = init_indices
        dimension = X.shape[1]
        count_circle = 0
        while True:
            self.num_feature = depth % dimension
            self.median = np.median(X[:, self.num_feature])
            right_indices = np.where(X[:, self.num_feature] >= self.median)[0]
            left_indices = np.where(X[:, self.num_feature] < self.median)[0]
            right_initial_indices = initial_indices[right_indices]
            left_initial_indices = initial_indices[left_indices]
            if right_indices.shape[0] >= leaf_size and left_indices.shape[0] >= leaf_size:
                self.right_child = KDTree(X[right_indices], leaf_size=leaf_size, depth=depth + 1,
                                          init_indices=right_initial_indices, parent=self)
                self.left_child = KDTree(X[left_indices], leaf_size=leaf_size, depth=depth + 1,
                                         init_indices=left_initial_indices, parent=self)
                break
            elif count_circle != dimension - 1:
                depth += 1
                count_circle += 1
                continue
            else:
                self.right_child = None
                self.left_child = None
                self.leaf_neighbors = init_indices
                break

    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.

        """

        list_neighbors = []
        for xi in X:
            path_to_leaf = []
            indices_in_leaf = self.go_to_leaf_and_get_indices(xi, path_to_leaf)
            values_leaf = self.init_X[indices_in_leaf]
            distances = np.linalg.norm(xi - values_leaf, axis=1)
            indices_leaf_k = np.argsort(distances)[:k]
            indices_in_leaf = indices_in_leaf[indices_leaf_k]
            distances = distances[indices_leaf_k]
            for path_i in reversed(path_to_leaf):
                if path_i.parent is None:
                    break
                indices_in_leaf, distances = self.go_up(xi, path_i, distances, k, indices_in_leaf)

            list_neighbors.append(list(indices_in_leaf))
        return list_neighbors

    def go_to_leaf_and_get_indices(self, x, path):
        path.append(self)
        if self.right_child is not None and self.left_child is not None:
            if x[self.num_feature] >= self.median:
                return self.right_child.go_to_leaf_and_get_indices(x, path)
            else:
                return self.left_child.go_to_leaf_and_get_indices(x, path)
        else:
            return self.leaf_neighbors

    def go_up(self, x, path, distance, k, ind):
        if np.any(np.abs(path.parent.median - x[path.parent.num_feature]) <= distance) or distance.shape[0] < k:
            if path is path.parent.left_child:
                all_leaf_near = self.get_all_leaf_from_node(path.parent.right_child)
            else:
                all_leaf_near = self.get_all_leaf_from_node(path.parent.left_child)
            value = self.init_X[all_leaf_near]
            distances_near = np.linalg.norm(x - value, axis=1)
            all_dist = np.append(distances_near, distance)
            indecies = np.append(all_leaf_near, ind)
            sort_indices_all_k = np.argsort(all_dist)[:k]
            distance = all_dist[sort_indices_all_k]
            return indecies[sort_indices_all_k], distance
        else:
            return ind, distance

    def get_all_leaf_from_node(self, node):
        indices = np.array([], dtype=np.int32)
        if node.right_child is not None and node.left_child is not None:
            indices = np.append(indices, self.get_all_leaf_from_node(node.right_child))
            indices = np.append(indices, self.get_all_leaf_from_node(node.left_child))
        else:
            indices = np.append(indices, node.leaf_neighbors)
        return indices



# Task 5


class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.k = n_neighbors
        self.leaf_size = leaf_size
        self.tree = None
        self.y = None

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.tree = KDTree(X, leaf_size=self.leaf_size)
        self.y = y

    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.


        """
        labels = np.unique(self.y)
        near_neighbors = self.tree.query(X, k=self.k)
        near_labels = self.y[near_neighbors]
        prob = None
        for label in labels:
            if prob is None:
                prob = np.mean(near_labels == label, axis=1)
            else:
                prob = np.vstack((prob, np.mean(near_labels == label, axis=1)))
        return list(prob.T)

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.


        """
        return np.argmax(self.predict_proba(X), axis=1)
