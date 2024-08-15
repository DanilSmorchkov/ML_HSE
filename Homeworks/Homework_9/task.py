from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
import queue

# Task 0


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    labels, counts = np.unique(x, return_counts=True)
    return (counts / x.shape[0]) @ (1 - (counts / x.shape[0]))


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    labels, counts = np.unique(x, return_counts=True)
    return - (counts / x.shape[0]) @ (np.log2(counts / x.shape[0]))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    l, r = left_y.shape[0], right_y.shape[0]
    y_concat = np.concatenate((left_y, right_y))
    return (l + r) * criterion(y_concat) - l * criterion(left_y) - r * criterion(right_y)


# Task 1

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, ys):
        labels, counts = np.unique(ys, return_counts=True)
        self.y = labels[np.argmax(counts)]
        self.probability = {label: counts[i] / ys.shape[0] for i, label in enumerate(labels)}

class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """

    def __init__(self, split_dim: Union[int, None], split_value: Union[float, None],
                 left: Union['DecisionTreeNode', DecisionTreeLeaf, None],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf, None],
                 depth: int = 0) -> None:
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        self.depth = depth


class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        self.root = None
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = int(np.sqrt(X.shape[1])) if max_features == "auto" else max_features
        self.fit(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        self.root = self.create_best_partition(X, y)

    def create_best_partition(self, X_part: np.ndarray, y_part:np.ndarray, depth=0):
        if len(np.unique(y_part)) == 1:
            return DecisionTreeLeaf(y_part)
        if self.max_depth is not None:
            if depth == self.max_depth:
                return DecisionTreeLeaf(y_part)

        split_feat_ind, best_mask, max_gain, split_value = self._find_best_split(X_part, y_part)

        if max_gain == -0.1:
            return DecisionTreeLeaf(y_part)

        left_X = X_part[best_mask]
        left_y = y_part[best_mask]

        right_X = X_part[~best_mask]
        right_y = y_part[~best_mask]

        if len(left_X) == 0 or len(right_X) == 0:
            return DecisionTreeLeaf(y_part)

        return DecisionTreeNode(split_dim=split_feat_ind, split_value=split_value,
                                right=self.create_best_partition(right_X, right_y, depth+1),
                                left=self.create_best_partition(left_X, left_y, depth+1))

    def _find_best_split(self, X_part: np.ndarray, y_part: np.ndarray):
        max_gain = -0.1
        split_feat_ind = None
        split_value = 0.5
        feature_inds = np.random.choice(np.arange(0, X_part.shape[1]), size=self.max_features, replace=False)
        best_mask = None

        for feature_ind in feature_inds:
            mask = X_part[:, feature_ind] < split_value
            y_left = y_part[mask]
            y_right = y_part[~mask]
            if len(y_left) >= self.min_samples_leaf and len(y_right) >= self.min_samples_leaf:
                curr_gain = gain(left_y=y_left, right_y=y_right, criterion=self.criterion)
                if curr_gain > max_gain:
                    best_mask = mask
                    max_gain = curr_gain
                    split_feat_ind = feature_ind
        return split_feat_ind, best_mask, max_gain, split_value

    def predict(self, X: np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        list_ans = []
        for x_i in X:
            node = self.root
            while isinstance(node, DecisionTreeNode):
                if x_i[node.split_dim] >= node.split_value:
                    node = node.right
                else:
                    node = node.left
            list_ans.append(node.probability)
        return [max(p.keys(), key=lambda k: p[k]) for p in list_ans]

    
# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.out_X = []
        self.out_y = []
        self.estimators = []

    def fit(self, X, y):
        for i in range(self.n_estimators):
            bootstrap_inds = np.random.randint(low=0, high=X.shape[0], size=X.shape[0])
            out_of_bag_inds = np.setdiff1d(np.arange(X.shape[0]), bootstrap_inds)
            self.out_X.append(X[out_of_bag_inds])
            self.out_y.append(y[out_of_bag_inds])
            boots_X = X[bootstrap_inds]
            boots_y = y[bootstrap_inds]
            self.estimators.append(DecisionTree(X=boots_X, y=boots_y, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                                criterion=self.criterion, max_features=self.max_features))

    def predict(self, X):
        pred = []
        for estimator in self.estimators:
            pred.append(estimator.predict(X))
        pred = np.array(pred)
        preds = []
        for x_ind in range(pred.shape[1]):
            values, counts = np.unique(pred[:, x_ind], return_counts=True)
            max_value = values[counts == np.max(counts)][0]
            preds.append(max_value)
        preds = np.array(preds)
        return preds

# Task 3


def feature_importance(rfc):
    matrix_importance = np.zeros((len(rfc.estimators), rfc.out_X[0].shape[1]))
    for ind, estimator in enumerate(rfc.estimators):
        preds = estimator.predict(rfc.out_X[ind])
        acc = np.mean(rfc.out_y[ind] == preds)
        for feature_ind in range(rfc.out_X[ind].shape[1]):
            feature_obj = np.random.choice(rfc.out_X[ind][:, feature_ind],
                                           size=rfc.out_X[ind][:, feature_ind].shape[0],
                                           replace=False)
            out_bag_X_shuff = rfc.out_X[ind].copy()
            out_bag_X_shuff[:, feature_ind] = feature_obj
            out_bag_shuff_pred = estimator.predict(out_bag_X_shuff)
            accuracy_shuff = np.mean(rfc.out_y[ind] == out_bag_shuff_pred)
            matrix_importance[ind, feature_ind] = acc - accuracy_shuff
    return np.mean(matrix_importance, axis=0)


# Task 4

rfc_age = RandomForestClassifier(max_depth=15, n_estimators=15, min_samples_leaf=15)
rfc_gender = RandomForestClassifier(max_depth=15, n_estimators=15, min_samples_leaf=15)

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model
catboost_rfc_age = CatBoostClassifier(verbose=False)
catboost_rfc_age.load_model(__file__[:-7] + 'catboost_model1.dump')

catboost_rfc_gender = CatBoostClassifier(verbose=False)
catboost_rfc_gender.load_model(__file__[:-7] + 'catboost_model2.dump')
