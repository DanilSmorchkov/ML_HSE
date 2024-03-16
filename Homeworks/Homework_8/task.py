from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
import queue


# Task 1


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


# Task 2

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
        self.X = None
        self.y = None
        self.leaf = None


# Task 3


class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = DecisionTreeNode(depth=0, left=None, right=None, split_dim=None, split_value=None)
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth if max_depth else np.inf
        self.min_samples_leaf = min_samples_leaf

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
        self.root.X = X
        self.root.y = y
        que = queue.Queue()
        que.put(self.root)
        while not que.empty():
            node = que.get()
            max_gain = -0.1
            i_ = 0
            for ind_feature, feat in enumerate(node.X.T):
                sort_ind = np.argsort(feat)
                feat = feat[sort_ind]
                y_sorted = node.y[sort_ind]

                for ind in range(1, feat.shape[0]):
                    if ind < self.min_samples_leaf or y_sorted.shape[0] - ind < self.min_samples_leaf:
                        continue
                    if y_sorted[ind] == y_sorted[ind - 1] and max_gain != -0.1:
                        continue

                    y_right = y_sorted[ind:]

                    y_left = y_sorted[:ind]

                    gain_node = gain(y_left, y_right, self.criterion)
                    if gain_node > max_gain:
                        max_gain = gain_node

                        node.split_dim = ind_feature
                        node.split_value = feat[ind]
                        i_ = ind
            if max_gain == -0.1 or (node.y[node.X[:, node.split_dim] >= node.split_value].shape[0] == 0 or
                                    node.y[node.X[:, node.split_dim] < node.split_value].shape[0] == 0):
                node.leaf = DecisionTreeLeaf(node.y)
                continue
            if node.X.shape[0] - i_ < 2 * self.min_samples_leaf or node.depth == self.max_depth - 1 or max_gain < 0.1:
                node.right = DecisionTreeLeaf(node.y[node.X[:, node.split_dim] >= node.split_value])
            else:
                node.right = DecisionTreeNode(depth=node.depth + 1, left=None, right=None, split_dim=None,
                                              split_value=None)
                node.right.y = node.y[node.X[:, node.split_dim] >= node.split_value]
                node.right.X = node.X[node.X[:, node.split_dim] >= node.split_value]
                que.put(node.right)
            if i_ < 2 * self.min_samples_leaf or node.depth == self.max_depth - 1 or max_gain < 0.1:
                node.left = DecisionTreeLeaf(node.y[node.X[:, node.split_dim] < node.split_value])
            else:
                node.left = DecisionTreeNode(depth=node.depth + 1, left=None, right=None, split_dim=None,
                                             split_value=None)
                node.left.y = node.y[node.X[:, node.split_dim] < node.split_value]
                node.left.X = node.X[node.X[:, node.split_dim] < node.split_value]
                que.put(node.left)

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """
        list_ans = []
        for x_i in X:
            que = queue.Queue()
            que.put(self.root)
            while not que.empty():
                node = que.get()
                if isinstance(node, DecisionTreeNode) and node.leaf:
                    que.put(node.leaf)
                    continue
                if isinstance(node, DecisionTreeLeaf):
                    list_ans.append(node.probability)
                    break
                feature, value = node.split_dim, node.split_value
                if x_i[feature] >= value:
                    que.put(node.right)
                else:
                    que.put(node.left)
        return list_ans

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
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]



# noise = 0.35
# X, y = make_moons(1500, noise=noise)
# X_test, y_test = make_moons(200, noise=noise)
# tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
# tree.fit(X, y)
# plot_2d(tree, X, y)
# plot_roc_curve(y_test, tree.predict_proba(X_test))


# Task 4
task4_dtc = DecisionTreeClassifier(max_depth=5, )
