import numpy as np
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List, Tuple
from sklearn.datasets import make_blobs, make_moons, load_iris
# Task 1


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    _, counts = np.unique(x, return_counts=True)
    p = counts/len(x)
    gini_ = 1 - np.sum(p*p)
    return gini_
    pass


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    _, counts = np.unique(x, return_counts=True)
    p = counts/len(x)
    entropy_ = np.sum(-p*np.log2(p))
    return entropy_
    pass


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
    num_left = len(left_y)
    num_right = len(right_y)
    num_all = num_left + num_right
    I_left = criterion(left_y)
    I_right = criterion(right_y)
    I_all = criterion(np.hstack([left_y, right_y]))
    IG = I_all - num_left/num_all*I_left - num_right/num_all*I_right
    return IG
    pass


# Task 2

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, ys):
        y, count = np.unique(ys, return_counts=True)
        self.y = y[count.argmax()]


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

    def __init__(self, split_dim: int, split_value: float,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right

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
                 min_samples_leaf: int = 40):
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
        self.root = None
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_leaf = min_samples_leaf
        self.criterion = gini if criterion == 'gini' else entropy

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
        self.root = self._fit(X, y, depth=0)
        pass

    def _fit(self, X: np.ndarray, y: np.ndarray, depth: int) -> Union[DecisionTreeNode, DecisionTreeLeaf]:

        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) <= self.min_samples_leaf:
            return DecisionTreeLeaf(y)

        split_dim, split_value = self._find_best_split(X, y)
        if split_dim is None and split_value is None:
            return DecisionTreeLeaf(y)
        left_mask = X[:, split_dim] <= split_value
        right_mask = ~left_mask

        left_subtree = self._fit(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._fit(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(split_dim=split_dim, split_value=split_value, left=left_subtree, right=right_subtree)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:

        def search_feat(col):
            best_information_gain = -float('inf')
            best_split_value = 0.0
            # df = pandas.DataFrame({'feat': col, 'y': y})
            # df = df.sort_values(by='feat').reset_index()
            # r_bound = df['feat'][df['y'] != df['y'].shift(-1)][:-1]
            # div_values = r_bound.sample(int(len(r_bound)*0.2+1))
            # for div_value in div_values:
            #     left_mask = col <= div_value
            #     right_mask = ~left_mask
            #
            #     information_gain = gain(
            #         y[left_mask], y[right_mask], self.criterion)
            #
            #     if information_gain > best_information_gain:
            #         best_information_gain = information_gain
            #         best_split_value = div_value

            sort_ind = np.argsort(col)
            y_sorted = y[sort_ind]

            for index_value in range(1, y.shape[0]):
                if index_value < self.min_samples_leaf or y_sorted.shape[0] - index_value < self.min_samples_leaf:
                    continue
                if y_sorted[index_value] == y_sorted[index_value - 1] and best_information_gain != -float('inf'):
                    continue

                y_right = y_sorted[index_value:]

                y_left = y_sorted[:index_value]

                gain_node = gain(y_left, y_right, self.criterion)
                if gain_node > best_information_gain:
                    best_information_gain = gain_node
                    best_split_value = col[sort_ind][index_value]
            if best_information_gain == -float('inf') or best_information_gain == 0:
                best_split_value = None
                best_information_gain = None
            return best_information_gain, best_split_value

        best_split = np.apply_along_axis(search_feat, axis=0, arr=X)
        if best_split[0, 1] is None and best_split[1, 1] is None:
            return None, None
        best_split_dim = np.argmax(best_split[0, :])
        best_split_value = best_split[1][best_split_dim]


        return best_split_dim, best_split_value

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
        res = []
        for x in X:
            p = self.root
            while True:
                if isinstance(p, DecisionTreeLeaf):
                    res.append({p.y: 1.0})
                    break
                else:
                    split_value = p.split_value
                    split_feat = x[p.split_dim]
                    p = p.left if split_feat <= split_value else p.right
        return res

        pass

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


X, y = load_iris(return_X_y=True)
tree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, criterion='gini')
tree.fit(X, y)

# Task 4
task4_dtc = DecisionTreeClassifier(
    criterion='gini', max_depth=6, min_samples_leaf=30)
