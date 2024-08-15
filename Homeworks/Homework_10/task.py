import string
import re
import numpy as np
import pandas
import random
import copy
from typing import NoReturn

# Task 1


def cyclic_distance(points, dist):
    points = np.array(points)
    distances = dist(points, np.roll(points, -1, axis=0))
    return np.sum(distances)


def l2_distance(p1, p2):
    if p1.ndim == 1:
        p1 = p1.reshape(1, -1)
        p2 = p2.reshape(1, -1)
    return np.linalg.norm(p1 - p2, axis=1)


def l1_distance(p1, p2):
    if p1.ndim == 1:
        p1 = p1.reshape(1, -1)
        p2 = p2.reshape(1, -1)
    return np.sum(np.abs(p1 - p2), axis=1)


# Task 2

class HillClimb:
    def __init__(self, max_iterations, dist, mode='change'):
        self.max_iterations = max_iterations
        self.dist = dist # Do not change
        self.n = None
        self.mode = mode
    
    def optimize(self, X):
        return self.optimize_explained(X)[-1]
    
    def optimize_explained(self, X):
        permutation_history = np.expand_dims(np.arange(X.shape[0]), axis=0)
        X = X.copy()
        self.n = X.shape[0]
        distance = cyclic_distance(X, self.dist)
        last_distance = distance
        for _ in range(self.max_iterations):
            inds = permutation_history[-1]
            curr_dist = last_distance
            best_distance = last_distance
            best_ind = None
            for i in range(self.n-1):
                for j in range(i+1, self.n):
                    changed_inds = inds.copy()
                    new_dist, new_inds = self._change_dist(X, changed_inds, i, j, curr_dist)
                    if new_dist < best_distance:
                        best_distance = new_dist
                        best_ind = new_inds

            if best_ind is None:
                break

            last_distance = best_distance
            permutation_history = np.concatenate((permutation_history, np.expand_dims(best_ind, axis=0)), axis=0)

        return permutation_history

    def _change_dist(self, X, inds, i, j, cur_dist):
        new_dist: float = 0.0
        if self.mode == 'change':
            old_delta_1 = self.dist(X[inds[i]], X[inds[i - 1]])
            old_delta_2 = self.dist(X[inds[i]], X[inds[(i + 1) % self.n]])
            old_delta_3 = self.dist(X[inds[j]], X[inds[j - 1]])
            old_delta_4 = self.dist(X[inds[j]], X[inds[(j + 1) % self.n]])

            new_dist = cur_dist - old_delta_1 - old_delta_2 - old_delta_3 - old_delta_4

            inds[i], inds[j] = inds[j], inds[i]

            new_delta_1 = self.dist(X[inds[i]], X[inds[i - 1]])
            new_delta_2 = self.dist(X[inds[i]], X[inds[(i + 1) % self.n]])
            new_delta_3 = self.dist(X[inds[j]], X[inds[j - 1]])
            new_delta_4 = self.dist(X[inds[j]], X[inds[(j + 1) % self.n]])

            new_dist += new_delta_1 + new_delta_2 + new_delta_3 + new_delta_4

        elif self.mode == 'reverse':
            old_delta_1 = self.dist(X[inds[i]], X[inds[i - 1]])
            old_delta_4 = self.dist(X[inds[j]], X[inds[(j + 1) % self.n]])

            new_dist = cur_dist - old_delta_1 - old_delta_4

            inds = np.concatenate((inds[:i], np.flip(inds[i:j]), inds[j:]))

            new_delta_1 = self.dist(X[inds[i]], X[inds[i - 1]])
            new_delta_4 = self.dist(X[inds[j]], X[inds[(j + 1) % self.n]])

            new_dist += new_delta_1 + new_delta_4

        return new_dist, inds


# Task 3


class Genetic:
    def __init__(self, iterations, population, survivors, distance, prob_mutation=0.2):
        self.pop_size = population
        self.surv_size = survivors
        self.dist = distance
        self.iters = iterations
        self.prob_mutation = prob_mutation

        self.best_permutation = None
        self.best_distance = None
        self.X = None
        self.n = None

    def optimize(self, X: np.ndarray):
        _ = self.optimize_explain(X)
        return self.best_permutation

    def optimize_explain(self, X: np.ndarray):
        self.X = X
        self.n = X.shape[0]
        first_generation = np.array([np.random.choice(self.n, size=self.n, replace=False) for i in range(self.pop_size)])
        all_generations = np.expand_dims(first_generation, axis=0)

        for _ in range(self.iters):
            survivors = self._choose_survivors(all_generations[-1])
            children = None
            for i in range(self.pop_size // self.surv_size):
                np.random.shuffle(survivors)
                parents_pair = np.split(survivors, self.surv_size // 2, axis=0)

                for pair in parents_pair:
                    two_children = self._crossing(pair)
                    proba = np.random.randint(1, 1001)
                    if proba <= self.prob_mutation * 1000:
                        self._mutations(two_children[0])
                        self._mutations(two_children[1])
                    if children is None:
                        children = two_children
                    else:
                        children = np.vstack((children, two_children))
            all_generations = np.concatenate((all_generations, np.expand_dims(children, 0)), axis=0)
        return all_generations

    def _mutations(self, permutation: np.ndarray):
        i, j = np.random.choice(np.arange(permutation.shape[0]), 2, replace=False)
        permutation[i], permutation[j] = permutation[j], permutation[i]

    def _crossing(self, pair: np.ndarray):
        child1 = np.array([])
        child2 = np.array([])
        first, second = pair
        i, j = (f := np.random.randint(self.n)), f + self.n // 2
        if j <= self.n:
            child1 = np.append(child1, first[i:j]).astype(int)
            child2 = np.append(child2, [np.append(first[:i], first[j:])]).astype(int)
        else:
            child2 = np.append(child2, first[j % self.n:i]).astype(int)
            child1 = np.append(child1, [np.append(first[:j % self.n], first[i:])]).astype(int)
        second_remaining1 = second[~np.isin(second, child1)]
        child1 = np.append(child1, second_remaining1).astype(int)
        second_remaining2 = second[~np.isin(second, child2)]
        child2 = np.append(child2, second_remaining2).astype(int)

        children = np.vstack((child1, child2))

        return children

    def _choose_survivors(self, permutations: np.ndarray):
        func = lambda l: cyclic_distance(self.X[l], self.dist)
        survivors = permutations[np.apply_along_axis(func, 1, permutations).argsort()][:self.surv_size]
        if self.best_distance is None or self.best_distance > cyclic_distance(self.X[survivors[0]], self.dist):
            self.best_permutation = survivors[0]
            self.best_distance = cyclic_distance(self.X[survivors[0]], self.dist)

        return survivors


# gen = Genetic(200, 100, 20, l1_distance)
#
#
# def synthetic_points(count=25, dims=2):
#     return np.random.randint(40, size=(count, dims))
#
#
# X = synthetic_points()
# populations = gen.optimize_explain(X)
# Task 4

class BoW:
    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        """
        Составляет словарь, который будет использоваться для векторизации предложений.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ),
            по которому будет составляться словарь.
        voc_limit : int
            Максимальное число слов в словаре.

        """
        self.voc_limit = voc_limit
        tokens = self._make_arr_words(X)
        unique_words, counts = np.unique(tokens, return_counts=True)
        self.vocabulary = unique_words[np.argsort(-counts)][:self.voc_limit]

    def _make_arr_words(self, X: np.ndarray):
        tokens = self._processing(X)
        return tokens

    def _processing(self, X):
        lst_words = []
        for doc in X:
            lst_words.extend(self._tokenize(doc))
        return np.array(lst_words)

    def _tokenize(self, doc):
        s = re.sub(r"^\d+\s|\s\d+\s|\s\d+$", " ", doc)
        return re.split(r'\W+', s.lower())

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизует предложения.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ),
            который необходимо векторизовать.

        Return
        ------
        np.ndarray
            Матрица векторизованных предложений размерности (n_sentences, vocab_size)
        """
        embeddings = np.zeros((X.shape[0], self.voc_limit))
        doc: str
        for i, doc in enumerate(X):
            tokens = self._tokenize(doc)
            unique_words, counts = np.unique(tokens, return_counts=True)
            for word, freq in zip(unique_words, counts):
                indx = np.where(self.vocabulary == word)
                embeddings[i, indx] = freq
        return embeddings


# Task 5

class NaiveBayes:
    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Параметр аддитивной регуляризации.
        """
        self.log_x_y = None
        self.log_y = None
        self.num_classes = None
        self.classes = None
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Оценивает параметры распределения p(x|y) для каждого y.
        """
        self.classes, counts_y = np.unique(y, return_counts=True)
        self.num_classes = self.classes.size
        self.log_y = np.log(counts_y / np.sum(counts_y))

        x_y = np.zeros((self.num_classes, X.shape[1]))
        for i, cls in enumerate(self.classes):
            inds = np.where(cls == y)[0]
            X_y_class_i = X[inds]
            N_y = X_y_class_i.sum()
            x_y[i] = (np.sum(X_y_class_i, axis=0) + self.alpha) / (N_y + self.alpha * X.shape[1])
        self.log_x_y = np.log(x_y)

    def predict(self, X: np.ndarray) -> list:
        """
        Return
        ------
        list
            Предсказанный класс для каждого элемента из набора X.
        """
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]

    def log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return
        ------
        np.ndarray
            Для каждого элемента набора X - логарифм вероятности отнести его к каждому классу.
            Матрица размера (X.shape[0], n_classes)
        """
        log_matrix = np.zeros((X.shape[0], self.num_classes))
        for i, x in enumerate(X):
            log_matrix[i] = np.sum(x * self.log_x_y, axis=1) + self.log_y
        return log_matrix
