from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import copy
import cv2
from collections import deque
from typing import NoReturn

# Task 1


class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", 
                 max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """
        
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None
        self.max_min = None
        
    def fit(self, X: np.array, y=None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        
        """
        init_dict = {'random': self.random_init, 'sample': self.sample_init, 'k-means++': self.k_mean_plus_init}
        dimension_fut = X.shape[1]
        init_dict[self.init](X, dimension_fut)
        for i in range(self.max_iter):
            prev_centroids = self.centroids.copy()
            dists = np.zeros((X.shape[0], self.n_clusters))
            for ci in range(self.n_clusters):
                dists[:, ci] = np.linalg.norm(X - self.centroids[ci, :], axis=1)
            groupid = np.argmin(dists, axis=1)
            for ci in range(self.n_clusters):
                cluster_X = X[groupid == ci]
                if not cluster_X.shape[0]:
                    if not np.all(self.max_min):
                        self.max_min = np.array([np.max(X, axis=0), np.min(X, axis=0)])
                    self.centroids[ci, :] = np.random.uniform(low=self.max_min[1], high=self.max_min[0],
                                                              size=(1, dimension_fut))
                else:
                    self.centroids[ci, :] = np.mean(cluster_X, axis=0)
            if np.all(prev_centroids == self.centroids):
                break

    def random_init(self, X, dim):
        if not np.all(self.max_min):
            self.max_min = np.array([np.max(X, axis=0), np.min(X, axis=0)])
        self.centroids = np.random.uniform(low=self.max_min[1], high=self.max_min[0], size=(self.n_clusters, dim))

    def sample_init(self, X, dim):
        ridx = np.random.choice(X.shape[0], self.n_clusters)
        self.centroids = X[ridx, :]

    def k_mean_plus_init(self, X, dim):
        ridx = np.random.choice(X.shape[0], 1)
        self.centroids = X[ridx, :]
        while self.centroids.shape[0] != self.n_clusters:
            dists = np.zeros((X.shape[0], self.centroids.shape[0]))
            for ci in range(self.centroids.shape[0]):
                dists[:, ci] = np.linalg.norm(X - self.centroids[ci, :], axis=1)
            min_centr_dist = np.min(dists, axis=1) ** 2
            prob = np.cumsum(min_centr_dist / np.sum(min_centr_dist))
            random_point = np.random.random()
            new_centroid = X[np.where(prob > random_point)[0][0], :]
            self.centroids = np.append(self.centroids, new_centroid.reshape((1, X.shape[1])), axis=0)

    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        
        """
        dists = np.zeros((X.shape[0], self.n_clusters))
        for ci in range(self.n_clusters):
            dists[:, ci] = np.linalg.norm(X - self.centroids[ci, :], axis=1)
        groupid = np.argmin(dists, axis=1)
        return groupid
    
# Task 2


class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        
        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть 
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean 
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.leaf_size = leaf_size
        
    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        neighbors = tree.query_radius(X, self.eps)
        core_points_ind = set(np.where(np.array([len(i) for i in neighbors]) >= self.min_samples)[0])
        adj = {i: set() for i in range(X.shape[0])}
        for core_point in core_points_ind:
            for neighbor in neighbors[core_point]:
                if neighbor in core_points_ind:
                    adj[core_point].add(neighbor)
                    adj[neighbor].add(core_point)
        colors = [-1] * X.shape[0]
        color = 0
        for core_point in core_points_ind:
            if colors[core_point] == -1:
                self.dfs_with_colors(core_point, colors, color, adj)
                color += 1
        for i, point_color in enumerate(colors):
            if point_color == -1 and (core_neighbors := list(core_points_ind.intersection(set(neighbors[i])))):
                if self.metric == 'euclidean':
                    distances_to_cores = np.linalg.norm(X[core_neighbors] - X[i], axis=1)
                elif self.metric == 'manhattan':
                    distances_to_cores = np.linalg.norm(X[core_neighbors] - X[i], axis=1, ord=1)
                elif self.metric == 'chebyshev':
                    distances_to_cores = np.linalg.norm(X[core_neighbors] - X[i], axis=1, ord=np.inf)
                nearest_core_index = distances_to_cores.argsort()[0]
                colors[i] = colors[core_neighbors[nearest_core_index]]
        return colors

    def dfs_with_colors(self, v, colors, color, adj):
        colors[v] = color
        for u in adj[v]:
            if colors[u] == -1:
                self.dfs_with_colors(u, colors, color, adj)



# Task 3


class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        dist_link = {'average': self.link_average, 'single': self.link_single, 'complete': self.link_complete}
        cluster_distances = np.linalg.norm(np.repeat(X, X.shape[0], axis=0) - np.tile(X, reps=(X.shape[0], 1)),
                                           axis=1).reshape((X.shape[0], X.shape[0]))
        cluster_indices = np.arange(X.shape[0], dtype=np.int32)
        np.fill_diagonal(cluster_distances, np.NaN)
        cluster_labels = np.zeros(X.shape[0], dtype=np.int32)
        clusters = {i: {i} for i in range(X.shape[0])}
        while cluster_distances.shape[0] > self.n_clusters:
            flat_min = np.nanargmin(cluster_distances)
            matrix_first_cluster = flat_min // cluster_distances.shape[0]
            matrix_second_cluster = flat_min % cluster_distances.shape[0]
            first_cluster = cluster_indices[matrix_first_cluster]
            second_cluster = cluster_indices[matrix_second_cluster]
            new_distances = dist_link[self.linkage](clusters, cluster_distances, first_cluster,
                                                    second_cluster, matrix_first_cluster, matrix_second_cluster)
            cluster_distances[matrix_first_cluster, :] = new_distances
            cluster_distances[:, matrix_first_cluster] = new_distances
            cluster_distances = np.delete(cluster_distances, matrix_second_cluster, axis=0)
            cluster_distances = np.delete(cluster_distances, matrix_second_cluster, axis=1)
            cluster_indices = np.delete(cluster_indices, matrix_second_cluster)
            clusters[first_cluster].update(clusters[second_cluster])
            del clusters[second_cluster]
        color = 0
        for cluster_label in clusters:
            cluster_labels[list(clusters[cluster_label])] = color
            color += 1
        return cluster_labels

    def link_average(self, clust, clust_dist, first, second, matrix_first, matrix_second):
        new_dist = np.array(
            [(len(clust[first]) * clust_dist[matrix_first, i_cluster] +
              len(clust[second]) * clust_dist[matrix_second, i_cluster]) /
             (len(clust[first]) + len(clust[second]))
             for i_cluster in range(clust_dist.shape[0])])
        return new_dist

    def link_single(self, clust, clust_dist, first, second, matrix_first, matrix_second):
        new_dist = np.array([np.min((clust_dist[matrix_first, i_cluster],
                                    clust_dist[matrix_second, i_cluster])) for i_cluster in range(clust_dist.shape[0])])
        return new_dist

    def link_complete(self, clust, clust_dist, first, second, matrix_first, matrix_second):
        new_dist = np.array([np.max((clust_dist[matrix_first, i_cluster],
                                    clust_dist[matrix_second, i_cluster])) for i_cluster in range(clust_dist.shape[0])])
        return new_dist
