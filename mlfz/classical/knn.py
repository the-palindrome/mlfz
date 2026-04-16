import numpy as np
from .base import Model


def _validate_metric(metric: str):
    if metric != "euclidean":
        raise NotImplementedError(
            "Only the 'euclidean' metric is supported without SciPy."
        )


def _euclidean_distance_map(X_training, X):
    X_training = np.asarray(X_training)
    X = np.asarray(X)

    X_training_sq = np.sum(X_training * X_training, axis=1, keepdims=True)
    X_sq = np.sum(X * X, axis=1, keepdims=True).T
    distances_sq = X_training_sq + X_sq - 2 * (X_training @ X.T)
    return np.sqrt(np.maximum(distances_sq, 0))


class KNNClassifier(Model):
    def __init__(self, k: int, metric: str = "euclidean"):
        self.k = k
        _validate_metric(metric)
        self.metric = metric

    def fit(self, X, Y):
        self.X_training = X
        self.Y_training = Y

    def predict(self, X):
        dmap = _euclidean_distance_map(self.X_training, X)
        nearest_idx = np.argsort(dmap, axis=0)[: self.k, :]
        nearest_lbl = self.Y_training[nearest_idx]
        pred = np.array([np.bincount(labels).argmax() for labels in nearest_lbl.T])
        return pred


class KNNRegressor(Model):
    def __init__(self, k: int, metric: str = "euclidean"):
        self.k = k
        _validate_metric(metric)
        self.metric = metric

    def fit(self, X, Y):
        self.X_training = X
        self.Y_training = Y

    def predict(self, X):
        dmap = _euclidean_distance_map(self.X_training, X)
        nearest_idx = np.argsort(dmap, axis=0)[: self.k, :]
        nearest_lbl = self.Y_training[nearest_idx]
        pred = nearest_lbl.mean(axis=0)
        return pred
