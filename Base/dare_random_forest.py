from .dare_tree import DaRETree
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error as mse


class dare_rf(BaseEstimator, RegressorMixin):
    def __init__(self, n_trees, max_depth, mode="greedy"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.mode = mode
        self.trees = [DaRETree(self.max_depth, self.mode) for _ in range(self.n_trees)]

    def fit(self, data):
        for tree in self.trees:
            bootstrap_sample = data[
                np.random.choice(data.shape[0], size=len(data), replace=True)
            ]
            tree.fit(bootstrap_sample)

    def predict(self, data):
        predictions = np.array([tree.predict(data) for tree in self.trees])
        return np.mean(predictions, axis=0)

    def delete_instances(self, instances):
        """
        Delete multiple instances from each tree in the forest.
        """
        for tree in self.trees:
            tree.delete_instances(instances)

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mse(y, y_pred)  # Negative MSE for optimization
