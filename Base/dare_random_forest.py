from .dar_tree import DaRETree
import numpy as np


class DaRERandomForest:
    def __init__(self, n_trees, max_depth, mode="greedy"):
        self.n_trees = n_trees
        self.trees = [DaRETree(max_depth, mode) for _ in range(n_trees)]

    def fit(self, data):
        for tree in self.trees:
            bootstrap_sample = data[
                np.random.choice(data.shape[0], size=len(data), replace=True)
            ]
            tree.fit(bootstrap_sample)

    def predict(self, data):
        predictions = np.array([tree.predict(data) for tree in self.trees])
        return np.mean(predictions, axis=0)

    def delete_instance(self, instance):
        for tree in self.trees:
            tree.delete_instance(instance)
