import numpy as np
import random
from .node import TreeNode


class DaRETree:
    def __init__(self, max_depth, mode="greedy"):
        self.max_depth = max_depth
        self.mode = mode  # "greedy" or "random"
        self.root = None

    def fit(self, data):
        self.root = self._train(data, depth=0)

    def predict(self, data):
        return np.array([self._predict_instance(self.root, x) for x in data])

    def delete_instance(self, instance):
        self.root = self._delete(self.root, depth=0, instance=instance)

    def _train(self, data, depth):
        if self._stopping_criteria(data, depth):
            return self._leaf_node(data)

        if depth < self.max_depth:
            node = (
                self._random_node(data)
                if self.mode == "random"
                else self._greedy_node(data)
            )
        else:
            node = self._greedy_node(data)

        left_data, right_data = self._split_data(node, data)
        node.left = self._train(left_data, depth + 1)
        node.right = self._train(right_data, depth + 1)
        return node

    def _delete(self, node, depth, instance):
        if node.is_leaf():
            node.remove_instance(instance)
            return node

        if self.mode == "random":
            node = self._random_node_delete(node, depth, instance)
        else:
            node = self._greedy_node_delete(node, depth, instance)

        attr, threshold = node.split_attr, node.split_value
        if instance[attr] <= threshold:
            node.left = self._delete(node.left, depth + 1, instance)
        else:
            node.right = self._delete(node.right, depth + 1, instance)

        return node

    def _random_node(self, data):
        node = TreeNode()
        node.split_attr = random.choice(range(data.shape[1] - 1))
        node.split_value = random.uniform(
            np.min(data[:, node.split_attr]), np.max(data[:, node.split_attr])
        )
        return node

    def _greedy_node(self, data):
        node = TreeNode()
        best_score = float("inf")
        best_attr, best_value = None, None

        for attr in range(data.shape[1] - 1):
            thresholds = np.unique(data[:, attr])
            for value in thresholds:
                score = self._compute_split_score(data, attr, value)
                if score < best_score:
                    best_score, best_attr, best_value = score, attr, value

        if best_attr is None or best_value is None:
            return self._leaf_node(data)  # No valid split found, return a leaf node

        node.split_attr, node.split_value = best_attr, best_value
        return node

    def _random_node_delete(self, node, depth, instance):
        data = self._get_data_from_leaves(node)
        data = np.delete(data, np.where(np.all(data == instance, axis=1)), axis=0)
        return self._train(data, depth)

    def _greedy_node_delete(self, node, depth, instance):
        data = self._get_data_from_leaves(node)
        data = np.delete(data, np.where(np.all(data == instance, axis=1)), axis=0)
        return self._train(data, depth)

    def _leaf_node(self, data):
        node = TreeNode()
        node.is_leaf_node = True
        node.value = np.mean(data[:, -1])
        node.instances = data
        return node

    def _split_data(self, node, data):
        if node.split_attr is None or node.split_value is None:
            return data, np.empty((0, data.shape[1]))  # Send all data to one branch

        left_mask = data[:, node.split_attr] <= node.split_value
        right_mask = ~left_mask
        return data[left_mask], data[right_mask]

    def _compute_split_score(self, data, attr, value):
        left = data[data[:, attr] <= value]
        right = data[data[:, attr] > value]
        return len(left) * np.var(left[:, -1]) + len(right) * np.var(right[:, -1])

    def _stopping_criteria(self, data, depth):
        return len(data) < 2 or depth >= self.max_depth

    def _predict_instance(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.split_attr] <= node.split_value:
            return self._predict_instance(node.left, x)
        else:
            return self._predict_instance(node.right, x)

    def _get_data_from_leaves(self, node):
        if node.is_leaf():
            return node.instances
        return np.vstack(
            (
                self._get_data_from_leaves(node.left),
                self._get_data_from_leaves(node.right),
            )
        )
