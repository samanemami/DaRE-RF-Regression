import numpy as np


class TreeNode:
    def __init__(self):
        self.split_attr = None
        self.split_value = None
        self.left = None
        self.right = None
        self.is_leaf_node = False
        self.value = None
        self.instances = None

    def is_leaf(self):
        return self.is_leaf_node

    def remove_instance(self, instance):
        self.instances = np.delete(
            self.instances, np.where(np.all(self.instances == instance, axis=1)), axis=0
        )
        if len(self.instances) > 0:
            self.value = np.mean(self.instances[:, -1])
