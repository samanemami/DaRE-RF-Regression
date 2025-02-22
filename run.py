# %%
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from Base.dare_random_forest import DaRERandomForest
from sklearn.metrics import root_mean_squared_error as rmse

if __name__ == "__main__":
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)
    data = np.hstack((X, y.reshape(-1, 1)))

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    forest = DaRERandomForest(n_trees=10, max_depth=5, mode="greedy")
    forest.fit(train_data)

    predictions_before = forest.predict(test_data[:, :-1])
    print("RMSE before unlearning:", rmse(test_data[:, -1], predictions_before))

    instance_to_remove = train_data[0]
    forest.delete_instance(instance_to_remove)

    predictions_after = forest.predict(test_data[:, :-1])
    print("RMSE after unlearning:", rmse(test_data[:, -1], predictions_after))


