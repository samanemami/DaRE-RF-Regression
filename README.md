# Regression Variant of [Original Model Name]

## Overview
This repository contains a regression-based variant of the original model developed by [Jonathan Brophy and Daniel Lowd](https://github.com/jjbrophy47/dare_rf) 
as described in [Machine Unlearning for Random Forests](https://proceedings.mlr.press/v139/brophy21a/brophy21a.pdf). 
The main objective of this variant is to adapt the original architecture for regression tasks that require continuous output prediction. 
The original model, designed for classification tasks, has been modified to support regression problems by changing the loss function and modifying the related methods.

## Key Features
- **Regression-based architecture**: Modified the original classification DaRE RF architecture to handle regression tasks.


# Acknowledgments

I would like to thank [Machine Unlearning for Random Forests](https://proceedings.mlr.press/v139/brophy21a/brophy21a.pdf) for their groundbreaking work on [DaRE], which served as the foundation for this project.

# Citation

If you use this work or the original model in your research, please consider citing the original paper.

""" bibtex
@InProceedings{pmlr-v139-brophy21a,
  title = 	 {Machine Unlearning for Random Forests},
  author =       {Brophy, Jonathan and Lowd, Daniel},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {1092--1104},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/brophy21a/brophy21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/brophy21a.html},
  abstract = 	 {Responding to user data deletion requests, removing noisy examples, or deleting corrupted training data are just a few reasons for wanting to delete instances from a machine learning (ML) model. However, efficiently removing this data from an ML model is generally difficult. In this paper, we introduce data removal-enabled (DaRE) forests, a variant of random forests that enables the removal of training data with minimal retraining. Model updates for each DaRE tree in the forest are exact, meaning that removing instances from a DaRE model yields exactly the same model as retraining from scratch on updated data. DaRE trees use randomness and caching to make data deletion efficient. The upper levels of DaRE trees use random nodes, which choose split attributes and thresholds uniformly at random. These nodes rarely require updates because they only minimally depend on the data. At the lower levels, splits are chosen to greedily optimize a split criterion such as Gini index or mutual information. DaRE trees cache statistics at each node and training data at each leaf, so that only the necessary subtrees are updated as data is removed. For numerical attributes, greedy nodes optimize over a random subset of thresholds, so that they can maintain statistics while approximating the optimal threshold. By adjusting the number of thresholds considered for greedy nodes, and the number of random nodes, DaRE trees can trade off between more accurate predictions and more efficient updates. In experiments on 13 real-world datasets and one synthetic dataset, we find DaRE forests delete data orders of magnitude faster than retraining from scratch while sacrificing little to no predictive power.}
}
"""

