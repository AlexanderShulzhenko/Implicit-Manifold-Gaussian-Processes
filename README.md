# Graph-Gaussian-Processes-Experiments
This repo provides complete model based on Graph Gaussian Processes that are integrated in GPFlow.
# Installation
# Library usage
Quick guide on how to use this library:
```python
>>> from graph_gaussian_process.graph_gaussian_process_model_regression import GraphGPR
>>> train_size = 500
>>> train_ind = np.random.choice(len(X), train_size, replace=False) # X is data inputs
>>> model = GraphGPR()
>>> mean, cov = model.fit_and_predict(X,train_ind,ys) # ys correspond to X[train_ind]
```
For more detailed explanation check exaple notebooks.
# Examples
In section "Experiments" we provide notebooks for regression problems and classification problems.
