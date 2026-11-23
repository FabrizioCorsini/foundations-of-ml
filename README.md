# Foundations of Machine Learning: exercises

Exercises for the Foundations of Machine Learning course (Politecnico di Bari, A.Y. 2025/26): classic machine learning algorithms implemented from scratch in NumPy, then with scikit-learn and PyTorch.

## From scratch in NumPy

| File | What it implements |
|---|---|
| `linear_regression.py` | Linear regression with full-batch, mini-batch and stochastic gradient descent, with optional regularisation |
| `logistic_regression.py` | Logistic regression trained with gradient descent |
| `classification_nn.py`, `regression_nn.py` | Feed-forward neural networks with backpropagation, for classification and regression |
| `kmeans.py`, `kmedoids.py` | K-Means and K-Medoids clustering |
| `classification_metrics.py`, `regression_metrics.py`, `evaluation.py` | Confusion matrix, accuracy and error rate; MAE, MSE and RMSE |

Scripts that use them: `01_univariate.py`, `01_multivariate.py`, `02_polynomial.py`, `03_multivariate*.py`, `02_main_logistic.py`, `main_classification.py`, `main_regression.py`, and `04_normal.py` (closed-form normal equation).

## With scikit-learn

| Topic | Files |
|---|---|
| Preprocessing, grid search and pipelines | `01_preprocessing.py`, `02_grid.py`, `Sklearn_Base.py`, `Sklearn_Grid.py`, `Sklearn_Pipeline_Grid.py` |
| Information gain | `03_information_gain.py` |
| Decision trees and random forests | `decision_tree_classifier.py`, `decision_tree_classifier_grid.py`, `random_forest_classifier.py` |
| SVM, kernels, PCA and k-fold cross-validation | `SVM.py`, `main_kernel.py`, `main_pca.py`, `main_pca_kfold.py`, `main_pipeline.py` |
| Clustering: elbow method, K-Means, K-Medoids and DBSCAN | `clustering.py`, `wcss_dbscan.py` |
| Anomaly detection with a Gaussian mixture | `anomaly.py` |

## With PyTorch

- `autoencoder.py` and `main.py`: an autoencoder trained on Fashion-MNIST; an SVM is then trained both on the raw pixels and on the encoded features, to compare them.
- `PyTorch_Fundamentals_FML.ipynb`: tensors, automatic differentiation with autograd, and a neural network for binary classification on the diabetes dataset.

## Running

Python 3 with `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `torch` and `torchvision`. The scripts read their data from a `datasets/` folder, which is not included.
