"""
A numpy implementation of the Linear Discriminant Algorithm (LDA).

Core Idea:
==========
* LDA is used for reducing dimensionality of the features. LDA projects
  high-D features into low-D features while maximizing the class separability.

* In other words, it maximizes the variance between classes and minimizes the
  variance within class. (Fisher's criterion)

* Note that it is a supervised technique since it requires labels unlike PCA.

* It assumes that all the classes follows the gaussian distribution and
  have same covariance.

* It can also be used as classifier.

Cons:
======
* Assumption of Gaussianity and same covariance.
* Not good for if classes are not linearly separable.

"""

import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eigh


class LDA:
    def __init__(self, num_classes, out_dim=2):
        self.num_classes = num_classes
        self.out_dim = out_dim

    def fit(self, data, labels):

        N, dim = data.shape

        # Compute Local mean and global mean.
        mu_local = np.array([data[labels == idx].mean(axis=0) for idx in range(self.num_classes)])
        mu_global = data.mean(axis=0)

        # Compute within class scatter matrix (S_w)
        s_w = np.zeros((dim, dim))
        for cls in range(self.num_classes):
            diff = data[labels == cls] - mu_local[cls]
            s_w += diff.T @ diff

        # Compute between class scatter matrix (S_b)
        s_b = np.zeros((dim, dim))
        for cls in range(self.num_classes):
            n_c = len(data[labels == cls])
            diff = mu_local[cls] - mu_global
            s_b += n_c * np.outer(diff, diff)

        #  Rayleigh quotient. solution => eigen values of inverse(s_w) * s_b
        s_w += 1e-6 * np.eye(s_w.shape[0])
        eigen_vals, eigen_vecs = eigh(s_b, s_w)

        # Sort and retain max out_dim eigen vals.
        sorted_indices = np.argsort(eigen_vals)[::-1]
        principal_eig_vecs = eigen_vecs[:, sorted_indices[:self.out_dim]]

        # Project to lower dimension.
        return data @ principal_eig_vecs


if __name__ == "__main__":
    # Setup seed.
    np.random.seed(42)

    # Provide configs path as input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dim', type=int, default=2)

    # Get configs Path.
    args = parser.parse_args()

    # Data.
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    # Call LDA
    lda = LDA(num_classes=3, out_dim=args.out_dim)
    X_lda = lda.fit(X, y)

    # 5. Plot the result
    # Plot the result
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1],
                    label=iris.target_names[label], alpha=0.7, s=50)

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Projection of Iris Dataset (2D)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nTraining complete.")

    # All done
    print("\nTraining complete.")
