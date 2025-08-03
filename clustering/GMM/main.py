"""
A numpy implementation of the Gaussian Mixture Model (GMM) clustering
algorithm.

Core Idea:
==========
* It is like extension of k-means, where each centroid has a gaussian distribution
  and each datapoints comes from a mixture of these Gaussians.
* GMM works on the assumption that the data points are coming from mixture of
  Gaussians.
* For a given data point, instead of hard cluster labels, it assigns a soft
  probability for each cluster.
* It could be helpful, if the clusters are of elliptical shape. Can be clustered using
  anisotropic gaussians.
* Can also generate new samples (unlike k means, and dbscan) by sampling from underlying
  GMM.

Cons:
======
* Assumption of Gaussianity.
* Requires K.
* Do not handle noise like dbscan.

Implementation:
===============
Step-1: Initialize K.
Step-2:
  1) Initialize mean: Run kmeans and find centroids.
  2) Initialize covariance: Identity Matrix
  3) Initialize weight: 1/k (equal weights)
Step-3:
  1) Expectation Step: Compute responsibility for each data point. probability that each data points
    are coming from a particular cluster. (k values)
  2) Maximization Step: Update means, covariance, and weights based on this responsibility value.
Step-4: Compute log-likelihood that data points are coming from underlying GMM.
Step-5: Repeat Step-3 and 4 until log-likelihood converges.
"""

import argparse
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

# Import kmeans
import sys

sys.path.append('../')
from Kmeans.main import kmeans
from tqdm import tqdm
from matplotlib.patches import Ellipse
from sklearn.datasets import make_classification


# Visualize
def plot_gmm_results(data, gmm, responsibilities=None):
    plt.figure(figsize=(8, 8))

    # Plot data points
    if responsibilities is None:
        plt.scatter(data[:, 0], data[:, 1], s=5)
    else:
        labels = np.argmax(responsibilities, axis=1)
        plt.scatter(data[:, 0], data[:, 1], c=labels, s=5, cmap='tab10')

    # Plot Gaussian ellipses
    for i in range(gmm.k):
        mean = gmm.means[i]
        cov = gmm.covariances[i]

        # Correct eigen-decomposition
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * np.sqrt(vals)

        ellip = Ellipse(xy=mean, width=width, height=height, angle=angle,
                        edgecolor='black', facecolor='none', linewidth=2)
        plt.gca().add_patch(ellip)
        plt.plot(*mean, 'kx', markersize=10, markeredgewidth=2)

    plt.title("GMM Clustering Result")
    plt.axis('equal')
    plt.grid(True)
    plt.show()


class GaussianMixtureModel:
    def __init__(self, k, eps=1e-4, max_iterations=1000):
        self.k = k
        self.eps = eps
        self.max_iterations = max_iterations
        self.means = None
        self.covariances = None
        self.weights = None

    def initialize_gmm(self, data):
        N, dim = data.shape
        # Step-1: Initialize Gaussians mean, covariances, and weights for each gaussian.
        self.means, _ = kmeans(data, self.k, visualize=False)
        self.covariances = np.array([np.identity(dim) for _ in range(self.k)])
        self.weights = 1 / self.k * np.ones(self.k)

    def E_step(self, data):
        N, dim = data.shape
        responsibilty_matrix = np.zeros((N, self.k))
        for idx_1 in range(N):
            for idx_2 in range(self.k):
                responsibilty_matrix[idx_1, idx_2] = self.weights[idx_2] * multivariate_normal.pdf(data[idx_1],
                                                                                                   self.means[idx_2],
                                                                                                   self.covariances[idx_2]
                                                                                                   )
        responsibilty_matrix /= responsibilty_matrix.sum(axis=1)[:, np.newaxis]

        return responsibilty_matrix

    def M_step(self, data, responsibilty_matrix):
        N, dim = data.shape
        self.weights = 1 / N * responsibilty_matrix.sum(axis=0)
        self.means = (responsibilty_matrix.T @ data) / responsibilty_matrix.sum(axis=0)[:, np.newaxis]
        for idx in range(self.k):
            diff = data - self.means[idx]  # shape: (N, D)
            weighted_cov = np.einsum("ni,nj->ij", diff, diff * responsibilty_matrix[:, idx][:, np.newaxis])
            self.covariances[idx] = weighted_cov / responsibilty_matrix[:, idx].sum()

    def fit(self, data):
        # Helpers
        N, dim = data.shape
        prev_log_likelihood = -np.inf

        # Step-1: Initialize Gaussians mean, covariances, and weights for each gaussian.
        self.initialize_gmm(data)

        for _ in tqdm(range(self.max_iterations)):
            # Step-2: E-step: Compute responsibility of each Gaussian for each data point.
            # (N, k) matrix
            responsibilty_matrix = self.E_step(data)

            # Step-3: M-step: update mean, covariances, and weights.
            self.M_step(data, responsibilty_matrix)

            # Step-4: Check for stopping criterion.
            log_likelihood = self.compute_log_likelihood(data)
            if np.abs(log_likelihood - prev_log_likelihood) <= self.eps:
                break
            # Replace
            prev_log_likelihood = log_likelihood

    def compute_log_likelihood(self, data):
        log_likelihood = 0.0

        for point in data:
            prob = 0.0
            for idx in range(self.k):
                prob += self.weights[idx] * multivariate_normal.pdf(point, self.means[idx], self.covariances[idx])

            log_likelihood += np.log(prob)

        return log_likelihood


if __name__ == "__main__":
    # Setup seed.
    np.random.seed(42)

    # Provide configs path as input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3)

    # Get configs Path.
    args = parser.parse_args()

    # Data
    from sklearn.datasets import make_blobs

    data, y = make_blobs(
            n_samples=300,
            centers=3,
            cluster_std=[2.5, 0.6, 4.2],  # Larger std dev = more overlap
            random_state=42
        )

    # Run GMM
    gmm = GaussianMixtureModel(k=args.k)
    gmm.fit(data)

    # Visualize
    plot_gmm_results(data, gmm, responsibilities=gmm.E_step(data))
    # All done
    print("\nTraining complete.")
