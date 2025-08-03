"""
A numpy implementation of the t-Distributed Stochastic Neighbor Embedding (t-SNE)
algorithm.

Core Idea:
==========
* t-sne is a non-linear dimensionality reduction problem mostly used for visualizing
  high-D data in low-D (2D/3D) for human interpretation and analysis. It is used in
  visualizing images/word-embedding etc. in low-D.

* The idea is that compute pairwise similarity in high-D using Gaussian kernels and compute
  pairwise similarity in low-D using Cauchy distribution (DOF=1) and minimize the KL-Divergence (information-loss)
  between these two distribution.

* In lower-D cauchy is chosen because it takes cares of the crowding problem since it is long-tail
  distribution, it will push away dissimilar data far apart and only keeps similar data together.

* In high-D, we have sigma for each Gaussian kernel which roughly measures the spread of local
  gaussian distribution, we want to ensure that sigma should be such that this spread should remain
  same across all points. perplexity is another measure of spread and more intuitive which basically tells
  the number of effective neighbours for a point. then sigma is calculated for every point in a way that
  perplexity of every point should remain same across all point.
"""

import torch
import argparse
import numpy as np
from sklearn.manifold._t_sne import _joint_probabilities
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class TSNE:
    def __init__(self, perplexity, out_dim, num_iters=1000):
        self.perplexity = perplexity
        self.out_dim = out_dim
        self.num_iters = num_iters

    def fit(self, data):
        N, _ = data.shape

        # Step-1: Compute Joint prob. distribution of pairwise
        # distances in high-D.
        P = self.joint_prob_highD(data)

        # Step-2: Symmetrize the P + numerical stability.
        P = (P + P.T) / (2 * N)
        P = np.maximum(P, 1e-12)

        #### Switching to Pytorch for feasible optimization ###
        P = torch.from_numpy(P)

        # Step-3: Initialize low-D points. -- To be optimized later
        # through Gradient Descent
        import torch.nn as nn
        Y = nn.Parameter(torch.randn(N, self.out_dim))

        # Optimizer.
        optimizer = torch.optim.SGD([Y], lr=200.0, momentum=0.5)

        for _ in tqdm(range(self.num_iters)):
            optimizer.zero_grad()
            # Compute Joint prob. distribution of pairwise similarity
            Q = self.joint_prob_lowD(Y)
            # Compute loss
            loss = self.kl_divergence(P, Q)
            # Back-propagate loss
            loss.backward()
            # Update params.
            optimizer.step()

            if _ % 100 == 0:
                print(f"Iteration {_}: KL divergence = {loss.item():.4f}")

        return Y.detach().numpy()

    def joint_prob_highD(self, data):
        # Pairwise distances.
        distances = pairwise_distances(data, squared=True)
        # Joint prob distribution.
        P = _joint_probabilities(distances, desired_perplexity=self.perplexity, verbose=False)

        from scipy.spatial.distance import squareform
        P = squareform(P)  # Converts from condensed (1D) to full symmetric matrix
        np.fill_diagonal(P, 0.0)

        return P

    @staticmethod
    def joint_prob_lowD(Y):
        """Compute pairwise similarities (Q) using Student t-distribution."""
        sum_Y = torch.sum(Y ** 2, dim=1)
        dist = -2 * torch.mm(Y, Y.t()) + sum_Y[:, None] + sum_Y[None, :]
        dist = 1 / (1 + dist)
        dist.fill_diagonal_(0)
        Q = dist / dist.sum()
        Q = torch.clamp(Q, min=1e-12)
        return Q

    @staticmethod
    def kl_divergence(P, Q):
        # Ensure P and Q are normalized and clamped
        P = P / torch.sum(P)
        Q = Q / torch.sum(Q)

        P = torch.clamp(P, min=1e-12)
        Q = torch.clamp(Q, min=1e-12)

        return torch.sum(P * torch.log(P / Q))


if __name__ == "__main__":
    # Setup seed.
    np.random.seed(42)

    # Provide configs path as input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--out_dim', type=int, default=2)
    parser.add_argument('--num_iters', type=int, default=1000)

    # Get configs Path.
    args = parser.parse_args()

    # Define Data
    # Load MNIST (and subsample for speed)
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(int)

    # Subsample: 100 per class
    X_sub, y_sub = [], []
    for cls in np.unique(y)[0:5]:
        idx = np.where(y == cls)[0][:100]
        X_sub.append(X[idx])
        y_sub.append(y[idx])
    X = np.vstack(X_sub)
    y = np.hstack(y_sub)

    # Standardize
    X = StandardScaler().fit_transform(X)

    # Run t-SNE
    tsne = TSNE(perplexity=args.perplexity, out_dim=args.out_dim, num_iters=args.num_iters)
    Y_tsne = tsne.fit(X)

    # Visualize
    plt.figure(figsize=(10, 8))
    for label in np.unique(y):
        plt.scatter(Y_tsne[y == label, 0], Y_tsne[y == label, 1], label=str(label), alpha=0.6, s=10)
    plt.title("PyTorch t-SNE on MNIST Subsampled")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nTraining complete.")
