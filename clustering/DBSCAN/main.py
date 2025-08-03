"""
A numpy implementation of the Density Based Spatial Clustering
of Applications with Noise (DBSCAN) algorithm.

1. DBSCAN is based on the notion of density. Clusters are formed
   where points are dense and sparse points are considered as noise/outlier.
   Unlike k-means which does not have notion of noise/outliers.

2. DBSCAN is specially useful when data contains noise which
   is the case in real life applications. It has several advantages
   over k-means clustering. a) can find arbitrarily shape clusters. (non-convex)
   b) Robust to noise and outliers.

3. DBSCAN has two important parameters eps and min_samples.
   eps: Maximum distance between two points to be considered as
        neighbours.
   min_samples: Minimum number of data points required to form a
                dense region. i.e., to be considered as core point.

4.  a) Core point: a point is considered as core point if it has
      min_samples within eps distance.
    b) Border point: a point which is not a core point but lies
       within eps of core point.
    c) Noise point: a point which is neither Core point nor Border point.
"""

import numpy as np
import argparse
from collections import deque
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors


class DBSCAN:
    def __init__(self, min_samples=5, eps=None):
        self.min_samples = min_samples
        self.eps = eps
        self.labels_ = None  # Cluster labels for each data point.

    def fit(self, data):

        # Find the optimal epsilon for the data using
        # K-distance graph. (Finding elbow/knee)
        self.eps = self.find_optimal_eps(data, self.min_samples)

        # Helper variables.
        N, _ = data.shape
        self.labels_ = np.full(N, -1)
        cluster_id = 0
        visited = np.zeros(N, dtype=bool)

        # Loop over all data points.
        for idx in range(N):
            point = data[idx]

            # Check if the current idx is already visited.
            if visited[idx]:
                continue

            # Get neighbours of the current point.
            neighbours = self.get_neighbours(point, data)

            # Check if this point is a core point.
            if len(neighbours) >= self.min_samples:
                # Core Point --> Expand cluster.
                self.expand_clusters(data, idx, neighbours, cluster_id, visited)

            # Increment cluster-id
            cluster_id += 1

    @staticmethod
    def find_optimal_eps(X, min_samples):
        nbrs = NearestNeighbors(n_neighbors=min_samples - 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        k_distances = np.sort(distances[:, -1])

        # Locate knee
        knee = KneeLocator(
            x=range(len(k_distances)),
            y=k_distances,
            curve='convex',
            direction='increasing'
        )

        # Relaxing the knee point bit (taking point after knee)
        return 1.5 * knee.knee_y

    def get_neighbours(self, point, data):
        return np.where(np.linalg.norm(data - point, axis=1) < self.eps)[0]

    def expand_clusters(self, data, idx, neighbours, cluster_id, visited):
        # Assign current point to the cluster id.
        self.labels_[idx] = cluster_id

        # Current point is visited.
        visited[idx] = True

        # Keep on expanding neighbours until all core points are explored.
        queue = deque(neighbours)

        while queue:
            neighbour_idx = queue.popleft()
            if not visited[neighbour_idx]:
                # This neighbour point is visited.
                visited[neighbour_idx] = True
                # Check neighbour's neighbour. (chain reaction)
                neighbours_neighbour = self.get_neighbours(data[neighbour_idx], data)
                # Extend only if it is a core point.
                if len(neighbours_neighbour) >= self.min_samples:
                    queue.extend(neighbours_neighbour)

            # Assign cluster id.
            self.labels_[neighbour_idx] = cluster_id


if __name__ == "__main__":
    # Setup seed.
    np.random.seed(42)

    # Provide configs path as input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=None)
    parser.add_argument('--min_samples', type=int, default=5)

    # Get configs Path.
    args = parser.parse_args()

    # Run
    # Sample dataset
    X, _ = make_blobs(
        n_samples=600,
        centers=5,  # 5 clusters
        cluster_std=0.80,  # spread
        random_state=42
    )
    # X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    # Apply custom DBSCAN
    db = DBSCAN(min_samples=5, eps=args.eps)
    db.fit(X)

    # Visualize
    plt.scatter(X[:, 0], X[:, 1], c=db.labels_, cmap='rainbow', s=30)
    plt.title("DBSCAN Clustering")
    plt.show()

    # # Save figure
    # plt.savefig(f"cluster.png")

    # All done
    print("\nTraining complete.")
