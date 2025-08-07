import argparse
import os
import shutil

import numpy as np
from matplotlib import pyplot as plt
import subprocess


def kmeans_pp_init(data, k):
    """
    Kmeans ++ initialization. Initializes centroids which are far
    apart from each other.
    """
    N, dim = data.shape
    centroids = np.zeros((k, dim))

    # Step-1: Pick a random sample from the data and assign as centroid.
    centroids[0] = data[np.random.randint(N)]

    min_dists = None
    for idx in range(k-1):
        # Step-2: Calculate distance based pdf for the next centroid pick.
        if min_dists is None:
            dists = np.linalg.norm(data - centroids[idx], axis=1)
            min_dists = dists + 0.
        else:
            min_dists = np.minimum(min_dists, np.linalg.norm(data - centroids[idx], axis=1))

        dists_pdf = min_dists/min_dists.sum()

        # Select next centroid based on this pdf.
        next_idx = np.random.choice(len(dists_pdf), p=dists_pdf)
        centroids[idx + 1] = data[next_idx]

    return centroids


def assign_centroids(data, centroids):
    """
    Assign centroids to the datapoints. Given data and centroids,
    returns a list of indices indicating belonging of datapoints to
    the clusters.
    """
    N, _ = data.shape
    k, _ = centroids.shape
    distances = np.zeros((N, k))
    for idx, centroid in enumerate(centroids):
        distances[:, idx] = np.linalg.norm((data - centroid), axis=1)

    return np.argmin(distances, axis=1)


def update_centroids(data, centroids, cluster_idxs):
    """
    Update centroids based on the cluster indices.
    """
    k, _ = centroids.shape
    for idx in range(k):
        # If a centroid doesn't have any assign datapoint; don't update.
        if len(data[cluster_idxs == idx]) != 0:
            centroids[idx] = data[cluster_idxs == idx].mean(axis=0)

    return centroids


def visualize_clusters(data, centroids_tracker):

    # Colours for centroids.
    colors = ["red", "green", "blue"]
    centroid_labels = ["Centroid-1", "Centroid-2", "Centroid-3"]

    # Create a folder for saving plot images.
    if os.path.exists("images"):
        # Remove
        shutil.rmtree("images")

    # Create
    os.makedirs("images")

    for idx, centroid in enumerate(centroids_tracker):
        # Create a figure
        plt.figure(figsize=(6, 6))
        # Plot 2D data
        plt.scatter(data[:, 0], data[:, 1], c="brown", alpha=0.5, label="Data points")
        # Plot centroids
        for i in range(3):
            plt.scatter(centroid[i, 0], centroid[i, 1],
                        c=colors[i], edgecolor='black', marker='X',
                        s=100, label=centroid_labels[i])

        # Formatting the plot.
        plt.title(f"K-means Clustering | Iteration: {idx}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()

        # Save plot
        plt.savefig(f"images/{idx}.png")
        plt.close()

    # Create a video using ffmpeg.
    command = [
        "ffmpeg",
        "-y",  # overwrite output file without asking
        "-framerate", "1",
        "-i", os.path.join("images", "%d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "kmeans_pp.mp4"
    ]
    subprocess.run(command, check=True)


def kmeans(data, k=3, eps=1e-4, max_iter=1000,
           visualize=False, pp_init=True):
    """
    A numpy implementation of the algorithm K-means clustering.
    :param data: Data points
    :param k: number of clusters
    :param eps: stopping criterion
    :param max_iter: Maximum number of iterations before stopping.
    :param visualize: Visualize a convergence GIF for centroids.
    :param pp_init: To use kmeans++ initialization for centroids or not.

    :return: centroids: List of centroids
    :return: cluster_idxs:  List of cluster assignments for each data-point.
    """
    # Step-1: Initialization of the K-centroids.
    N, dim = data.shape
    centroids = np.random.randn(k, dim) if not pp_init else kmeans_pp_init(data, k)
    cluster_idxs = None

    # Track centroids for later visualization.
    centroids_ = centroids + 0.
    centroids_tracker = [centroids_]

    # Repeat
    for _ in range(max_iter):
        # Step-2: Assign points to the centroids.
        cluster_idxs = assign_centroids(data, centroids)

        # Step-3: Update the Centroids.
        centroids_prev = centroids + 0.
        centroids = update_centroids(data, centroids, cluster_idxs)
        centroids_tracker.append(centroids + 0.)  # For tracking

        # Step-4: Check for the Stopping Criterion.
        # If Max centroid movements between two consecutive iterations
        # is less than epsilon.
        if max(np.linalg.norm(centroids - centroids_prev, axis=1)) <= eps:
            break

    # Create a video
    if visualize:
        try:
            visualize_clusters(data, centroids_tracker)
        except:
            print(f"Visualization is only implemented for k=3 and 2-D data.")

    return centroids, cluster_idxs


if __name__ == "__main__":
    # Setup seed.
    np.random.seed(42)

    # Provide configs path as input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--epsilon', type=float, default=0.0001)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--pp_init', type=bool, default=True,
                        help="use kmeans++ initialization or random")

    # Get configs Path.
    args = parser.parse_args()

    # Generate 3 clusters
    cluster1 = np.random.randn(100, 2) * 0.5 + np.array([5, 5])
    cluster2 = np.random.randn(100, 2) * 0.5 + np.array([-5, -5])
    cluster3 = np.random.randn(100, 2) * 0.5 + np.array([5, -5])

    # Combine into one dataset
    data = np.vstack([cluster1, cluster2, cluster3])

    centroids, cluster_idxs = kmeans(data, k=args.k, eps=args.epsilon,
                                     max_iter=args.max_iterations, pp_init=args.pp_init
                                     )

    # All done
    print("\nTraining complete.")
