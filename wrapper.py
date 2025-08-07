"""
A unified wrapper function for all the algorithms.
"""
import numpy as np

# Import Clustering algorithms.
from clustering.Kmeans.main import kmeans
from clustering.DBSCAN.main import DBSCAN
from clustering.GMM.main import GaussianMixtureModel

# Import Dim Reduction algorithms.
from dim_reduction.LDA import LDA
from dim_reduction.tSNE import TSNE


def run(data, method):
    if method == "kmeans":
        centroids, cluster_idxs = kmeans(data, k=3)
        # Save cluster labels
        np.savetxt("output/kmeans_labels.csv", cluster_idxs, fmt="%d", delimiter=",")
        # Plot


if __name__ == "__main__":
    pass
