"""
Implementation of Support Vector Machine (SVM) classifier.

* Support Vector Machine (SVM) finds the optimal decision boundary which best separates the classes
  while maximizing the margin between them.
* Margin is the distance between decision boundary (or hyperplane) and the closest points
  from the classes (also called support vectors).
* It poses the optimization problems as follows: find w, b such that max 2/norm(w) s.t y_i (w.T * x_i + b) >= 1 for all i's.
* In other words; minimize 1/2 * ||w||^2 s.t y_i (w.T * x_i + b) >= 1 for all i's.

In real-world data, the classes are not perfectly separable and hence you don't find such optimal
boundary. To cop-up with that Soft-margin SVM was introduced.

* Soft-margin SVM allows for some mis-classification using a slack variable epsilon for every point
  which penalize the misclassified points.

* Dual form of the primal optimization problem (using lagrangian multipliers) allows the SVM to use
  kernel functions since dual forms has dot product of <x.t , x>, which can be replaced with any non-
  linear kernels.

* SVM can also find non-linear decision boundaries using kernel functions, which implicitly transforms
  the data into high dimensions and find the linear hyperplane in high dim space. which can corresponds
  to non-linear decision boundary in low-dim input space.
"""

if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.datasets import make_circles

    # Generate non-linear toy dataset
    X, y = make_circles(n_samples=300, factor=0.4, noise=0.1, random_state=42)

    # Create SVM classifiers with different kernels
    models = [
        ('Linear', svm.SVC(kernel='linear', C=1)),
        ('RBF (Gaussian)', svm.SVC(kernel='rbf', gamma=2, C=1)),
        ('Polynomial (degree 3)', svm.SVC(kernel='poly', degree=3, C=1))
    ]

    # Fit all models
    for _, model in models:
        model.fit(X, y)

    # Plotting helper
    def plot_decision_boundary(model, title):
        h = 0.01
        x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
        y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.title(title)
        plt.axis('off')


    # Plot all three models
    plt.figure(figsize=(15, 4))
    for i, (name, model) in enumerate(models, 1):
        plt.subplot(1, 3, i)
        plot_decision_boundary(model, name)

    plt.tight_layout()
    plt.show()
