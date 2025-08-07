"""
We'll explore linear regression, and it's variant in this notebook. and also see when and where to use this.
It's maths, use cases, drawback, limitation. etc.

Linear Regression: Given a set of independent features and a dependent variable y, find the best fit
                      line (in 2D) and hyperplane (in higher dims). eq. y = Wx + epsilon

* Minimizes Sum of mean squared error between gt and predicted. find w such that minimize(Xw - y)**2
   for which the solution is w = (X.T @ X)^-1 X.T * y. Note that M = X.T @ X is positive semi-definite matrix.
   Vanilla Linear regression is good if features are low dims.
   
* As feature dimension increases, There's a chance of multicolinearity -- one or more features are linear combination
   of other features. In that case some of the eigen values of M can go to zero and M^-1 may blow up and introduce
   numerical stability and overfitting (small change in input can cause large change in output i.e. high variance)
   
Ridge Regression (L2): To tackle the situation of multicolinearity and overfitting Ridge regression was introduced.

* Ridge Regression solves the problem of minimize(Xw - y)**2 with constraint that L2-norm(w) = 0
  which is minimize (Xw-y)^2 + lambda * 2norm(w) and the solution of this is w = (X.T @ X + lambda)^-1 X.T * y
  
* This lambda will make sure that eigen value never becomes zero and solution doesn't blow up.

* Hence it reduces the overfitting (or variance) but may slightly increases the bias.

Lasso Regression (L1):Lasso Regression solves the problem of minimize(Xw - y)**2 with constraint that L1-norm(w) = 0

* This is used when you need sparse solution (many weights are zero.)
* Good for interpretability (which features are important)
* This will greatly increase the bias and highly reduce the variance.
 
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # 🎲 1. Create synthetic data with multicollinearity
    np.random.seed(0)
    n_samples = 100
    X_base = np.random.randn(n_samples, 1)
    X = np.hstack([
        X_base,
        X_base + 0.01 * np.random.randn(n_samples, 1),  # highly correlated
        np.random.randn(n_samples, 1)  # independent noise
    ])
    true_coef = np.array([3, 3, 0.5])  # ground truth weights

    y = X @ true_coef + 0.5 * np.random.randn(n_samples)  # add noise

    # 🔀 2. Train/test split and scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 📊 3. Fit models
    models = {
        'Linear': LinearRegression(),
        'Ridge (α=1)': Ridge(alpha=1),
        'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=10000)
    }

    coefs = {}
    train_scores = {}
    test_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        coefs[name] = model.coef_
        train_scores[name] = model.score(X_train, y_train)
        test_scores[name] = model.score(X_test, y_test)

    # 🖼️ 4. Plot coefficient values
    plt.figure(figsize=(10, 5))
    bar_width = 0.2
    indices = np.arange(X.shape[1])

    for i, (name, coef) in enumerate(coefs.items()):
        plt.bar(indices + i * bar_width, coef, bar_width, label=name)

    plt.xticks(indices + bar_width, [f'Feature {i}' for i in range(X.shape[1])])
    plt.ylabel("Coefficient Value")
    plt.title("Coefficient Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 📈 5. Print performance
    print("Model Performance:")
    for name in models:
        print(f"{name}: Train R² = {train_scores[name]:.3f}, Test R² = {test_scores[name]:.3f}")