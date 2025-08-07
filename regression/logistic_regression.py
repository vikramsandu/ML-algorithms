"""
* Logistic regression is the extension of the linear regression for classification.
* This takes the linear output of regression and apply sigmoid to convert into
class probability.
* It linearly models the log-odds, not the probability.
* This should be used if relation between input-output is linear or logit-linear.
* Use if You want a probabilistic interpretation of predictions, fast , simple model.
* It is optimized using BCE-Loss (log loss) which turns out to be convex function wrt parameters
  hence convergence is guaranteed.
"""

