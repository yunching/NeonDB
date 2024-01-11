import numpy as np
from sklearn.covariance import LedoitWolf

# Assume we have some returns data in a 2D numpy array
returns = np.array([[0.01, 0.02, -0.01],
                    [-0.01, 0.01, 0.02],
                    [0.02, -0.01, 0.01]])

# Estimate the covariance matrix using Ledoit-Wolf shrinkage estimator
lw = LedoitWolf()
lw.fit(returns)
cov_matrix = lw.covariance_

print(cov_matrix)

def calculate_risk_contribution(weights, cov_matrix):
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    security_risk_contributions = np.dot(cov_matrix, weights) / portfolio_variance
    return security_risk_contributions

# Example usage
weights = np.array([0.25, 0.35, 0.4])  # Example weights
cov_matrix = np.array([[0.04, 0.02, 0.01],
                       [0.02, 0.09, 0.03],
                       [0.01, 0.03, 0.06]])  # Example covariance matrix

risk_contributions = calculate_risk_contribution(weights, cov_matrix)
print(risk_contributions)