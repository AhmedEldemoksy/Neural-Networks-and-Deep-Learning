import numpy as np
# Define inputs and weights
inputs = np.array([1, 0]) # Example input
weights = np.array([0.5, -0.5]) # Initial weights
bias = 0.2 # Bias term
# Compute weighted sum
weighted_sum = np.dot(inputs, weights) + bias
# Apply activation function (Step function)
output = 1 if weighted_sum > 0 else 0
print(f"Perceptron Output: {output}")