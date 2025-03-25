import numpy as np

# Step function activation
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Perceptron function
def perceptron(weights, inputs, bias):
    return step_function(np.dot(inputs, weights) + bias)

# Update rule for perceptron learning
def perceptron_update(weights, inputs, bias, learning_rate, target, output):
    error = target - output
    weights += learning_rate * error * inputs  # Adjust weights
    bias += learning_rate * error  # Adjust bias
    return weights, bias

# XOR Perceptron Training
def train_xor_perceptron(epochs=10, learning_rate=0.1):
    # Initialize perceptrons with random weights
    p1_weights = np.random.rand(2)  # OR Gate
    p1_bias = np.random.rand()

    p2_weights = np.random.rand(2)  # AND Gate
    p2_bias = np.random.rand()

    p3_weights = np.random.rand(2)  # Final XOR Gate
    p3_bias = np.random.rand()

    # XOR training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])  # Expected XOR output

    for epoch in range(epochs):
        for i in range(len(X)):
            x1, x2 = X[i]
            target = Y[i]

            # Compute intermediate perceptron outputs
            p1_output = perceptron(p1_weights, np.array([x1, x2]), p1_bias)  # OR(x1, x2)
            p2_output = perceptron(p2_weights, np.array([x1, x2]), p2_bias)  # AND(x1, x2)

            # Compute final XOR output
            p3_output = perceptron(p3_weights, np.array([p1_output, 1 - p2_output]), p3_bias)

            # Update weights for each perceptron separately
            p1_weights, p1_bias = perceptron_update(p1_weights, np.array([x1, x2]), p1_bias, learning_rate, x1 or x2, p1_output)
            p2_weights, p2_bias = perceptron_update(p2_weights, np.array([x1, x2]), p2_bias, learning_rate, x1 and x2, p2_output)
            p3_weights, p3_bias = perceptron_update(p3_weights, np.array([p1_output, 1 - p2_output]), p3_bias, learning_rate, target, p3_output)

        print(f"Epoch {epoch+1} completed")

    return p1_weights, p1_bias, p2_weights, p2_bias, p3_weights, p3_bias

# Train the XOR perceptron
p1_w, p1_b, p2_w, p2_b, p3_w, p3_b = train_xor_perceptron(epochs=20)

# Test the trained XOR perceptron
def test_xor(x1, x2):
    p1_output = perceptron(p1_w, np.array([x1, x2]), p1_b)  # OR Gate
    p2_output = perceptron(p2_w, np.array([x1, x2]), p2_b)  # AND Gate
    p3_output = perceptron(p3_w, np.array([p1_output, 1 - p2_output]), p3_b)  # XOR Gate
    return p3_output

# Print test results
print("\nXOR Gate Test Results:")
for x in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]):
    print(f"XOR({x[0]}, {x[1]}) = {test_xor(x[0], x[1])}")
