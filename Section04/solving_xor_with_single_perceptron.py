import numpy as np

# Define the step function (activation function for perceptrons)
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Define the perceptron
def perceptron(weights, inputs, bias):
    return step_function(np.dot(inputs, weights) + bias)

# Define the perceptron update rule
def perceptron_update(weights, inputs, bias, learning_rate, target, output):
    error = target - output

    weights = weights.astype(np.float64)
    weights += learning_rate * error * inputs
    bias += learning_rate * error
    return weights, bias

# Define the XOR problem decomposition using multiple perceptrons
def xor_using_perceptrons(x1, x2, learning_rate=0.1, epochs=1):
    # Initialize weights and biases for each perceptron
    p1_weights = np.array([1, 1])
    p1_bias = 1
    
    p2_weights = np.array([1, 1])
    p2_bias = 1
    
    p3_weights = np.array([1])
    p3_bias = 1
    
    p4_weights = np.array([1, 1])
    p4_bias = 1
    
    # Expected output for XOR
    target = (x1 ^ x2)  # XOR operation

    # Training for a given number of epochs
    for epoch in range(epochs):
        # Compute outputs for P1 (x1 OR x2)
        p1_output = perceptron(p1_weights, np.array([x1, x2]), p1_bias)
        
        # Compute outputs for P2 (x1 AND x2)
        p2_output = perceptron(p2_weights, np.array([x1, x2]), p2_bias)
        
        # Compute outputs for P3 (NOT (x1 AND x2))
        p3_output = perceptron(p3_weights, np.array([p2_output]), p3_bias)
        
        # Compute final output for P4 ((x1 OR x2) AND NOT (x1 AND x2))
        p4_output = perceptron(p4_weights, np.array([p1_output, p3_output]), p4_bias)
        
        # Update weights and biases using the perceptron learning rule
        p1_weights, p1_bias = perceptron_update(p1_weights, np.array([x1, x2]), p1_bias, 0.1, target, p1_output)
        p2_weights, p2_bias = perceptron_update(p2_weights, np.array([x1, x2]), p2_bias, 0.1, target, p2_output)
        p3_weights, p3_bias = perceptron_update(p3_weights, np.array([p2_output]), p3_bias, 0.1, target, p3_output)
        p4_weights, p4_bias = perceptron_update(p4_weights, np.array([p1_output, p3_output]), p4_bias, 0.1, target, p4_output)
        
        # Print epoch result
        print(f"Epoch {epoch+1}: Input=({x1}, {x2}), Predicted XOR={p4_output}, Target={target}")

    return p4_output

# Test the network for all input combinations and run for 10 epochs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

for x in inputs:
    xor_using_perceptrons(x[0], x[1], epochs=5)
