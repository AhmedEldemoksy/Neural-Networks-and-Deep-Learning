import numpy as np

# Step 1: Training Data for AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# Step 2: Initialize parameters
weights = np.zeros(2)
bias = 0
learning_rate = 0.1
epochs = 10

# Step 3: Activation function (Step Function)
def step_function(z):
    return 1 if z >= 0 else 0

# Step 4: Training
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step_function(linear_output)
        
        error = y[i] - prediction
        
        # Update rule
        weights += learning_rate * error * X[i]
        bias += learning_rate * error
        
        print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {prediction}, Error: {error}")

print("\nTraining Complete!")
print("Final Weights:", weights)
print("Final Bias:", bias)

# Step 5: Testing
print("\nTesting:")
for i in range(len(X)):
    linear_output = np.dot(X[i], weights) + bias
    prediction = step_function(linear_output)
    print(f"Input: {X[i]} -> Output: {prediction}")