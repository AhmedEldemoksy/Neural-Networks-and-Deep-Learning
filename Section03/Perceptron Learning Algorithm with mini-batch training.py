import numpy as np

# Generate synthetic data (2D points, 2 classes)
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple decision boundary

# Initialize weights and bias
weights = np.random.rand(2)
bias = np.random.rand()

# Hyperparameters
learning_rate = 0.1
epochs = 75
batch_size = 20  # Mini-batch training
iterations_per_epoch = len(X) // batch_size  # Total iterations per epoch

# Training Loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # Shuffle data for mini-batch training
    indices = np.random.permutation(len(X))
    X_shuffled, y_shuffled = X[indices], y[indices]

    for i in range(iterations_per_epoch):
        start = i * batch_size
        end = start + batch_size
        X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]
        # Forward pass
        predictions = np.dot(X_batch, weights) + bias
        predictions = np.where(predictions > 0, 1, 0)  # Step activation
        # Compute error
        errors = y_batch - predictions
        # Update weights and bias using perceptron learning rule
        weights += learning_rate * np.dot(X_batch.T, errors)
        bias += learning_rate * np.sum(errors)
        print(f" Iteration {i+1}/{iterations_per_epoch}, Batch Error: {np.sum(errors)}")
print("\nFinal Weights:", weights)
print("Final Bias:", bias)

