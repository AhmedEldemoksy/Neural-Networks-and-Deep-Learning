# Import the necessary library
import numpy as np
from sklearn.datasets import make_blobs #function
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #function 80% for training 20% for testing
from sklearn.preprocessing import StandardScaler #class // mean=0 SD=1
from sklearn.linear_model import Perceptron #class

# Generate a linearly separable dataset with two classes
X, y = make_blobs(n_samples=1000,
    n_features=2,
    centers=2,
    cluster_std=3,
    random_state=23)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
    y,
    test_size=0.2,
    random_state=23,
    shuffle=True
    )

# Scale the input features to have zero mean and unit variance mean=0 , SD=1
scaler = StandardScaler()  #Cons
X_train = scaler.fit_transform(X_train) #Fit , Tras
X_test = scaler.transform(X_test) # traf

# Set the random seed legacy
np.random.seed(23)

# Initialize the Perceptron with the appropriate number of inputs
perceptron = Perceptron(max_iter=100, random_state=23)

# Train the Perceptron on the training data
perceptron.fit(X_train, y_train) # training input, output

# Testing Prediction
pred = perceptron.predict(X_test)

# Test the accuracy of the trained Perceptron on the testing data
accuracy = np.mean(pred == y_test)
print("Accuracy:", accuracy)

# Plot the dataset
plt.scatter(X_test[:, 0], X_test[:, 1], c=pred)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
