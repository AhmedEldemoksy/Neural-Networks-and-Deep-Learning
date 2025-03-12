import numpy as np

y_test = np.array([1, 0, 1, 1, 0, 1])   
pred = np.array([1, 0, 0, 1, 0, 1])   

comparison = pred == y_test
print("comparison resuil:", comparison)


accuracy = np.mean(comparison)
print("accurcy:", accuracy)
