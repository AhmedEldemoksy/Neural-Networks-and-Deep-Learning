# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:36:05 2025

@author: menas
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_colors = 100
input_dim = 3  # RGB
grid_size = 10
learning_rate = 0.5
sigma = 1.5
num_iterations = 1000

# 1. Generate random RGB data
data = np.random.rand(num_colors, input_dim)

# 2. Initialize SOM weights randomly
weights = np.random.rand(grid_size, grid_size, input_dim)

# ---- Plot Initial SOM Grid ----
def plot_som_grid(weights, title):
    plt.figure(figsize=(6, 6))
    for i in range(grid_size):
        for j in range(grid_size):
            color = weights[i, j]
            plt.fill_between([j, j + 1], [i, i], [i + 1, i + 1], color=color)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.title(title)
    plt.show()

plot_som_grid(weights, "SOM Color Grid Before Training")

# ---- Train SOM ----
def gaussian_neighborhood(dist_sq, sigma):
    return np.exp(-dist_sq / (2 * sigma ** 2))

for t in range(num_iterations):
    x = data[np.random.randint(0, num_colors)]
    dist = np.linalg.norm(weights - x, axis=2)
    bmu_idx = np.unravel_index(np.argmin(dist), (grid_size, grid_size))
    
    # Decay learning rate and sigma
    lr = learning_rate * (1 - t / num_iterations)
    sig = sigma * (1 - t / num_iterations)
    
    for i in range(grid_size):
        for j in range(grid_size):
            dist_sq = (i - bmu_idx[0])**2 + (j - bmu_idx[1])**2
            h = gaussian_neighborhood(dist_sq, sig)
            weights[i, j] += lr * h * (x - weights[i, j])

print("Training completed!")

# ---- Plot Final SOM Grid ----
plot_som_grid(weights, "SOM Color Grid After Training")
