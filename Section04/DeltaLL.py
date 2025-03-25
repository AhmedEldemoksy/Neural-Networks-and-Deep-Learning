# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:59:20 2025

@author: menas
"""

import numpy as np
x = np.array([[1, -2, 0, -1],
              [0, 1.5, -0.5, -1],
              [-1, 1, 0.5, -1]])

d = np.array([[-1],
              [-1],
              [1]])

w = np.array([1, -1, 0, 0.5])

c = 0.1

epochno = 30

# calculateing output
def calculate_output(weights, instance):
    sum = np.dot(instance, w)
    return np.round(np.tanh(sum))

for j in range(epochno):
  print(["epoch"+str(j)])
  for i in range(3):
      net = np.dot(x[i, :], w)
      fnet = np.tanh(net) 
      fnetd = 1 - np.tanh(net) ** 2 
      e = d[i, 0] - fnet
      deltaw = c * e * fnetd * x[i, :]

      w = w + deltaw
  print(w)
  print('prediction for [1, -2, 0, -1]: ' + str(calculate_output(w, np.array([[1, -2, 0, -1]]))))
  print('prediction for [0, 1.5, -0.5, -1]: ' + str(calculate_output(w, np.array([[0, 1.5, -0.5, -1]]))))
  print('prediction for [-1, 1, 0.5, -1]: ' + str(calculate_output(w, np.array([[-1, 1, 0.5, -1]]))))

