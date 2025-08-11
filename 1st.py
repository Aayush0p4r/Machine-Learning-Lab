"""Matrix Operations Example"""

import numpy as np

A = np.array([[1, 2], [3, 4]])

B = np.array([[5, 6], [7, 8]])

addition = A + B

elementwise_multiplication = A * B

matrix_multiplication = np.dot(A, B)

transpose_A = A.T
transpose_B = B.T

print("Matrix Addition:\n", addition)
print("------------------------------")
print("Element-wise Multiplication:\n", elementwise_multiplication)
print("------------------------------")
print("Matrix Multiplication (Dot Product):\n", matrix_multiplication)
print("------------------------------")
print("Transpose of A:\n", transpose_A)
print("Transpose of B:\n", transpose_B)
print("------------------------------")
