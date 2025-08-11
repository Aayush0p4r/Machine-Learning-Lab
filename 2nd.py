import numpy as np

A = np.array([[4, 2], [1, 3]])

# eigenvalues, eigenvectors = np.linalg.eig(A)
transpose_A = A.T
eigenvalues, eigenvectors = np.linalg.eig(transpose_A)


print("Transpose of A:\n", transpose_A)
print("Eigenvalues of A:\n", eigenvalues)
print("Eigenvectors of A:\n", eigenvectors) 