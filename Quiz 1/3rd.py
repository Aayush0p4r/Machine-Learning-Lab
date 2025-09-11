A = [1, 2, 3, 4]
B = [4, 3, 2, 1]

matrix = [[0 for a in range(len(A[0]))] for b in range(len(A))]

for i in range(len(A)):
    for j in range(len([0])):
        matrix[i][j] = A[i][j] * B[i][j]

print("Matrix Multiplication:")
for row in matrix:
    print(row)