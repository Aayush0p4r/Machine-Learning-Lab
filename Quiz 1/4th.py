A = [1, 2, 3, 4]

transpose = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

print("Transpose of Matrix:")
for row in transpose:
    print(row)