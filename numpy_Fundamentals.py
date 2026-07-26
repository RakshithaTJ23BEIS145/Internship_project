import numpy as np
import pandas as pd

arr1 = np.array([1, 2, 3, 4, 5])

arr2 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

arr3 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("1D Array:")
print(arr1)
print("Shape:", arr1.shape)

print("\n2D Array:")
print(arr2)
print("Shape:", arr2.shape)

print("\n3D Array:")
print(arr3)
print("Shape:", arr3.shape)


print("\nBroadcasting:")
broadcast_result = arr2 + np.array([10, 20, 30])
print(broadcast_result)


print("\nVectorized Operations:")
print("arr1 * 2 =", arr1 * 2)
print("arr1 + 5 =", arr1 + 5)
print("arr1 ** 2 =", arr1 ** 2)


A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print("\nMatrix Multiplication:")
print(A @ B)


df = pd.read_csv("data.csv")


numeric_df = df.select_dtypes(include=np.number)

print("\nMean:")
print(numeric_df.mean())

print("\nStandard Deviation:")
print(numeric_df.std())

print("\nCorrelation Matrix:")
print(numeric_df.corr())