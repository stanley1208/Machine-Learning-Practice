import numpy as np

arr=np.random.randn(4,4)
print(arr)
print(arr.mean())
print(np.mean(arr))
print(arr.sum())
print(arr.std())

print(arr.mean(axis=1))
print(arr.sum(0))

arr2=np.random.randn(4)
print(arr2)
arr2.sort()
print(arr2)

arr3=np.random.randn(4,4)
# print(arr3)
# arr3.sort(1)
# print(arr3)
print(arr3)
arr3.sort(1)
print(arr3)


