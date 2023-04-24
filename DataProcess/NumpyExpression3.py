import numpy as np


arr=np.arange(10)
print(arr)
print(arr[1])
print(arr[4:])

arr[0:4]=11
print(arr)

arr_copy=arr.copy()
print(arr_copy)