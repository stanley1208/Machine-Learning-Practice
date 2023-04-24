import numpy as np

arr=np.arange(15).reshape((3,5))
print(arr)
print(arr.transpose())
print(arr.T)

arr2=np.arange(24).reshape((2,3,4))
print(arr2)
print(arr2.transpose((1,2,0)))
print(arr2.transpose((1,0,2)))
