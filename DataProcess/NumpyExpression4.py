import numpy as np


arr1=np.array([[1,2,3],[4,5,6]])
print(arr1[0][1])
print(arr1[0,1])

names=np.array(['Tony','Jack','Robin'])
print(names=='Tony')

print((names=='Tony') & (names=='Robin'))   # and
print((names=='Tony') | (names=='Robin'))   # or

# Fancy index
arr=np.empty((8,4))
print(arr)
for i in range(8):
    arr[i]=i
print(arr)
print(arr[[4,3,0,6]])

arr=np.arange(32).reshape((8,4))
print(arr)
print(arr[[1,5,7,2]])
print(arr[[1,5,7,2],[0,3,1,2]])
print(arr[[1,5,7,2]])
print(arr[[1,5,7,2]][:,[0,3,1,2]])
print(arr[[1,5,7,2]])
print(arr[np.ix_([1,5,7,2],[0,3,1,2])])