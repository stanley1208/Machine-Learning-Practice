import numpy as np

arr=np.arange(10)
# print(arr)
np.save('any_array',arr)
print(np.load('any_array.npy'))

arr1=np.arange(10)
np.savez('any_array_1',a=arr1)  # a -> keyword
print(np.load('any_array_1.npz')['a'])

arr2=np.arange(10)
np.savez('any_array_2.npz',a=arr1)  # a -> keyword
print(np.load('any_array_2.npz')['a'])

arr3=np.arange(10)
np.savetxt('any_array.txt',arr3,delimiter=',')  # delimiter=','
print(np.loadtxt('any_array.txt',delimiter=','))