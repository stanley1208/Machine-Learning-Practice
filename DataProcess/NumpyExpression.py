import numpy as np

data=[1,2,3,4,5]
n=np.array(data*10)
print(data)
print(n)

print(n.shape)  # datashape
print(n.dtype)  # datatype

arr=[[1,2,3,4],[1,2,3,4]]
arr2=np.array(arr)
print(arr2)
# print(np.array(arr))
print(arr2.ndim) # dimension
print(arr2.shape)

# not a good method
arr=[['1','2',3,4],[5,6,7,8]]
arr2=np.array(arr)
print(arr2)
print(arr2.dtype)   # unicode

arr=[[1,2,3,4],[5,6,7,8]]
arr2=np.array(arr)
print(arr2)
print(arr2.dtype)   # int32

arr=[[1.1,2,3,4],[5,6,7,8]]
arr2=np.array(arr)
print(arr2)
print(arr2.dtype)   #float64

print(np.zeros(10))
print(np.ones((2,3)))
print(np.empty((2,3,4)))

print(np.arange(10)) # 0~9

arr=np.array([1.2,1.6,1.8,-2.3,-5.8])
print(arr)
print(arr.dtype)
#   int 8 int16 int32 int64
#   float16 float32 float64 float128
print(arr.astype(np.int64))