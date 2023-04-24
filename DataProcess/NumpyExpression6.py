import numpy as np

# np.where = x if condition else y
x_arr=np.array([1.1,1.2,1.3])
y_arr=np.array([2.1,2.2,2.3])
condition=np.array([True,False,True])
result=[(x if c else y) for x,y,c in zip(x_arr,y_arr,condition)]
print(result)

r=np.where(condition,x_arr,y_arr)
print(r)

arr=np.random.randn(4,4)
print(arr)

arr_1=np.where(arr>0,2,-2)
print(arr_1)

arr_2=np.where(arr>0,2,arr)
print(arr_2)