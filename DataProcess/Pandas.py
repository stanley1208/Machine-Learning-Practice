import pandas as pd
from pandas import Series,DataFrame


obj1=Series([1,2,3,4,5])

print(obj1)
print(obj1.values)
print(obj1.index)

print()

obj2=Series(['a','b','c','d','e'],index=[1,2,3,4,5])

print(obj2)
print(obj2[1])
print(obj2[2])

print()

data={'a':10000,'b':20000,'c':30000}
obj3=Series(data)

print(obj3)

keys=['a','c']
obj3_1=Series(data,index=keys)
print(obj3_1)

data2={'a':None,'b':20000,'c':30000}
obj4=Series(data2)
print(obj4)

print(pd.isnull(obj4))
print(pd.notnull(obj4))
print(obj4.isnull())

print()

data3={'aaa':None,'bbbb':25,'Stanley':None,'Jack':500}
obj5=Series(data3)
obj5.name=('NameAngAnge')
obj5.index.name='000'
print(obj5.values)
print(obj5.index.values)
print(obj5)