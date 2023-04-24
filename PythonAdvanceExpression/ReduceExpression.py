from functools import reduce

f=lambda x,y:x+y
r=reduce(f,[1,2,3,4,5],100)

print(r)