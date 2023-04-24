list1=[1,2,3,4,5]
r=map(lambda x:x+x,list1)
print(list(r))

ml=map(lambda x,y:x*x+y,[1,2,3,4,5],[1,2,3,4,5])
print(list(ml))