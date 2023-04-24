def is_not_none(s):
    return s and len(s.strip())>0

list1=[' ','','Hello','greedy',None,'ai']

result=filter(is_not_none,list1)
print(list(result))