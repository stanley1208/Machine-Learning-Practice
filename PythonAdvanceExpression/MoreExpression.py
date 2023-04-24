# list
list1=[1,2,3,4,5,6,7,8,9,10]
f=map(lambda x:x+x,list1)
print(list(f))

list2=[i+i for i in list1]
print(list2)

list3=[i**3 for i in list1]
print(list3)

list4=[i*i for i in list1 if i>3]
print(list4)

# set
list1={1,2,3,4,5,6,7,8,9,10}

list2={i+i for i in list1}
print(list2)

list3={i**3 for i in list1}
print(list3)

list4={i*i for i in list1 if i>3}
print(list4)

# dict
s={
    "aaa":20,
    "bbb":15,
    "ccc":31
}

s_key=[key+"aaa" for key in s.keys()]
print(s_key)

# key, value reverse
s1={value:key for key,value in s.items()}
print(s1)

s2={key:value for key,value in s.items() if key=='aaa'}
print(s2)
