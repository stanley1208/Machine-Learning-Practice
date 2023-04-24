def select(data):
    for i in range(7):
        print(data)
        for j in range(i+1,7):


            if data[i]>data[j]:
                data[i],data[j]=data[j],data[i]

    print()
data=[16,25,39,27,12,8,45,63]
print('原始資料為:')
for i in range(8):
    print('%3d'%data[i],end='')
print()
select(data)
print('排序後資料:')
for i in range(8):
    print('%3d'%data[i],end='')
print()