data=[16,25,39,27,12,8,45,63]
print('氣泡排序法:原始資料為:')
for i in range(8):
    print('%3d'%data[i],end='')
print()
for i in range(7,-1,-1):
    for j in range(i):
        if data[j]>data[j+1]:
            data[j],data[j+1]=data[j+1],data[j]
    print('第%3d 次排序後的結果是:'%(8-i),end='')
    for j in range(8):
        print('%3d'%data[j],end='')
    print()
print("排序後的結果為:")
for i in range(8):
    print('%3d' % data[i], end='')
print()