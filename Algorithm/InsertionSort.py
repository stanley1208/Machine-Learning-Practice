SIZE=8
def showdata(data):
    for i in range(SIZE):
        print('%3d'%data[i],end='')
    print()

def insert(data):
    for i in range(1,SIZE):
        tmp=data[i]
        no=i-1
        while no>=0 and tmp<data[no]:
            data[no+1]=data[no]
            no-=1
            data[no+1]=tmp

def main():
    data=[16,25,39,27,12,8,45,63]
    print('原始陣列是:')
    showdata(data)
    insert(data)
    print('排序後陣列是:')
    showdata(data)


main()