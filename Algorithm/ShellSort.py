SIZE=8

def showdata(data):
    for i in range(SIZE):
        print('%3d'%data[i],end='')
    print()

def shell(data,size):
    k=1
    jmp=size//2
    while jmp!=0:
        for i in range(jmp,size):
            tmp=data[i]
            j=i-jmp
            while tmp<data[j] and j>=0:
                data[j+jmp]=data[j]
                j=j-jmp
            data[jmp+j]=tmp
        print('第 %d 次排序過程:'%k,end='')
        k+=1
        showdata(data)
        print('----------------------')
        jmp=jmp//2

def main():
    data=[16,25,39,27,12,8,45,63]
    print('原始陣列是:    ')
    showdata(data)
    print('----------------------------')
    shell(data,SIZE)

main()

