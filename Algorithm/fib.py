def fib(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    else:
        return fib(n-1)+fib(n-2)


output=[None]*1000
def fib2(n):
    result=output[n]
    if result==None:
        if n==0:
            return 0
        elif n==1:
            return 1
        else:
            result=fib2(n-1)+fib2(n-2)
        output[n]=result
    return result

n=int(input())
for i in range(n+1):
    print('fib (%d)=%d'%(i,fib(i)))

n=int(input())
for i in range(n+1):
    print('fib (%d)=%d'%(i,fib2(i)))