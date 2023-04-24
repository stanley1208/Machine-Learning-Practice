def compute(a,b):
    n,s=10,0
    for i in range(n):
        for j in range(n):
            s=s+a*b
    return s

print(compute(1,2))

# 時間複雜度:O(n^2),空間複雜度:O(1)