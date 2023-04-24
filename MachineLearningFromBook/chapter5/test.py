nums=[1,5,3,2,7,9,5,3]


def length(nums):
    if len(nums)<=1:
        return len(nums)
    mem = [1 for _ in range(len(nums))]
    for j in range(1,len(nums)):
        for i in range(0,j):
            if nums[i]<nums[j]:
                mem[j]=max(mem[j],mem[i]+1)
                print(mem[j])
    return max(mem)

print(length(nums))