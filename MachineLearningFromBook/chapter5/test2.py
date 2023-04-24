def coinChange1(coins, amount):
    """
    :type coins: List[int]
    :type amount: int
    :rtype: int
    """
    coins.sort() #給硬幣從小到大排序
    dp = {0:0} #生成字典dp，並且當總金額為0時，最少硬幣個數為0
    print(dp)
    for i in range(1,amount + 1):
        print('i:',i)
        dp[i] = amount + 1 #因為硬幣個數不可能大於amount，所以賦值amount + 1便於比較
        print('dp[i]:',dp[i])
        for j in coins:
            if j <= i:

                dp[i]=min(dp[i],dp[i-j]+1)
                print('dp[i]:',dp[i])
    #for i in range(1,amount + 1):
    #print('dp[%d]:'%(i), dp[i])
    if dp[amount] == amount + 1: #當最小硬幣個數為初始值時，代表不存在硬幣組合能構成此金額
        return -1
    else:

        return dp[amount]

print(coinChange1([5,4,3,1],11))




