def binarySearch(list, keyWord):
    low = 0
    high = len(list) - 1

    while high >= low:
        mid = (low + high) // 2
        if keyWord < list[mid]:
            high = mid - 1
        elif keyWord == list[mid]:
            return mid
        else:
            low = mid + 1
    return -1

# 主程式
if __name__ == "__main__":

    # 輸入六個值
    num1 = input()
    num2 = input()
    num3 = input()
    num4 = input()
    num5 = input()
    num6 = input()

    # 根據題目建立兩個list
    list1 = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    list2 = [1.1, 3.3, 5.5, 7.7, 9.9, 11.1, 13.3, 15.5, 17.7]

    # 印出 list1
    print("list1:" + str(list1))
    # 尋找輸入的三個值
    print(num1 + " at " + str(binarySearch(list1, int(num1))))
    print(num2 + " at " + str(binarySearch(list1, int(num2))))
    print(num3 + " at " + str(binarySearch(list1, int(num3))))

    # 印出 list2
    print("\nlist2:" + str(list2))
    # 尋找輸入的三個值
    print(num4 + " at " + str(binarySearch(list2, float(num4))))
    print(num5 + " at " + str(binarySearch(list2, float(num5))))
    print(num6 + " at " + str(binarySearch(list2, float(num6))))