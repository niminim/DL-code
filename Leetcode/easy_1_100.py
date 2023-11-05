###
# 1. Two-sum - https://leetcode.com/problems/two-sum/
# https://github.com/SamirPaul1/DSAlgo/blob/main/01_Problem-Solving-LeetCode/1-two-sum/1-two-sum.py

def twoSum(nums, target):
    num_dict = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_dict:
            return [num_dict[complement], i]
        num_dict[num] = i
        print(num_dict)
    return []

# Test the function
nums = [2, 7, 11, 15]
target = 9
result = twoSum(nums, target)
print(result)

# More solutions: https://github.com/kamyu104/LeetCode-Solutions/blob/master/Python/add-two-numbers.py
# https://github.com/Garvit244/Leetcode/blob/master/1-100q/TwoSum.py
#####

