###
# 1. Two-sum - https://leetcode.com/problems/two-sum/
# https://github.com/SamirPaul1/DSAlgo/blob/main/01_Problem-Solving-LeetCode/1-two-sum/1-two-sum.py
# ChatGPT Solution (similiar to 1)
def twoSum(nums, target):
    num_dict = {} # num_dict stores the value-index pairs
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_dict: # checking the keys in num_dict
            return [num_dict[complement], i] # return the index of the complement (in num_dict) and current index
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

# for all solution we can create "solutions = []" from the start
# do the same, but go over all the numbers in the list
# when find a solution: solutions.append([num_dict[complement], i])
# at the end: return solutions
#####

###
# 9 - Palindrome Number - https://leetcode.com/problems/palindrome-number/
# Both are ChatGPT Solutions
def isPalindrome(x):
    # Special case: negative numbers are not palindromes
    if x < 0:
        return False

    # Convert the integer to a string
    x_str = str(x)

    # Check if the string is equal to its reverse
    return x_str == x_str[::-1]


def isPalindrome(x): # solve without converting to integer
    # Special case: negative numbers are not palindromes
    if x < 0:
        return False

    original = x
    reversed_num = 0

    while x > 0:
        digit = x % 10
        reversed_num = reversed_num * 10 + digit
        x //= 10

    # Check if the reversed number is equal to the original
    return original == reversed_num


# Test the function
num = 121
result = isPalindrome(num)
print(result)

# in this approach we build the reversed number from right to left - find the digit, multiply by 10, and add that to the "current" reversed number,
# then we drop the last digit from the "current number" by //10
# we do that until our number is no longer greater than zero, and that's our reversed number

####

# 13 - Roam to Integer - https://leetcode.com/problems/roman-to-integer/
# ChatGP solution
def romanToInt(s: str) -> int:
    roman_dict = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }

    total = 0
    prev_value = 0

    for char in s:
        current_value = roman_dict[char]
        total += current_value

        if current_value > prev_value:
            # Subtract the twice of the previous value because
            # it was added in the previous step
            total -= 2 * prev_value

        prev_value = current_value

    return total

# Test the function with examples
print(romanToInt("III"))      # Output: 3
print(romanToInt("IV"))       # Output: 4
print(romanToInt("IX"))       # Output: 9
print(romanToInt("LVIII"))    # Output: 58
print(romanToInt("MCMXCIV"))  # Output: 1994
#

# https://github.com/SamirPaulb/DSAlgo/blob/main/01_Problem-Solving-LeetCode/13-roman-to-integer/13-roman-to-integer.py
class Solution:
    def romanToInt(self, s: str) -> int:
        n = len(s)
        res = 0
        dic = {"I": 1,
               "IV": 4,
               "V": 5,
               "IX": 9,
               "X": 10,
               "XL": 40,
               "L": 50,
               "XC": 90,
               "C": 100,
               "CD": 400,
               "D": 500,
               "CM": 900,
               "M": 1000,
               }

        i = 0
        while i < n:
            if i < n - 1 and s[i:i + 2] in dic: # check a combination of two letters
                res += dic[s[i:i + 2]]
                i += 2
            else:
                res += dic[s[i:i + 1]] # go letter by letter (dic[s[i:i + 1]] = dic[s[i:]])
                i += 1

        return res

sol = Solution()
print(sol.romanToInt("III"))      # Output: 3
print(sol.romanToInt("IV"))       # Output: 4
print(sol.romanToInt("IX"))       # Output: 9
print(sol.romanToInt("LVIII"))    # Output: 58
print(sol.romanToInt("MCMXCIV"))  # Output: 1994



## 21 - Merge Two Sorted Lists - https://leetcode.com/problems/merge-two-sorted-lists/
# ChatGPT
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeTwoLists(l1, l2):
    # Create a dummy node to simplify the code
    dummy_head = ListNode()
    current = dummy_head

    # Iterate while both lists have nodes
    while l1 and l2:
        # Compare the values of the current nodes
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next

        # Move to the next node in the result list
        current = current.next

    # Attach the remaining nodes from either list (if any)
    if l1:
        current.next = l1
    else:
        current.next = l2

    # Return the merged sorted list starting from the dummy head's next node
    return dummy_head.next

# Example usage:
# Constructing two sorted linked lists representing numbers 1->2->4 and 1->3->4
l1 = ListNode(1, ListNode(2, ListNode(4)))
l2 = ListNode(1, ListNode(3, ListNode(4)))

result = mergeTwoLists(l1, l2)

# Printing the result linked list
while result:
    print(result.val, end=" ")
    result = result.next

# More solutions:
# https://redquark.org/leetcode/0021-merge-two-sorted-lists/ (with explanations)
# https://github.com/SamirPaulb/DSAlgo/blob/main/01_Problem-Solving-LeetCode/21-merge-two-sorted-lists/21-merge-two-sorted-lists.py
# https://medium.com/nerd-for-tech/leetcode-merge-two-sorted-lists-99cc19e1b06e

######