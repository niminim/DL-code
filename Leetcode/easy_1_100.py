###
# 1. Two-sum - https://leetcode.com/problems/two-sum/
# https://github.com/SamirPaul1/DSAlgo/blob/main/01_Problem-Solving-LeetCode/1-two-sum/1-two-sum.py
# ChatGPT Solution (similiar to 1)
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

###
# 9 - Palindrome Number
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

####

## 21 - Merge Two Sorted Lists
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