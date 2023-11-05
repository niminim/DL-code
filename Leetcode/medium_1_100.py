###
# 2. Add Two-Numbers - https://leetcode.com/problems/add-two-numbers/
# https://github.com/SamirPaulb/DSAlgo/tree/main/01_Problem-Solving-LeetCode/2-add-two-numbers

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode()
        cur = dummy

        carry = 0
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0

            # new digit val
            val = v1 + v2 + carry
            carry = val // 10
            val = val % 10
            cur.next = ListNode(val)  # as in one place we have to put a single digit

            # update pointers
            cur = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        # alternatively, the caryy in the while loop can be omitted and addressed now (for carry to be added at the beginning
        # (for a case like 3 digits + 2digits = 4 digits)

        return dummy.next

def print_linked_list_number_reverse(node):
    digits = []
    while node:
        digits.append(str(node.val))
        node = node.next
    number = int("".join(digits[::-1]))
    print(number)

### Example usage:
mode = 'hard1'
if mode == 'easy':
    l1 = ListNode(2, ListNode(4, ListNode(3)))
    l2 = ListNode(5, ListNode(6, ListNode(4)))
elif mode == 'hard1':
    l1 = ListNode(3, ListNode(2, ListNode(4, ListNode(3))))
    l2 = ListNode(5, ListNode(6, ListNode(4)))
elif mode == 'hard2':
    l1 = ListNode(8)
    l2 = ListNode(7)
elif mode == 'hard3':
    l1 = ListNode(8,ListNode(1))
    l2 = ListNode(7)

solution = Solution()
result = solution.addTwoNumbers(l1, l2)
print(print_linked_list_number_reverse(result))


def print_linked_list(node):
    # print the list (remember the reverse order)
    while node:
        print(node.val, end=" -> ")
        node = node.next
    print("None")

print_linked_list(l1)
print_linked_list(l2)

# More solutions: https://github.com/kamyu104/LeetCode-Solutions/blob/master/Python/add-two-numbers.py
####