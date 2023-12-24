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
            v1 = l1.val if l1 else 0 # the else is for a cse where one list is longer than the other
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


###
# 3. Longest Substring Without Repeating Characters  - https://leetcode.com/problems/longest-substring-without-repeating-characters/
# both set and map solutions are included - https://www.youtube.com/watch?v=qtVh-XEpsJo

# set solution - https://github.com/SamirPaulb/DSAlgo/blob/main/01_Problem-Solving-LeetCode/3-longest-substring-without-repeating-characters/3-longest-substring-without-repeating-characters.py
class Solution:
    def lengthOfLongestSubstring(self, s):
        track = set()
        l = 0
        res = 0

        for r in range(len(s)):
            print('* new r: ', r, '(s[r] - ', s[r],')')
            while s[r] in track:
                print('***** in while loop - while ', s[r], ' in track')
                print('track: ', track)
                print('remove ', s[l], ' from track')
                track.remove(s[l])
                print('current track: ', track)
                l += 1
                print(' new l: ', l)
            print('*** finished while loop')
            print('r: ', r, ', l: ', l)
            res = max(res, r - l + 1)
            print('res: ', res)
            print('track - add: ', s[r])
            track.add(s[r])
            print('updated track: ', track)
            print('***** finished r =', r, ' ******')

        return res

str0 = "abcabcd"
str1 = "abcabcbb"
str2 = "bbbbb"
str3 = "pwwkew"

str = Solution()
res = str.lengthOfLongestSubstring(str3)
print('max len: ', res)
#

# # map solution - https://github.com/Garvit244/Leetcode/blob/master/1-100q/03.py (this one is better)
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        mapSet = {}
        start, result = 0, 0

        for end in range(len(s)):
        	if s[end] in mapSet:
        		start = max(mapSet[s[end]], start)
        	result = max(result, end-start+1)
        	mapSet[s[end]] = end+1

        return result

sol = Solution()
result = sol.lengthOfLongestSubstring(str0)
#

# ChatGPT (also map solution)
def lengthOfLongestSubstring(s: str) -> int:
    char_index_map = {}
    max_length = 0
    start_index = 0

    for end_index, char in enumerate(s):
        if char in char_index_map and char_index_map[char] >= start_index:
            # If the character is repeated and its previous index is after or at the start_index,
            # update the start_index to the next position after the previous index.
            start_index = char_index_map[char] + 1

        # Update the index of the current character in the char_index_map.
        char_index_map[char] = end_index

        # Update the maximum length.
        max_length = max(max_length, end_index - start_index + 1)

    return max_length
lengthOfLongestSubstring(str0)
####

#### 6. Zigzag Conversion - https://leetcode.com/problems/zigzag-conversion/
# ChatGPT

def convert(s: str, numRows: int) -> str:
    if numRows == 1 or numRows >= len(s):
        return s

    result = [''] * numRows
    index, step = 0, 1

    for char in s:
        result[index] += char

        if index == 0:
            step = 1
        elif index == numRows - 1:
            step = -1

        index += step

    return ''.join(result)

# Test the function with examples
print(convert("PAYPALISHIRING", 3))
print(convert("PAYPALISHIRING", 4))

# Explanation:
# If numRows is 1 or greater than or equal to the length of the string s, no zigzag conversion is needed, so return the string as it is.
# Create a list result to store each row's characters.
# Initialize index to 0 and step to 1. Iterate through each character in s.
# For each character, add it to the corresponding row in the result list.
# Update index and step based on whether the current row is at the top or bottom of the zigzag pattern.
# Finally, join the rows in the result list to get the zigzag conversion.
# This solution has a time complexity of O(n), where n is the length of the input string
#

# https://www.youtube.com/watch?v=Q2Tw6gcVEwc
class Solution:
    def convert(self, s:str, numRows:int) -> str:
        if numRows == 1:
            return s

        res = ""
        for r in range(numRows):
            increment = 2 * (numRows - 1)
            for i in range(r, len(s), increment):
                res += s[i]
                if (r > 0 and r < numRows-1) and (i + increment - 2*r < len(s)):
                    res += s[i + increment - 2*r]
        return res

sol = Solution()
sol.convert("PAYPALISHIRING", 3)
#

# https://github.com/Garvit244/Leetcode/blob/master/1-100q/06.py

class Solution(object):
    def convert(self, s, numRows):

        if numRows == 1:
            return s

        result = ["" for _ in range(numRows)]
        row, down = 0, 1
        for char in s:
            result[row] += char

            if row == numRows - 1:
                down = 0
            if row == 0:
                down = 1

            if down:
                row += 1
            else:
                row -= 1
        final_string = ""
        for value in result:
            final_string += value
        return final_string

Solution().convert("PAYPALISHIRING", 3)
######