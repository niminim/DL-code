
### Bubble Sort

# Bubble Sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements,
# and swaps them if they are in the wrong order. Here's a Python implementation of Bubble Sort:

# At the end of the first iteration moves the largest to the end,
# At the second of the second iteration the second largest number is second from the right, and so on
# So after the first iteration we need to sort only the first n-1 numbers, after the second iteration we need to sort only the first n-2 numbers, and so on

def bubble_sort(arr):
    n = len(arr)

    # Traverse through all elements in the array
    for i in range(n):

        # Last i elements are already in place, so we don't need to check them
        for j in range(0, n - i - 1):

            # Swap if the element found is greater than the next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# Example usage:
my_list = [64, 34, 25, 12, 22, 11, 90]
print("Original List:", my_list)

bubble_sort(my_list)
print("Sorted List:", my_list)

# example https://www.javatpoint.com/bubble-sort-in-python

def my_bubble_sor(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0,n-1-i):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

my_bubble_sor(my_list)

###### End of Bubble Sort

######

### Binary Search
# Binary Search is an efficient algorithm for finding an element in a sorted array.
# Here's a Python implementation of Binary Search:

def binary_search(arr, target):
    """
    Perform binary search on a sorted array.

    Parameters:
    - arr (list): The sorted array.
    - target: The element to search for.

    Returns:
    - int: The index of the target if found, or -1 if not found.
    """

    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2  # Calculate the middle index

        if arr[mid] == target:
            return mid  # Element found, return its index
        elif arr[mid] < target:
            low = mid + 1  # Discard the left half
        else:
            high = mid - 1  # Discard the right half

    return -1  # Element not found

# Example usage:
sorted_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
target_element = 5

result = binary_search(sorted_list, target_element)

if result != -1:
    print(f"Element {target_element} found at index {result}.")
else:
    print(f"Element {target_element} not found in the list.")


# example - https://www.javatpoint.com/binary-search-in-python

def my_binary_search(arr, target):

    low, high = 0, len(arr)

    while low <= high:
        mid = (low + high)//2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1

result = my_binary_search(sorted_list, target_element)

######## End of Binary Search

