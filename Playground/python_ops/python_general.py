import numpy as np

# Currently for sorting and searching using python's built-in functions

# Array
my_list = [64, 34, 25, 12, 22, 11, 90]
sorted_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

### Sort Arrays
sorted_array = sorted(my_list)
my_list.sort() # in-place

# argsort - The numpy.argsort() function returns an array of indices that would sort the input array.
# You can then use these indices to access the elements in sorted order.
# If you want to sort along a specific axis for multidimensional arrays, you can specify the axis parameter in both functions.

sorted_indices = np.argsort(my_list)
sorted_array = my_list[sorted_indices]

# Sorting Along Axis:
my_2d_array = np.array([[5, 2, 8], [1, 3, 7]])

# Sort along columns (axis=1)
sorted_2d_array = np.sort(my_2d_array, axis=1)
print("Original 2D Array:\n", my_2d_array)
print("Sorted 2D Array along Columns:\n", sorted_2d_array)

# Sort along rows (axis=0)
sorted_2d_array = np.sort(my_2d_array, axis=0)
print("Original 2D Array:\n", my_2d_array)
print("Sorted 2D Array along Rows:\n", sorted_2d_array)
####


#### Search Array
index = my_list.index(22)

# Binary search
from bisect import bisect_left
target_number = 3
index = bisect_left(sorted_list, target_number)


# Search with condition
my_array = np.array([5, 2, 8, 1, 3])

# # Check if elements are greater than 3 (Get [True, False,  True, False, False])
result = my_array > 3

# Get indices where elements are greater than 3
indices = np.where(my_array > 3)
# or
bool_indices = np.nonzero(my_array > 3)

# Use the indices to get the elements
search_result = my_array[indices] # here we feed the explicit indices

search_result = my_array[bool_indices] # here we feed True/False for every element

print("Original Array:", my_array)
print("Elements greater than 3:", search_result)
#

# use np.any (and argwhere) to check a condition
my_array = np.array([5, 2, 8, 1, 3])

# Check if any element is equal to 8
result = np.any(my_array == 8)

print("Original Array:", my_array)
print("Is there an element equal to 8?", result)
if result:
    result_arg = np.argwhere(my_array == 8)
    print("Is there an element equal to 8 in index: ", result_arg)

# same for ndarray
my_array = np.array([[5, 2, 8, 1, 3],[10,20,30,8,50]])
# Check if any element is equal to 8
result = np.any(my_array == 8)

print("Original Array: ", my_array)
print("Is there an element equal to 8?", result)
if result:
    result_arg = np.argwhere(my_array == 8)
    print("Is there an element equal to 8 in index: ", result_arg)
####