import numpy as np

# Create an array from a list or tuple
arr = np.array([1, 2, 3])

# Create arrays filled with zeros or ones
zeros_arr = np.zeros((3, 4))
ones_arr = np.ones((2, 2))

# Create arrays with regularly spaced values
range_arr = np.arange(0, 10, 2)  # Start, stop, step
linspace_arr = np.linspace(0, 1, 5)  # Start, stop, num

# Generate arrays with random values
rand_arr = np.random.rand(2, 3)  # Uniform distribution [0, 1)
randn_arr = np.random.randn(2, 3)  # Standard normal distribution

# Create identity matrices or extract diagonal elements.
identity_matrix = np.eye(3)
diagonal_elements = np.diag([1, 2, 3])

## Array Manipulation:
arr1 = [1,2,3]
arr2 = [4,5,6]
# Reshape or flatten arrays
reshaped_arr = arr.reshape((3, 1))
flattened_arr = arr.flatten()

# Concatenate arrays horizontally or vertically
concatenated_arr = np.concatenate([arr1, arr2], axis=0)
vstack_arr = np.vstack([arr1, arr2])
hstack_arr = np.hstack([arr1, arr2])

# Split arrays into smaller arrays
subarrays = np.split(arr, 3)  # Split into 3 equal parts
hsplit_arrs = np.hsplit(arr, 3)  # Split horizontally into 3 parts

## Indexing and Slicing:

# Use slices and fancy indexing to access elements
sliced_arr = arr[1:3]
fancy_indexing_arr = arr[[0, 2, 1]]

# Return elements chosen from two arrays based on a condition
result_arr = np.where(arr > 2, 1, 0)

# More about Indexing and Slicing in NumPy:
# NumPy supports multidimensional arrays, and indexing and slicing can be applied along each dimension
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Accessing elements
element = arr_2d[1, 2]  # Get the element at row 1, column 2

# Boolean Indexing: you can use boolean arrays for indexing, which is particularly useful for conditional filtering
mask = arr > 3
filtered_arr = arr[mask]

# Slicing with Steps: Specify a step value when slicing to skip elements
sliced_arr = arr[1:10:2]  # Start from index 1, stop at index 10, step by 2

# Ellipsis (...): The ellipsis (...) can be used to represent multiple colons in a slice for higher-dimensional arrays.
arr_3d = np.random.rand(3, 4, 5)
sliced_arr = arr_3d[..., 2]  # Equivalent to arr_3d[:, :, 2]

# Integer Array Indexing: Use arrays of integers to index into another array
indices = np.array([0, 2, 1])
indexed_arr = arr[indices]

# Combining Slicing and Integer Indexing: You can combine slicing and integer indexing for more complex operations
combined_arr = arr[:3, [0, 2]]

# Conditional Indexing: Use the np.where function for conditional indexing
indices = np.where(arr > 3)
conditional_arr = arr[indices]

# Negative Indexing: Negative indices can be used to count elements from the end of the array
last_element = arr[-1]

# Indexing with Arrays of Booleans: Boolean arrays can be used for both selection and assignment
arr[arr > 3] = 0  # Set elements greater than 3 to 0

# Using np.ix_ for Cross-Indexing: The np.ix_ function allows cross-indexing for 2D arrays
arr_2d = np.random.rand(3, 4)
indices = (np.array([0, 1]), np.array([2, 3]))
cross_indexed_arr = arr_2d[np.ix_(*indices)]

# Slice Assignment:
arr[1:4] = 99

# Integer Array Indexing for Assigning: Assign values using integer array indexing
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([1, 3, 4])
arr[indices] = 0

# Row and Coulmn Summation:
# Example 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Sum columns (axis=0)
column_sums = np.sum(arr_2d, axis=0)

# Sum rows (axis=1)
row_sums = np.sum(arr_2d, axis=1)


### Other Functions:

# Find unique elements in an array
unique_elements = np.unique(arr)

# Save and load arrays to/from binary files
np.save('my_array.npy', arr)
loaded_arr = np.load('my_array.npy')

# Create coordinate matrices from coordinate vectors
x = np.array([1, 2, 3])
y = np.array([4, 5])
X, Y = np.meshgrid(x, y)

# Dimensions
# expand_dims takes two arguments: the array you want to modify and the axis along which you want to add a new dimension.
arr = np.array([1, 2, 3])

# Add a new axis to convert a 1D array to a column vector (2D array)
arr_expanded = np.expand_dims(arr, axis=1)

# newaxis is often used in array indexing to add a new axis.
arr = np.array([1, 2, 3])

# Add a new axis to convert a 1D array to a column vector (2D array)
arr_expanded = arr[:, np.newaxis]

# The np.squeeze function is used to remove single-dimensional entries from the shape of an array.
# It reduces the number of dimensions by removing axes with size 1. If a particular axis is specified, only that axis will be removed.
arr = np.array([[[1]], [[2]], [[3]]])
squeezed_arr = np.squeeze(arr, axis=1)



## Mathematical Operations

# Calculate sum, mean, and standard deviation
total_sum = np.sum(arr)
mean_value = np.mean(arr)
std_dev = np.std(arr)

# Perform matrix multiplication
matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
matrix2 = np.array([[7, 8], [9, 10], [11, 12]])
matrix_product = np.dot(matrix1, matrix2)

matrix_product = matrix1 @ matrix2

# Compute element-wise trigonometric and exponential functions
sin_values = np.sin(arr)
exponential_values = np.exp(arr)

## Linear Algebra:

# Compute the inverse and determinant of a matrix
matrix = np.array([[4, 7],
                   [2, 6]])
inverse_matrix = np.linalg.inv(matrix)
determinant_value = np.linalg.det(matrix)

# Compute the eigenvalues and eigenvectors of a square matrix
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# Statistical Functions:
# Find minimum, maximum, and median values
min_value = np.min(arr)
max_value = np.max(arr)
median_value = np.median(arr)

# Compute the q-th percentile of the data
arr = np.array([20, 15, 30, 25, 10, 35, 40, 50, 45, 5])
percentile_value = np.percentile(arr, 75)

