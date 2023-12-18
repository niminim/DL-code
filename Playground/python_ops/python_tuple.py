
### Tuple

## Tuple Access and Indexing:

# Indexing: Access individual elements using indexing.
my_tuple = (1, 2, 3)
element = my_tuple[1]
# Result: element=2

# Slicing: Extract a subset of elements using slicing.
my_tuple = (1, 2, 3, 4, 5)
subset = my_tuple[1:4]
# Result: subset=(2, 3, 4)

# len(): Retrieve the number of elements in the tuple.
my_tuple = (1, 2, 3)
length = len(my_tuple)
# Result: length=3

## Tuple Operations:

# Concatenation: Combine two tuples to create a new tuple.
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
concatenated_tuple = tuple1 + tuple2
# Result: concatenated_tuple=(1, 2, 3, 4, 5, 6)

# Repetition: Create a new tuple by repeating the elements.
my_tuple = (1, 2)
repeated_tuple = my_tuple * 3
# Result: repeated_tuple=(1, 2, 1, 2, 1, 2)

## Tuple Methods:

# count(item): Count the number of occurrences of a specified item.
my_tuple = (1, 2, 2, 3)
count = my_tuple.count(2)
# Result: count=2

# index(item): Find the index of the first occurrence of a specified item.
my_tuple = (1, 2, 3)
index = my_tuple.index(2)

# Conversion to Other Types:
# tuple(iterable): Convert an iterable (e.g., list) to a tuple.

my_list = [1, 2, 3]
converted_tuple = tuple(my_list)
# Result: converted_tuple=(1, 2, 3)

# Additions
# In the context of tuples, zip and "unzip" are operations that involve combining or separating tuples.

# zip Function:
# The zip function in Python is used to combine two or more iterables (e.g., tuples, lists) element-wise into tuples.
# It takes multiple iterables as arguments and returns an iterator of tuples where the i-th tuple contains the i-th element from each of the input iterables.

names = ('Alice', 'Bob', 'Charlie')
ages = (25, 30, 35)

## Zip the two tuples
zipped_data = zip(names, ages)

# Convert the result to a list for visualization
zipped_list = list(zipped_data)
# Result: [('Alice', 25), ('Bob', 30), ('Charlie', 35)]

## "Unzipping" with zip:
# To "unzip" the result of a zip operation, you can use the * unpacking operator along with the zip function.
# This effectively transposes the rows and columns of the zipped data.

zipped_data = [('Alice', 25), ('Bob', 30), ('Charlie', 35)]

# "Unzip" using the zip function and *
unzipped_names, unzipped_ages = zip(*zipped_data)
# Result: unzipped_names=('Alice', 'Bob', 'Charlie'), unzipped_ages=(25, 30, 35)

