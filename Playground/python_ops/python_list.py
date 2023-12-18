
### Lists
# append(item): Adds an item to the end of the list.
my_list = [1, 2, 3]
my_list.append(4)
# Result: [1, 2, 3, 4]

# extend(iterable): Extends the list by appending elements from an iterable.
my_list = [1, 2, 3]
my_list.extend([4, 5])
# Result: [1, 2, 3, 4, 5]

# insert(index, item): Inserts an item at a specified position in the list.
my_list = [1, 2, 3]
my_list.insert(1, 4)
# Result: [1, 4, 2, 3]

# remove(item): Removes the first occurrence of the specified item.
my_list = [1, 2, 3, 2]
my_list.remove(2)
# Result: [1, 3, 2]

# pop([index]): Removes and returns the item at the specified index. If no index is provided, it removes and returns the last item.
my_list = [1, 2, 3]
popped_item = my_list.pop(1)
# Result: popped_item=2, my_list=[1, 3]


### List Access and Searching:

# index(item[, start[, end]]): Returns the index of the first occurrence of the specified item. Optional arguments allow searching within a specific range.
my_list = [1, 2, 3, 2]
index = my_list.index(2)
# Result: index=1

# count(item): Returns the number of occurrences of the specified item in the list.
my_list = [1, 2, 3, 2]
count = my_list.count(2)
# Result: count=2

### List Sorting and Reversing:

# sort([key=None, reverse=False]): Sorts the list in ascending order. Optional parameters allow custom sorting.
my_list = [3, 1, 4, 1, 5, 9]
my_list.sort()
# Result: [1, 1, 3, 4, 5, 9]

# reverse(): Reverses the order of the elements in the list.
my_list = [1, 2, 3]
my_list.reverse()
# Result: [3, 2, 1]

## Other List Operations:

# len(): Returns the number of elements in the list.
my_list = [1, 2, 3]
length = len(my_list)
# Result: length=3

# clear(): Removes all items from the list.
my_list = [1, 2, 3]
my_list.clear()
# Result: []

# copy(): Returns a shallow copy of the list.
my_list = [1, 2, 3]
copy_of_list = my_list.copy()


# More
# Loop over two lists - use enumerate and zip
list1 = ['apple', 'banana', 'orange']
list2 = [5, 10, 7]

for index, (element1, element2) in enumerate(zip(list1, list2)):
    print(f'Index {index}: {element1} - Quantity: {element2}')