
### Set

# Sets in Python are unordered collections of unique elements. They support a variety of built-in methods for common set operations.
# Here are some of the key functions and methods you can use with sets:

# Set Creation:
# set(iterable): Create a new set from an iterable (e.g., list, tuple).

my_list = [1, 2, 2, 3]
my_set = set(my_list)
# Result: {1, 2, 3}

## Set Operations:

# add(element): Add an element to the set. If the element is already present, the set remains unchanged.
my_set = {1, 2, 3}
my_set.add(4)
# Result: {1, 2, 3, 4}

# update(iterable): Add multiple elements to the set.
my_set = {1, 2, 3}
my_set.update([3, 4, 5])
# Result: {1, 2, 3, 4, 5}

# remove(element): Remove an element from the set. Raises an error if the element is not present.
my_set = {1, 2, 3}
my_set.remove(2)
# Result: {1, 3}

# discard(element): Remove an element from the set if it is present. Does not raise an error if the element is not found.
my_set = {1, 2, 3}
my_set.discard(2)
# Result: {1, 3}

# pop(): Remove and return an arbitrary element from the set. Raises an error if the set is empty.
my_set = {1, 2, 3}
popped_element = my_set.pop()

# clear(): Remove all elements from the set.
my_set = {1, 2, 3}
my_set.clear()
# Result: set()

# copy(): Create a shallow copy of the set.
my_set = {1, 2, 3}
copy_of_set = my_set.copy()

# difference_update(other_set, ...): Remove elements from the set that are also present in other sets.
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set1.difference_update(set2)
# Result: set1 is now {1, 2}

# intersection_update(other_set, ...): Update the set with common elements from itself and other sets.
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set1.intersection_update(set2)
# Result: set1 is now {3}

# symmetric_difference_update(other_set): Update the set with elements that are unique to itself and another set.
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set1.symmetric_difference_update(set2)
# Result: set1 is now {1, 2, 4, 5}

## Set Membership and Comparison:

# in: Check if an element is in the set.
my_set = {1, 2, 3}
is_present = 2 in my_set
# Result: True

# issubset(other_set): Check if the set is a subset of another set.
set1 = {1, 2}
set2 = {1, 2, 3}
is_subset = set1.issubset(set2)
# Result: True

# issuperset(other_set): Check if the set is a superset of another set.
set1 = {1, 2, 3}
set2 = {1, 2}
is_superset = set1.issuperset(set2)
# Result: True


## Set Mathematical Operations:
# union(other_set, ...): Return a new set containing all distinct elements from the sets.
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union_set = set1.union(set2)
# Result: {1, 2, 3, 4, 5}

# intersection(other_set, ...): Return a new set containing common elements from the sets.
set1 = {1, 2, 3}
set2 = {3, 4, 5}
intersection_set = set1.intersection(set2)
# Result: {3}

# difference(other_set, ...): Return a new set with elements from the set that are not in other sets.
set1 = {1, 2, 3}
set2 = {3, 4, 5}
difference_set = set1.difference(set2)
# Result: {1, 2}

# symmetric_difference(other_set): Return a new set with elements that are unique to each set.
set1 = {1, 2, 3}
set2 = {3, 4, 5}
symmetric_difference_set = set1.symmetric_difference(set2)
# Result: {1, 2, 4, 5}


