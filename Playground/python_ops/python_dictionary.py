# Dictionary
# Dictionaries in Python are versatile data structures that store key-value pairs.
# They offer a variety of built-in methods for performing operations on the keys, values, and items within the dictionary.
# Here are some common functions and methods you can use with dictionaries:

# Note: Starting from Python 3.7, the built-in dict type also maintains order.

### Dictionary Creation and Modification:
# dict(): Create a new dictionary.
my_dict = dict()
# or
my_dict = {}

# dict(key1=value1, key2=value2, ...): Create a dictionary with specified key-value pairs.
person = dict(name='Alice', age=30, city='Wonderland')

# len(dict): Get the number of key-value pairs in the dictionary.
size = len(my_dict)

# clear(): Remove all items from the dictionary.
my_dict.clear()

# copy(): Create a shallow copy of the dictionary.
copy_of_dict = my_dict.copy()

# update(dict2): Merge the contents of another dictionary into the current dictionary.
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
dict1.update(dict2)
# Result: {'a': 1, 'b': 3, 'c': 4}


##3 Accessing Dictionary Elements:

# get(key, default): Get the value associated with a key. Return a default value if the key is not found.
value = my_dict.get('key', default_value)

# keys(): Get a list of all keys in the dictionary.
all_keys = my_dict.keys()

# values(): Get a list of all values in the dictionary.
all_values = my_dict.values()

# items(): Get a list of key-value pairs (tuples) in the dictionary.
all_items = my_dict.items()


### Dictionary Manipulation:

# pop(key, default): Remove and return the value associated with a key. Return a default value if the key is not found.
popped_value = my_dict.pop('key', default_value)

# popitem(): Remove and return the last key-value pair as a tuple.
last_item = my_dict.popitem()

# setdefault(key, default): Get the value associated with a key. If the key is not found, set the key to the default value.
value = my_dict.setdefault('key', default_value)


### Dictionary Comprehension:
# Dictionary Comprehension: Create a dictionary using a concise syntax.
squares = {x: x**2 for x in range(5)}
# Result: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}


### Membership Testing:
# in: Check if a key is in the dictionary.
is_present = 'key' in my_dict

# not in: Check if a key is not in the dictionary.
is_absent = 'key' not in my_dict


### Removal and Deletion:
# del: Delete a key or the entire dictionary.
del my_dict['key']
# or
del my_dict

### More
# Merging Dictionaries (Python 3.9 and later):
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged_dict = dict1 | dict2
# Result: {'a': 1, 'b': 3, 'c': 4}

## Nested Dictionaries:
# Nested Dictionaries: Dictionaries can be nested to represent more complex data structures.
nested_dict = {'person': {'name': 'Alice', 'age': 30}}

# Pretty Printing:
# Module: pprint
#Description: Use pprint for more readable printing of nested dictionaries.
from pprint import pprint
pprint(nested_dict)

# Dictionary Comprehension with Conditions:
# Dictionary Comprehension with Conditions: Create dictionaries based on conditions.
my_dict = {x: x**2 for x in range(10) if x % 2 == 0}
# Result: {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

## JSON Serialization:
# JSON Serialization: Dictionaries are often used to represent JSON-like structures, and Python provides methods for JSON serialization and deserialization.
import json

my_dict = {'name': 'Alice', 'age': 30}
json_string = json.dumps(my_dict)  # Convert to JSON string

## Counting with Counters:
# Counter: The collections module includes the Counter class, which is a specialized dictionary for counting hashable objects.
from collections import Counter

my_list = [1, 2, 3, 1, 2, 1, 4, 5]
counter = Counter(my_list)
# Result: Counter({1: 3, 2: 2, 3: 1, 4: 1, 5: 1})

## Handling Duplicates:
# Handling Duplicates: Dictionaries automatically handle key uniqueness, making them useful for eliminating duplicates.
unique_elements = list(set(my_list))  # Removing duplicates using a set

## Efficient Lookup:
# Efficient Lookup: Dictionaries provide O(1) average time complexity for key lookups, making them efficient for large datasets.
value = my_dict['key']  # Fast lookup by key


## Configuration Settings:
# Configuration Settings: Dictionaries are commonly used to store configuration settings for applications.
config = {'debug': True, 'timeout': 30}

## Efficient Key Searches:
# Efficient Key Searches: Searching for keys in dictionaries is more efficient than searching for values in lists.
key_exists = 'key' in my_dict

# switching between keys and values:
my_dict = {'a': 1, 'b': 2, 'c': 3}

# Switch keys and values using a dictionary comprehension
switched_dict = {value: key for key, value in my_dict.items()}

## Dictionary Filtering:
# Description: Filter a dictionary based on a condition.
filtered_dict = {k: v for k, v in my_dict.items() if condition}





### OrderedDict maintains the order of the keys based on the order of their insertion,
# providing a way to iterate over the items in the order they were added.

# Description: Create an OrderedDict using the regular dictionary syntax.
from collections import OrderedDict

ordered_dict = OrderedDict({'a': 1, 'b': 2, 'c': 3})

# Reordering: Change the order of keys using move_to_end method.
ordered_dict.move_to_end('a')  # Move 'a' to the end

# Equality Check: Order matters when checking equality.
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 2, 'a': 1}
print(dict1 == dict2)  # True for regular dict, False for OrderedDict

# OrderedDict vs. Regular Dict: OrderedDict maintains order, while a regular dict does not guarantee order.
regular_dict = {'a': 1, 'b': 2}
ordered_dict = OrderedDict({'a': 1, 'b': 2})

# OrderedDict maintains order
print(list(ordered_dict.keys()))  # ['a', 'b']

# Regular dict order is not guaranteed
print(list(regular_dict.keys()))  # Order can vary

# Initializing from a List of Tuples:
pairs = [('a', 1), ('b', 2), ('c', 3)]
ordered_dict = OrderedDict(pairs)

# Note: Starting from Python 3.7, the built-in dict type also maintains order.
