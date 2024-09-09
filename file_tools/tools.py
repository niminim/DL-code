import sys

def get_size(obj, seen=None):
    """Recursively finds the size of objects, including nested elements."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Avoid counting the same object multiple times
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(vars(obj), seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size


# Example usage:
my_dict = {'a': [1, 2, 3], 'b': {'nested_key': 'nested_value'}, 'c': 42}
size_in_bytes = get_size(my_dict)
size_in_megabytes = size_in_bytes / (1024 ** 2)
print(f"Size of dictionary in MB: {size_in_megabytes:.6f} MB")
####

from pympler import asizeof
# should be more accurate
# Example usage:
my_dict = {'a': [1, 2, 3], 'b': {'nested_key': 'nested_value'}, 'c': 42}
size_in_bytes = asizeof.asizeof(my_dict)
size_in_megabytes = size_in_bytes / (1024 ** 2)
print(f"Size of dictionary in MB: {size_in_megabytes:.6f} MB")
###############################