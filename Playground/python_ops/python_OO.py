
### Classes and Objects:
# A class is a blueprint for creating objects. Objects are instances of classes.
# Define a class using the class keyword:
class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model

# Create objects from the class:
my_car = Car(make="Toyota", model="Camry")

### Attributes and Methods:
# Attributes are variables that store data within an object
# Methods are functions defined within a class
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        print("Woof!")

# Access attributes and call methods using dot notation:
my_dog = Dog(name="Buddy", age=3)
print(my_dog.name)
my_dog.bark()

### Constructor (__init__ method):
# The __init__ method initializes the object's attributes when an object is created.
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

### Inheritance:
# Inheritance allows a class to inherit attributes and methods from another class.
# Create a derived class that inherits from a base class:
class ElectricCar(Car):
    def __init__(self, make, model, battery_capacity):
        super().__init__(make, model)
        self.battery_capacity = battery_capacity

## Here's an expanded explanation of inheritance in Python:
# Base Class (Parent Class):
# A base class serves as a blueprint for one or more derived classes.
# It defines common attributes and methods that are shared among the derived classes.
class Animal:
    def __init__(self, species):
        self.species = species

    def make_sound(self):
        pass  # Placeholder for a common behavior

# Derived Class (Child Class):
# A derived class inherits from a base class using the class DerivedClassName(BaseClassName):
# The derived class can add new attributes and methods or override existing ones.
# Extending the Base Class:
class Dog(Animal):
    def __init__(self, species, breed):
        super().__init__(species)
        self.breed = breed

    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def __init__(self, species, color):
        super().__init__(species)
        self.color = color

    def make_sound(self):
        return "Meow!"
# In this example, both Dog and Cat are derived from the Animal base class.
# They inherit the species attribute and the make_sound method. Each derived class extends the base class by adding
# specific attributes (breed for Dog, color for Cat) and providing their own implementation of the make_sound method.

# Overriding Methods:
class Bird(Animal):
    def __init__(self, species, wingspan):
        super().__init__(species)
        self.wingspan = wingspan

    def make_sound(self):
        return "Chirp!"

# In this case, the Bird class also inherits from Animal but provides its own implementation of the make_sound method.
# This is an example of method overriding, where the derived class provides a specific implementation for a method already defined in the base class.


## Super() Function:
# The super() function is used to call methods from the base class within the derived class.
# It allows the derived class to invoke the methods of the base class and initialize its attributes.
class Reptile(Animal):
    def __init__(self, species, scale_type):
        super().__init__(species)
        self.scale_type = scale_type

# Types of Inheritance:
# Single Inheritance: A derived class inherits from only one base class.

# Multiple Inheritance: A derived class inherits from more than one base class
class FlyingDog(Dog, Bird):
    def __init__(self, species, breed, wingspan):
        Dog.__init__(self, species, breed)
        Bird.__init__(self, species, wingspan)

## Multilevel Inheritance: A derived class is created from another derived class
class BabyDog(Dog):
    def __init__(self, species, breed, age):
        super().__init__(species, breed)
        self.age = age

## Hierarchical Inheritance: Multiple derived classes inherit from a single base class
class Poodle(Dog):
    def __init__(self, species, size):
        super().__init__(species, "Poodle")
        self.size = size

class Siamese(Cat):
    def __init__(self, species, personality):
        super().__init__(species, "Siamese")
        self.personality = personality


## Benefits of Inheritance:

# Code Reusability:
#     Common functionalities can be defined in a base class and reused in multiple derived classes.
#
# Modularity:
#     Code is organized into smaller, manageable units (classes), making it easier to maintain and understand.
#
# Polymorphism:
#     Different classes can be treated as instances of a common base class, allowing for flexibility and dynamic behavior.




### Encapsulation:
# Encapsulation restricts access to certain attributes or methods, making them private or protected.
# Use a single underscore _ for protected attributes and double underscore __ for private attributes:
class BankAccount:
    def __init__(self, balance):
        self._balance = balance  # Protected attribute

    def _withdraw(self, amount):
        self._balance -= amount  # Protected method

### Polymorphism:
# Polymorphism allows objects of different classes to be treated as objects of a common base class.
# Achieved through method overloading and method overriding.
class Shape:
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2


