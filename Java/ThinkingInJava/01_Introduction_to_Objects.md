# Introduction to Objects
## The progress of abstraction
1. Everything is an object.
2. A program is a bunch of objects telling each other what to do by sending messages.
3. Each object has its own memory made up of other objects.
4. Every object has a type / class.
5. All objects of a particular type can receive the same messages.

## An object has an interface
...
## An object provides services
...
## The hidden implementation
- **Client programmers**, those who consumes classes in their applications, would collect a toolbox full of classes to use for rapid application development.
- **Class creators**, those who create new data types, should build a class that ***exposes only what's necessary to the client programmer and keeps everything else hidden***.

Java uses three explicit keywords to set the boundaries in a class: `public`, `private` and `protected`.

- `public` means the following element is available to everyone.
- `private` means that no one can access that element except the class creator. In other words, only methods of that type can access that element.
- `protected` allows its inheriting class to have access to that `protected` member
- default access, also called **package access** (when there are no specifiers): members can be accessed by other classes in the same package, but outside of the package those same members appear to be `private`.

## Reusing the implementation
- **Composition**: place an object of one class inside a new class
- **Aggregation**: dynamic composition

## Inheritance
- Inheritance expresses the similarity between types by using the concept of base types and derived types.
- If you simply inherit a class and don't do anything else, the methods from the base-class interface **come right along into** the derived class.
- To differentiate derived class from the base class:
  - Add brand new methods that are not part of the base-class interface.
  - To override a method, you simply create a new definition for the method in the derived class.

## Is-a vs. is-like-a relationships
- **Is-a** means that the derived class ***has exactly the same interface*** as the base class.
- **Is-like-a** means that the derived class is not a perfect substitution when new methods are added to the derived class.

## Interchangeable objects with polymorphism
- Treating an object not as the specific type that it is, but instead as its base type, allows you to write code that doesn't depend on specific types.
- **Early binding**: P26