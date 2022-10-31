## You manipulate objects with references
- Although you treat everything as an object, the identifier you manipulate is actually a **reference** to an object.
- A reference **doesn't have to be** connected to an object:
  ```java
  // You've only created a reference, not an object
  String s; // s isn't actually attached to anything
  ```

## You must create all the objects
Connect the reference with a new object by using the `new` operator.
```java
String s = new String("asdf");
String s = "asdf"; // also correct, but this is a special feature for strings
```
### Where storage lives
1. **Registers** insider the processor: the fastest storage, but number of registers is severely limited. No direct control in Java.
2. **The stack** in the RAM area: the second fastest storage. It has direct support from the processor via its stack pointer. The stack pointer is moved down to create new memory and moved up to release that memory. The exact lifetime of the items stored on the stack has to be known. So only object references may be stored on the stack, objects themselves are not placed on the stack.
3. **The heap** also in the RAM area: 