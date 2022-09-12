# The way Java works
1. Create a java source code, e.g., `Party.java`
2. Compile `Party.java` by running `javac Party.java` to generate a `Party.class` file. `Party.class` is made up of bytecodes, which is not binary but platform-independent. 
3. Run `java Party` to force Java Virtual Machine (JVM) to translate the bytecode in `Party.class` into binary code that the underlying platform understands.

# Code structure in Java
- Put a class in a source file. Note the name of this class and the name of this source file should be the same.
- Put methods in a class
- Put statements in a method

```java
// MyFirstApp.java
// class
public class MyFirstApp {
    //method
    public static void main(String[] args) {
        //statement
        System.out.print("I Rule!");
    }
}
```

# Anatomy of a class
JVM looks for the class that's given at the command line, Then it starts looking for a specially-written `main` method in this class. Every Java application has to have at least one `class` and at least one `main` method per application.

# `System.out.print` vs. `System.out.println`
- `System.out.print` keeps printing yo the same line.
- `System.out.println` inserts a new line after printing its argument.

# Example: Phrase-O-Matic
- Make three lists of words
- Randomly pick one word from each of the three lists
- Print out the result.

```java
// PhraseOMatic.java
public class PhraseOMatic {
    public static void main(String[] args) {
        String[] wordListOne = {"Hello", "Hi", "Hallo"};
        String[] wordListTwo = {"smart", "clever", "handsome"};
        String[] wordListThree = {"Kevin", "Sam", "Winnie"};
        
        int oneLength = wordListOne.length;
        int twoLength = wordListTwo.length;
        int threeLength = wordListThree.length;

        // random() returns a random number between 0 and not-quite-1
        int rand1 = (int) (Math.random() * oneLength);
        int rand2 = (int) (Math.random() * twoLength);
        int rand3 = (int) (Math.random() * threeLength);

        String phrase = wordListOne[rand1] + " " + wordListTwo[rand2] + " " + wordListThree[rand3];

        System.out.println(phrase);
    }
}
```
