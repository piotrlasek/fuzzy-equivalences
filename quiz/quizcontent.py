# Sample data that would be processed
quiz_content = """
# Question 1
## What does the `flatMap` method do in Java Streams?

1. Maps each element to a single result element
2. Filters out elements based on a condition
3. Maps each element to multiple elements and flattens the result
4. Reduces the elements of the stream into a single summary result

* Correct answer: 3

# Question 2
## What is the purpose of the `WatchService` API in Java?

1. To modify file attributes
2. To monitor file system events like file creation, modification, or deletion
3. To list all files in a directory
4. To increase file system security

* Correct answer: 2

# Question 3
## Which method reference can be used to sort a list of strings in alphabetical order?

1. String::toUpperCase
2. String::compareTo
3. String::length
4. String::toLowerCase

* Correct answer: 2

# Question 4
## How do you collect stream results into a list?

1. collect(Collectors.toMap())
2. collect(Collectors.toList())
3. collect(Collectors.toSet())
4. collect(Collectors.groupingBy())

* Correct answer: 2

# Question 5
## Which lambda expression correctly checks if a number is even?

1. n -> n % 2
2. n -> n % 2 != 0
3. n -> n % 2 == 0
4. n -> n = 2

* Correct answer: 3

# Question 6
## What is used to create a new `WatchService`?

1. File.newWatchService()
2. FileSystem.getWatchService()
3. Path.newWatchService()
4. FileSystem.newWatchService()

* Correct answer: 4

# Question 7
## How can you sort a list of `Person` objects by age using a method reference?

1. Collections.sort(people, Person::getAge)
2. Collections.sort(people, Comparator.comparing(Person::getAge))
3. Collections.sort(people, Person::compareTo)
4. Collections.sort(people, Comparator.comparingInt(Person::getAge))

* Correct answer: 2

# Question 8
## Which operation on Stream can be parallelized?

1. collect()
2. map()
3. forEach()
4. All of the above

* Correct answer: 4

# Question 9
## What is the output type of the `filter` method in a Stream?

1. Object
2. Stream
3. Boolean
4. Integer

* Correct answer: 2

# Question 10
## How do you transform a list of `String` to uppercase using Streams?

1. list.map(String::toUpperCase)
2. list.stream().map(String::toUpperCase).collect(Collectors.toList())
3. list.toUpperCase()
4. list.stream().toUpperCase().collect(Collectors.toList())

* Correct answer: 2

# Question 11
## What does `Files.list(Path dir)` return?

1. Stream<Path>
2. List<Path>
3. Set<Path>
4. Path[]

* Correct answer: 1

# Question 14
## How do you get the file system of a Path object?

1. Path.getFileSystem()
2. Path.toFileSystem()
3. Path.fileSystem()
4. Path.system()

* Correct answer: 1

# Question 15
## Which of the following is NOT a terminal operation?

1. map()
2. forEach()
3. reduce()
4. collect()

* Correct answer: 1

# Question 16
## What does `Files.isDirectory(Path path)` check?

1. If the path is hidden
2. If the path is readable
3. If the path is a directory
4. If the path is executable

* Correct answer: 3

# Question 19
## Which functional interface must be implemented to use with `Files.newDirectoryStream(Filter<Path> filter)`?

1. Predicate<Path>
2. Supplier<Path>
3. Consumer<Path>
4. DirectoryStream.Filter<Path>

* Correct answer: 4

# Question 20
## What is a valid way to create a `Stream<Path>` from a directory?

1. Path.getStream()
2. Files.walk(Path)
3. Files.list(Path)
4. Path.stream()

* Correct answer: 3

# Question 1
## What does a Predicate interface in Java return?

1. An object
2. A boolean value
3. An integer
4. A string

* Correct answer: 2

# Question 2
## Which method is used to filter elements in a stream?

1. filter
2. map
3. reduce
4. collect

* Correct answer: 1

# Question 3
## What does the map function in a stream do?

1. Filters elements based on a condition
2. Transforms each element in the stream
3. Combines elements of the stream
4. Joins elements into a string

* Correct answer: 2

# Question 4
## What does the reduce method primarily do?

1. Filters specific elements
2. Maps values to keys
3. Accumulates elements into a single summary result
4. Splits a string into an array

* Correct answer: 3

# Question 5
## Which Java class is used to concatenate strings with a delimiter?

1. StringBuilder
2. StringSplitter
3. StringJoiner
4. StringCombiner

* Correct answer: 3

# Question 6
## What is the result of calling Optional's ifPresent() method?

1. It returns true if a value is present.
2. It executes a block of code if the value is present.
3. It combines two Optional objects.
4. It throws an exception if the value is not present.

* Correct answer: 2

# Question 7
## Which feature introduced in Java 8 greatly facilitates functional programming?

1. Abstract classes
2. Lambda expressions
3. Static methods
4. Generics

* Correct answer: 2

# Question 8
## What is the purpose of the Collectors class in Java?

1. To group elements in a stream
2. To split arrays into smaller arrays
3. To lock elements in multi-threaded operations
4. To implement encryption algorithms

* Correct answer: 1

# Question 9
## How can you create a stream from a collection?

1. Using the Stream.of() method
2. Using the collect() method
3. Using the stream() method on the collection
4. Using the new Stream() constructor

* Correct answer: 3

# Question 10
## Which of these is a terminal operation in the Stream API?

1. map
2. filter
3. forEach
4. flatMap

* Correct answer: 3

# Question 11
## Which method converts elements of a stream to int before performing operations?

1. mapToInt
2. toInt
3. asInt
4. getInt

* Correct answer: 1

# Question 12
## How do you find the sum of numbers in a list using streams?

1. list.stream().sum()
2. list.stream().reduce(0, Integer::sum)
3. list.stream().collect(Collectors.summingInt(Integer::intValue))
4. list.sum()

* Correct answer: 2

# Question 13
## What is the output of String.join(", ", "apple", "banana")?

1. applebanana
2. "apple", "banana"
3. apple, banana
4. apple-banana

* Correct answer: 3

# Question 14
## How do you obtain the maximum element from a stream of integers?

1. stream.max(Integer::compare)
2. stream.reduce(Integer::max)
3. stream.collect(Collectors.maxBy())
4. stream.max()

* Correct answer: 2

# Question 15
## Which of these is not a feature of lambda expressions in Java?

1. They can be used to implement any interface.
2. They allow functions to be passed as arguments.
3. They can access final and effectively final variables from the enclosing scope.
4. They simplify the use of the Stream API.

* Correct answer: 1

# Question 16
## What does Optional class represent?

1. A container object which may or may not contain a non-null value.
2. A collection that optionally allows duplicate values.
3. A data type for optional method parameters.
4. An Optional interface for implementing optional methods.

* Correct answer: 1

# Question 17
## What happens when a stream operation is parallelized?

1. The order of elements is guaranteed.
2. Operations are performed in multiple threads.
3. Only immutable objects can be processed.
4. The stream becomes sequential.

* Correct answer: 2

# Question 18
## Which of the following correctly creates a List from a Stream?

1. stream.toList()
2. stream.collect(Collectors.toList())
3. stream.asList()
4. List.from(stream)

* Correct answer: 2

# Question 19
## Which operation can be used to combine two streams into one?

1. concat
2. merge
3. join
4. unite

* Correct answer: 1

# Question 20
## What is a characteristic of immutable objects that is beneficial for functional programming?

1. They can change state.
2. They ensure thread safety.
3. They support inheritance.
4. They allow null values.

* Correct answer: 2


# Question 1
## What do lambda expressions in Java primarily offer to developers?

1. A more verbose way of writing code
2. A concise and expressive way to write code
3. Increased complexity in code
4. A new set of data types

* Correct answer: 2

# Question 2
## What do lambda expressions help reduce?

1. The essence of operations
2. The clarity of programming logic
3. The verbosity associated with anonymous inner classes
4. The need for external libraries

* Correct answer: 3

# Question 3
## What is the main purpose of functional interfaces in Java?

1. To provide multiple abstract methods
2. To define the target for lambda conversions with a single abstract method
3. To increase the verbosity of the code
4. To replace all traditional interfaces

* Correct answer: 2

# Question 4
## How do lambda expressions enhance Java's capabilities?

1. By reducing Java's expressivity
2. By simplifying the maintenance of complex codebases
3. By making Java less intuitive
4. By adding unnecessary complexity

* Correct answer: 2

# Question 5
## What role does the `@FunctionalInterface` annotation play?

1. It decreases the expressivity of interfaces
2. It makes the intent behind interfaces ambiguous
3. It ensures interfaces adhere to the functional interface contract
4. It enforces the use of multiple abstract methods in interfaces

* Correct answer: 3

# Question 6

## What benefit does the `@FunctionalInterface` annotation provide to developers?

1. Slows down the compilation process
2. Makes it harder to understand the interface's purpose
3. Helps quickly understand the interface's purpose
4. Requires additional libraries to function

* Correct answer: 3

# Question 7
## How do lambda expressions interact with existing library methods?

1. They are incompatible with existing library methods
2. They simplify the syntax for more readable code
3. They prevent the use of method references
4. They increase the need for verbose anonymous inner classes

* Correct answer: 2

# Question 8
## What does the introduction of lambda expressions signify in Java's evolution?

1. A step back in language development
2. A minor update with no real impact
3. A significant advancement
4. A replacement for object-oriented programming

* Correct answer: 3

# Question 9
## What does focusing on the essence of operations enable in programming?

1. Increased verbosity
2. More complex code
3. A clearer representation of programming logic
4. Less expressive code

* Correct answer: 3

# Question 10
## What allows developers to use lambda expressions instead of traditional interface implementations?

1. The complexity of lambda expressions
2. The simplicity provided by functional interfaces
3. The limitations of Java
4. The removal of interfaces from Java

* Correct answer: 2

# Question 11
## How does the `@FunctionalInterface` annotation affect the compiler?

1. It confuses the compiler
2. It allows the compiler to ignore interfaces
3. It helps the compiler enforce functional interface constraints
4. It makes the compilation process longer

* Correct answer: 3

# Question 12
## How does the shift towards a functional style impact Java's code expression?

1. Makes it less intuitive
2. Reduces its straightforwardness
3. Improves its ability to express complex logic in an intuitive manner
4. Decreases readability and maintainability

* Correct answer: 3

# Question 13
## What type of methods do functional interfaces have?

1. Multiple abstract methods
2. No abstract methods
3. One abstract method
4. Only static methods

* Correct answer: 3

# Question 14
## What can developers pass to methods that take a functional interface as a parameter?

1. Only anonymous inner classes
2. Only existing library methods
3. Lambda expressions or method references
4. Complex objects only

* Correct answer: 3

# Question 15
## What happens when a lambda expression is passed to a method?

1. The compiler ignores it
2. It is converted into an instance of the appropriate functional interface
3. It causes a runtime error
4. It is discarded for being inefficient

* Correct answer: 2

# Question 16
## What must match between the lambda expression and the functional interface's abstract method?

1. The name of the method
2. The parameters
3. The return type only
4. The number of static methods

* Correct answer: 2

# Question 17
## What may the synthesized method do if the return type of the lambda expression doesn't match the abstract method?

1. It may convert the return value to a proper assignable type
2. It will throw a compile-time error
3. It ignores the lambda expression
4. It changes the abstract method to match

* Correct answer: 1

# Question 18
## What is the key feature of lambda expressions regarding method calls?

1. They cannot be used in method calls
2. They can be passed as arguments to methods
3. They replace all methods in Java
4. They are only decorative and have no real use

* Correct answer: 2

# Question 19
## How do functional interfaces and lambda expressions improve Java applications?

1. By making them more complex and verbose
2. By enhancing readability, maintainability, and abstraction
3. By removing the ability to use collections
4. By enforcing the use of outdated programming practices

* Correct answer: 2

# Question 20
## What new programming style does modern Java embrace with the introduction of lambda expressions?

1. Imperative programming
2. Procedural programming
3. Functional programming
4. Assembly language programming

* Correct answer: 3
"""