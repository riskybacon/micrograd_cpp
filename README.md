# Intro
This is a port of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)
from Python to C++.

The reason I wrote this is because I'd like to increase my knowledge of C++ and
to help me understand more about frameworks like Pytorch.

# Goals
* Learn something about C++ and ML frameworks
* Keep structure as similar to original micrograd as possible
* No external dependencies required
* Train the same model as micrograd
* Avoid pointers when possible

# Build

`make` or `make -DUSE_DOT` if `graphviz` is installed and `dot` is in your path and you want `graph.png` to be written

# What I learned

* How to use references in vectors and sets
* How to capture nested functions for later use
* How to build a simple autograd framework

# Challenges

## Avoiding pointers

The `Value` class represents a node the computational graph. An instance needs
to be able to update it's children during the backward pass. The instance
can't have a copy of the child, it needs to refer to the child.

I've only used pointers in the past:

```c++
vector<Value*> children;
```

But one of the goals is to avoid pointers, so I need this:

```c++
vector<Value&> children;
```

Which is not valid C++. To store references, I used:

```c++
vector<std::reference_wrapper<Value>> children;
```

inside of the value class.

The next challenge arrived when drawing the graph.

To draw the graph, it needs to be traversed. While the graph is directed, it's
not guaranteed to be acyclic. I chose to avoid revisting nodes by placing them
in a set:

```c++
set<reference_wrapper<Value>> nodes;
```

In c++, a set is stored internally as a tree, and the implementation needs to be
able to compare objects with `a < b`. References can't be compared this way.

I added a `RefWrapperComparator` helper to get around this. It uses the address
of the underlying objects for the comparison.

I could have also used an `unordered_set`, which uses a hash table instead.
There's a similar challenge, and I think it should be solvable by passing
in a custom hash function, which I think could return the pointer as an
integer.

I suppose that it could be argued that I didn't avoid pointers because of what
I did.

## Nested functions

In micrograd, the backward function is nested within the operator function.
As I reflected on this, I found I was getting a bit confused about what state
was being captured by the closure and how to achieve this in C++.

I was also unsure about how to create a function variable that could be set
dynamically and then used to call that function without using pointers.


One of my goals is to keep the same structure, so I needed to do the same in
C++.

I solved this using `std::function` and lambdas. Because I chose those, I was
about to avoid using function pointers, and avoiding pointers is another goal
of this exercise.

The state capture turned out to be straighforward and I only needed:
* `out`: the new node in the graph that was being created by the operator
* `other`: the right hand side of the operation
* `this`: the left hand side of the operation

The lambda is then assigned to `out`'s `_backward` member, which is defined as
a `std::function<void()>`

## Graphing

I didn't find traversing the graph to draw it much of a conceptual challenge.
The challenge was building a PNG without requiring external dependencies.

I handled this by building a function that looks a lot like the one in
micrograd, but instead of using `Digraph`, I created a text file that could be
read by `dot` to build the PNG with an external program, that the user can
optionally install.

# Similar projects:

* [micrograd_cpp](https://github.com/Jac-Zac/micrograd_cpp/)
* [cpp-micrograd](https://github.com/10-zin/cpp-micrograd)
* [micrograd-cpp-2023](https://github.com/kfish/micrograd-cpp-2023)
* [milligrad](https://github.com/NerusSkyhigh/milligrad.cpp)