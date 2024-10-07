# Intro
This is a port of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd)
from Python to C++.

I wrote this to increase my knowledge of C++ and to understand more about
frameworks like Pytorch.

# Goals
* Learn more about C++ and ML frameworks
* Keep the structure and API as close as possible to original micrograd
* No external dependencies required
* Train the same model as micrograd

# Build and run:

```
make
./main
```

To run without invoking graphviz:

```
make EXTRA_CXXFLAGS="-DNO_GRAPHVIZ"
./main
```

# What I learned

* How to make a Python-like pointerless API in C++
* How to capture nested functions for later use
* How to build a simple autograd framework in C++

# Challenges

* Pointerless API
* Nesting and capturing the backward function within the operator
* Making members in `Value` reference the context
* DFS topo sort didn't work right

See [DESIGN](DESIGN.md) for how these challenges were handled.

# Similar projects

* [micrograd_cpp](https://github.com/Jac-Zac/micrograd_cpp/)
* [cpp-micrograd](https://github.com/10-zin/cpp-micrograd)
* [micrograd-cpp-2023](https://github.com/kfish/micrograd-cpp-2023)
* [milligrad](https://github.com/NerusSkyhigh/milligrad.cpp)