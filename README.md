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

# Design

I faced 4 design challenges:

* Pointerless API
* Nest and capturing the backward function within the operator
* Expose `.label`, `.grad`, etc as members, just like in micrograd
* My DFS topo sort didn't work right

## Pointerless API

I wanted to create an API that was as close to the original micrograd API as
possible, but in C++:

```c++
auto make_model() {
    auto a = Value(-2.0, "a");
    auto b = Value(3.0, "b");
    auto d = a * b;
    auto e = a + b;
    auto f = d * e;
    return f;
}

void main() {
    auto model = make_model();
    model.backward();
    draw_model();
}
```

Notice how there are no pointers.

But a, b, d, e are all allocated on the stack and go out of scope when
`make_model` returns. This requires a mechanism to capture references to `Value`
instances during operations.

I considered using `std::reference_wrapper` to store references in a
`std::vector`, but that will not work. The
`std::vector<std::reference_wrapper<Value>>` will be full of invalid
references when the `make_mode()` returns.

Let's think about what happens in the equivalent Python code:

```python
def make_model():
    a = Value(-2.0, "a")
    b = Value(3.0, "b")
    d = a * b
    e = a + b
    f = d * e
    return f

def main:
    model = make_model()
    model.backward()
}
```

Because of Python's pass-by-reference + reference counting model, the
variables `make_model()` are still available because they're captured and
referenced by `f` and its children.

I accomplished this in C++ by pushing all state into a separate context.
A `Value` instance contains a `std::shared_ptr<Context> ctx_` member, which
can be passed round.

This moves the data to the heap and adds a reference counting mechanism.

## Nested functions

In micrograd, the backward function is nested within the operator function:

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

I wanted to keep this same structure using C++. I accomplished this using
`std::function` and lambdas. This allowed me to avoid using function pointers
 or a big case statement, like I noticed in other C++ ports of micrograd.

The state that needed to be captured:
* `out`: the new node in the graph that was being created by the operator
* `other`: the right hand side of the operation
* `this`: the left hand side of the operation

It's important to reference only objects whose lifetime goes beyond the scope
of the operator function. I made a mistake when I used the locally stored result
of tanh in the closure, which went out of scope and was meaningless. I needed
to use `out.data` instead.

The lambda is assigned to `out`'s `_backward` member, which is defined as
a `std::function<void()>`

```c++
Value operator*(Value &other)
{
    auto out = Value(data * other.data, {*this, other}, "*");
    backward_ = [&out, &other, this]()
    {
        grad += other.data * out.grad;
        other.grad += data * out.grad;
    };
    return out;
}
```

## Adding members in `Value` to reference the context

Because I needed to place all of the Value members into a context, this breaks:

```c++
auto a = Value(2.0)
a.label = "a";   // Error, label is in the embedded context
a.ctx_->label = "a";   // Works, but ugly

// Could do:
a.label() = "a";   // Returns ctx_->label
```

But I wanted to keep the same API as the Python code. I solved this by adding
references to the `Value` struct:

```c++
struct Value {
    shared_ptr<Context> ctx_;
    string &label;

    Value() : ctx_(make_shared<Context>()), label(ctx_->label);
};
```

I'm neither for or against these sorts of shenanigans, it was useful for
achieving my goal of keeping the same API.

## Backward Pass

I replaced the DFS-based topological sort with BFS. I was getting errors with
DFS that were fixed with BFS.

# Similar projects

* [micrograd_cpp](https://github.com/Jac-Zac/micrograd_cpp/)
* [cpp-micrograd](https://github.com/10-zin/cpp-micrograd)
* [micrograd-cpp-2023](https://github.com/kfish/micrograd-cpp-2023)
* [milligrad](https://github.com/NerusSkyhigh/milligrad.cpp)