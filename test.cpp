#include <iomanip>
#include <iostream>
#include <sstream>

#include <micrograd/engine.hpp>
#include <micrograd/graphviz.hpp>

void is_close_helper(float a, float b, const char *file, const int line)
{
    float epsilon = 1e-6;
    float hi = b + epsilon;
    float lo = b - epsilon;

    if (a < lo || a > hi)
    {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << std::fixed << std::setprecision(10) << a << " != " << b << std::endl;
    }
}

template <typename T>
void is_equal_helper(const T &a, const T &b, const char *file, const int line)
{
    if (a != b)
    {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << a << " != " << b << std::endl;
    }
}

void is_equal_helper(const std::string &a, const char *b, const char *file, const int line)
{
    if (a != b)
    {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << a << " != " << b << std::endl;
    }
}

#define is_close(a, b) is_close_helper(a, b, __FILE__, __LINE__)
#define is_equal(a, b) is_equal_helper(a, b, __FILE__, __LINE__)

void test1(void)
{
    auto a = Value(2.0f, "a");
    auto b = Value(-3.0f, "b");
    auto c = Value(10.0f, "c");
    auto e = a * b;
    e.label = "e";
    auto d = e + c;
    d.label = "d";
    auto f = Value(-2.0f, "f");
    auto L = d * f;
    L.label = "L";

    L.backward();

    is_close(a.data, 2.0);
    is_close(a.grad, 6);
    is_equal(a.label, "a");

    is_close(b.data, -3.0f);
    is_close(b.grad, -4);
    is_equal(b.label, "b");

    is_close(c.data, 10);
    is_close(c.grad, -2);
    is_equal(c.label, "c");

    is_close(d.data, 4);
    is_close(d.grad, -2);
    is_equal(d.label, "d");

    is_close(e.data, -6);
    is_close(e.grad, -2);
    is_equal(e.label, "e");

    is_close(f.data, -2);
    is_close(f.grad, 4);
    is_equal(f.label, "f");

    is_close(L.data, -8.0);
    is_close(L.grad, 1);
    is_equal(L.label, "L");
}

void test2()
{
    // Inputs
    auto x1 = Value(2.0, "x1");
    auto x2 = Value(0.0, "x2");

    // Weights
    auto w1 = Value(-3.0, "w1");
    auto w2 = Value(1.0, "w2");

    // Bias
    auto b = Value(6.8813735870195432, "b");

    auto x1w1 = x1 * w1;
    x1w1.label = "x1*w1";
    auto x2w2 = x2 * w2;
    x2w2.label = "x2*w2";

    auto x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2.label = "x1w1 + x2w2";

    auto n = x1w1x2w2 + b;
    n.label = "n";

    auto o = n.tanh();
    o.label = "o";

    o.backward();

    is_close(x1.data, 2.0);
    is_close(x1.grad, -1.5);

    is_close(x2.data, 0.0);
    is_close(x2.grad, 0.5);

    is_close(w1.data, -3.0);
    is_close(w1.grad, 1.0);

    is_close(w2.data, 1.0);
    is_close(w2.grad, 0.0);

    is_close(b.data, 6.8813735870195432);
    is_close(b.grad, 0.5);

    is_close(x1w1.data, -6);
    is_close(x1w1.grad, 0.5);

    is_close(x2w2.data, 0);
    is_close(x2w2.grad, 0.5);

    is_close(x1w1x2w2.data, -6);
    is_close(x1w1x2w2.grad,  0.5);

    is_close(n.data, 0.8813734055);
    is_close(n.grad, 0.5);

    is_close(o.data, 0.7071067095);
    is_close(o.grad, 1.0);
}

void test3()
{
    auto a = Value(3.0, "a");
    auto b = a + a;

    b.backward();

    is_close(a.data, 3.0);
    is_close(a.grad, 2.0);
    is_close(b.data, 6);
    is_close(b.grad, 1);
}

void test4()
{
    auto a = Value(-2.0, "a");
    auto b = a + 1;
    is_close(b.data, -1);

    auto c = 2 + a;
    is_close(c.data, 0);
}

void test5()
{
    auto a = Value(-2.0, "a");
    auto b = a * 2;
    is_close(b.data, -4);

    auto c = 2. * a;
    is_close(c.data, -4);
}

int main(void)
{
    test1();
    test2();
    test3();
    test4();
    test5();
    return 0;
}