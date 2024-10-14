#include <iostream>
#include <iomanip>

#include <micrograd/engine.hpp>
#include <micrograd/nn.hpp>

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

template <typename T>
void not_equal_helper(const T &a, const T &b, const char *file, const int line)
{
    if (a == b)
    {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << a << " == " << b << std::endl;
    }
}

#define is_close(a, b) is_close_helper(a, b, __FILE__, __LINE__)
#define is_equal(a, b) is_equal_helper(a, b, __FILE__, __LINE__)
#define not_equal(a, b) not_equal_helper(a, b, __FILE__, __LINE__)

void test_instantiate()
{
    auto a = Value(-2.0, "a");
    auto b = Value(5.0, "b");
    auto c = a + b;
    c.label() = "c";

    is_close(a.data(), -2.0);
    is_close(a.grad(), 0.0);
    is_equal(a.label(), "a");

    is_close(b.data(), 5.0);
    is_close(b.grad(), 0.0);
    is_equal(b.label(), "b");

    is_close(c.data(), 3.0);
    is_close(c.grad(), 0.0);
    is_equal(c.label(), "c");
}

void test_add()
{
    auto a = Value(-2.0, "a");
    auto b = Value(5.0, "b");
    auto c = a + b;
    c.label() = "c";
    c.backward();

    is_close(a.grad(), 1.0);
    is_close(b.grad(), 1.0);
    is_close(c.data(), 3.0);
    is_close(c.grad(), 1.0);

    is_equal(a.ctx_, c.ctx_->prev[0]);
    is_equal(b.ctx_, c.ctx_->prev[1]);
}

void test_plus_equal()
{
    auto a = Value(1.0, "a_orig");
    auto b = Value(2.0, "b");
    auto a_orig_ctx = a.ctx_;

    a += b;
    a.label() = "a_new";

    not_equal(a_orig_ctx.get(), a.ctx_.get());
    is_equal(a.ctx_->prev[0].get(), a_orig_ctx.get());
    is_equal(a.ctx_->prev[0]->label, "a_orig");
    is_equal(a.ctx_->prev[1].get(), b.ctx_.get());
    is_equal(a.ctx_->prev[1]->label, "b");
    is_close(a.data(), 3.0);

    a.backward();

    is_close(a_orig_ctx->grad, 1.0);
    is_close(b.ctx_->grad, 1.0);

    auto c = Value(2.0, "c");
    c += Value(3.0, "c");
    is_close(c.data(), 5);
}

void test_sub()
{
    auto a = Value(12.0, "a");
    auto b = Value(2.0, "b");
    auto c = a - b;
    c.label() = "c";
    c.backward();

    is_close(a.grad(), 1.0);
    is_close(b.grad(), -1.0);
    is_close(c.data(), 10.0);
    is_close(c.grad(), 1.0);
}

void test_sub_equal()
{
    auto a = Value(100.0, "a_orig");
    auto b = Value(1.0, "b");
    auto a_orig_ctx = a.ctx_;

    a -= b;
    a.label() = "a_new";

    not_equal(a_orig_ctx.get(), a.ctx_.get());
    is_equal(a.ctx_->prev[0].get(), a_orig_ctx.get());
    is_equal(a.ctx_->prev[0]->label, "a_orig");
    is_equal(a.ctx_->prev[1].get(), b.ctx_.get());
    is_equal(a.ctx_->prev[1]->label, "b");
    is_close(a.data(), 3.0);

    a.backward();

    is_close(a_orig_ctx->grad, 1.0);
    is_close(b.ctx_->grad, 1.0);

    auto c = Value(2.0, "c");
    c += Value(3.0, "c");
    is_close(c.data(), 5);
}

void test_mul()
{
    auto a = Value(-2.0, "a");
    auto b = Value(5.0, "b");
    auto c = a * b;
    c.label() = "c";
    c.backward();

    is_close(a.grad(), 5.0);
    is_close(b.grad(), -2.0);
    is_close(c.data(), -10.0);
    is_close(c.grad(), 1.0);
}

void test_mul_equal()
{
    auto a = Value(10.0, "a_orig");
    auto b = Value(-2.0, "b");
    auto a_orig_ctx = a.ctx_;

    a *= b;
    a.label() = "a_new";

    not_equal(a_orig_ctx.get(), a.ctx_.get());

    is_equal(a.ctx_->prev[0].get(), a_orig_ctx.get());
    is_equal(a.ctx_->prev[0]->label, "a_orig");
    is_equal(a.ctx_->prev[1].get(), b.ctx_.get());
    is_equal(a.ctx_->prev[1]->label, "b");
    is_close(a.data(), -20.0);

    a.backward();

    is_close(a_orig_ctx->grad, b.data());
    is_close(b.grad(), a_orig_ctx->data);

    auto c = Value(2.0, "c");
    c *= Value(3.0, "c");
    is_close(c.data(), 6);
}

void test_div()
{
    auto a = Value(-100, "a");
    auto b = Value(2, "b");
    auto c = a / b;

    is_close(c.data(), -50);

    c.backward();

    is_close(a.grad(), 1.0 / b.data());
    is_close(b.grad(), -1 * std::pow(b.data(), -2) * a.data());

    auto d = Value(24, "d");
    auto d_orig_ctx = d.ctx_;
}

void test_tanh()
{
    for (float v = -100; v <= 100; v += 0.5)
    {
        auto a = Value(v);
        auto b = a.tanh();
        auto out = std::tanh(v);
        is_close(b.data(), out);
        b.backward();
        is_equal(a.grad(), (1 - out * out));
    }
}

void test_exp()
{
    for (float v = -100; v <= 100; v += 0.5)
    {
        auto a = Value(v);
        auto b = a.exp();
        auto out = std::exp(v);
        is_close(b.data(), out);
        b.backward();
        is_equal(a.grad(), b.data() * b.grad());
    }
}

void test_pow()
{
    for (float v = -100; v <= 100; v += 0.5)
    {
        for (float p = -10; p <= 10; p += 0.5)
        {
            auto a = Value(v);
            auto b = a.pow(p);
            auto out = std::pow(v, p);
            is_close(b.data(), out);
            b.backward();
            is_close(a.grad(), p * std::pow(v, p - 1));
        }
    }
}

void test_dot()
{
    size_t size = 10;
    std::vector<Value> a;
    std::vector<Value> b;

    for (size_t i = 0; i < size; i++)
    {
        a.push_back(Value(3));
        b.push_back(Value(2));
        a[i].label() = "a[" + std::to_string(i) + "]";
        b[i].label() = "b[" + std::to_string(i) + "]";
    }

    auto c = dot(a, b);
    c.backward();

    is_close(c.data(), a[0].data() * b[0].data() * size);
}

template <typename T>
void test_to_values()
{
    auto x = {static_cast<T>(2.0), static_cast<T>(3.0), static_cast<T>(-1.0)};
    auto xv = to_values({static_cast<T>(2.0), static_cast<T>(3.0), static_cast<T>(-1.0)});
    is_equal(x.size(), xv.size());

    size_t i = 0;
    for (auto xi : x)
    {
        is_equal(static_cast<float>(xi), xv[i].data());
        i++;
    }
}

void test_expr()
{
    auto a = Value(2.0f, "a");
    auto b = Value(-3.0f, "b");
    auto c = Value(10.0f, "c");
    auto e = a * b;
    e.label() = "e";
    auto d = e + c;
    d.label() = "d";
    auto f = Value(-2.0f, "f");
    auto L = d * f;
    L.label() = "L";

    L.backward();

    is_close(a.grad(), 6);
    is_close(b.grad(), -4);
    is_close(c.data(), 10);
    is_close(c.grad(), -2);
    is_close(d.data(), 4);
    is_close(d.grad(), -2);
    is_close(e.data(), -6);
    is_close(e.grad(), -2);
    is_close(f.grad(), 4);
    is_close(L.data(), -8.0);
    is_close(L.grad(), 1);
}

void test_neuron()
{
    size_t nin = 10;
    auto a = Neuron(nin);

    std::vector<Value> x;
    for (size_t i = 0; i < nin; i++)
    {
        x.push_back(Value(2.0));
        x[i].label() = "x[" + std::to_string(i) + "]";
        a.w[i].data() = 3.0;
    }
    a.b.data() = 1.0;

    is_equal(a.w.size(), nin);

    auto out = a(x);

    auto expected = x.front().data() * a.w.front().data() * nin + a.b.data();
    is_close(out.data(), std::tanh(expected));
}

void test_layer()
{
    // Not a real test, it just;
    // 1. Makes sure that it compiles
    // 2. Makes sure that it runs without aborting

    size_t nin = 3;
    size_t nout = 4;
    auto layer = Layer(nin, nout);
    std::vector<Value> x;

    for (size_t i = 0; i < nin; i++)
    {
        x.push_back(Value(1.0, "x[" + std::to_string(i) + "]"));
    }

    auto y = layer(x);

    is_equal(y.size(), nout);
}

void test_mlp()
{
    auto xv = {2.0f, 3.0f, -1.0f};
    auto x = to_values(xv);
    auto n = MLP(3, {4, 4, 1});
    auto o = n(x);
}

int main()
{
    test_instantiate();
    test_add();
    test_plus_equal();
    test_sub();
    test_mul();
    test_mul_equal();
    test_div();
    test_tanh();
    test_exp();
    test_pow();
    test_to_values<float>();
    test_to_values<double>();
    test_to_values<int>();
    test_to_values<size_t>();
    test_dot();
    test_expr();
    test_neuron();
    test_layer();
    test_mlp();
    return 0;
}