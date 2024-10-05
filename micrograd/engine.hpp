#pragma once
#include <cmath>
#include <functional>
#include <queue>
#include <ranges>
#include <set>
#include <string>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <iostream>

struct Value;

template<typename T>
struct IdHasher
{
    auto operator()(const T &v) const
    {
        return v.id();
    }
};

template<typename T>
struct IdEqual
{
    auto operator()(const T &a, const T &b) const
    {
        return a.id() == b.id();
    }
};

/**
 * Holds the all of the data for a Value. This will be used for placing Value
 * data on the heap and will be referred to with a shared_ptr
 */
struct Context
{
    using value_type = float;
    value_type data = 0;
    value_type grad = 0;
    std::string label;
    std::string op;
    std::function<void()> backward = []()
    {
        // std::cout << "nop backward for a=" << repr() << std::endl;
    };

    Context(value_type data) : data(data) {}
    Context(value_type data, const std::string &label) : data(data), label(label) {}
    Context(value_type data, const std::string &label, const std::string &op) : data(data), label(label), op(op) {}

    const size_t id() const
    {
        return reinterpret_cast<size_t>(this);
    }

    std::string repr() const
    {
        std::stringstream ss;
        ss << "(" << label << "," << this << ", data=" << data << ", grad=" << grad << ")";
        return ss.str();
    }
};

struct Value
{
    using value_type = Context::value_type;
    std::shared_ptr<Context> ctx_;
    std::vector<Value> prev_;

    // Convenience members to make updating context easy and to keep the same api as original micrograd
    value_type &data;
    value_type &grad;
    std::string &label;
    std::string &op;
    std::function<void()> &backward_;

    const size_t id() const
    {
        // Two Value instances that reference the same context will have the same id
        return ctx_->id();
    }

    Value(const value_type &data)
        : ctx_(std::make_shared<Context>(data)), data(ctx_->data), grad(ctx_->grad), label(ctx_->label), op(ctx_->op), backward_(ctx_->backward)
    {
        // ctx_->data = data;
        // std::cout << "Created " << id() << std::endl;
    }

    Value(const value_type &data, const std::string &label)
        : ctx_(std::make_shared<Context>(data, label)), data(ctx_->data), grad(ctx_->grad), label(ctx_->label), op(ctx_->op), backward_(ctx_->backward)
    {
        // ctx_->data = data;
        // ctx_->label = label;
        // std::cout << "Created " << id() << ", label=" << ctx_->label << std::endl;
    }

    Value(const value_type &data, std::vector<Value> &prev)
        : ctx_(std::make_shared<Context>(data)), prev_(prev), data(ctx_->data), grad(ctx_->grad), label(ctx_->label), op(ctx_->op), backward_(ctx_->backward)
    {
        // ctx_->data = data;
        // std::cout << "Created " << id() << ", prev=";
        // for (const auto &child : prev_) {
        //     std::cout << "(" << child.id() << ", label=" << child.label() << ") ";
        // }
        // std::cout << std::endl;
    }

    Value(const value_type &data, std::vector<Value> &prev, const std::string &op)
        : ctx_(std::make_shared<Context>(data, "", op)), prev_(prev), data(ctx_->data), grad(ctx_->grad), label(ctx_->label), op(ctx_->op), backward_(ctx_->backward)
    {
        // ctx_->data = data;
        // ctx_->op = op;
        // std::cout << "Created " << id() << ", op=" << op << ", prev=";
        // for (const auto &child : prev_) {
        //     std::cout << "(" << child.id() << ", label=" << child.label() << ") ";
        // }
        // std::cout << std::endl;
    }

    Value(const value_type &data, std::initializer_list<Value> &&prev)
        : ctx_(std::make_shared<Context>(data)), prev_(prev), data(ctx_->data), grad(ctx_->grad), label(ctx_->label), op(ctx_->op), backward_(ctx_->backward)
    {
        // ctx_->data = data;
        // std::cout << "Created " << id() << ", prev=";
        // for (const auto &child : prev_) {
        //     std::cout << "(" << child.id() << ", label=" << child.label() << ") ";
        // }
        // std::cout << std::endl;
    }

    Value(const value_type &data, std::initializer_list<Value> &&prev, const std::string &op)
        : ctx_(std::make_shared<Context>(data, "", op)), prev_(prev), data(ctx_->data), grad(ctx_->grad), label(ctx_->label), op(ctx_->op), backward_(ctx_->backward)
    {
        // ctx_->data = data;
        // ctx_->op = op;
        // std::cout << "Created " << id() << ", op=" << op << ", prev=";
        // for (const auto &child : prev_) {
        //     std::cout << "(" << child.id() << ", label=" << child.label() << ") ";
        // }
        // std::cout << std::endl;
    }

    std::string repr() const
    {
        return ctx_->repr();
    }

    Value operator+(Value &other)
    {
        auto &a = *this;
        auto &b = other;

        auto out = Value(a.data + b.data, {a, b}, "+");
        // std::cout << "forward for a=" << a.repr() << " + b=" << b.repr() << ", out=" << out.repr() << std::endl;
        backward_ = [&out, &other, this]()
        {
            grad += out.grad;
            other.grad += out.grad;
            // std::cout << "+: backward for a=" << a.repr() << ", b=" << b.repr() << ", out=" << out.repr() << std::endl;
        };
        return out;
    }

    Value operator+(value_type other)
    {
        auto o = Value(other);
        return *this + o;
    }

    Value operator-()
    {
        auto neg = Value(-1.0f);
        return *this * neg;
    }

    Value operator-(Value &other)
    {
        auto neg = -other;
        return *this + neg;
    }

    Value operator*(Value &other)
    {
        auto &a = *this;
        auto &b = other;

        auto out = Value(a.data * b.data, {a, b}, "*");
        // std::cout << "forward for a=" << a.repr() << " * b=" << b.repr() << ", out=" << out.repr() << std::endl;
        backward_ = [&out, &a, &b]()
        {
            a.grad += b.data * out.grad;
            b.grad += a.data * out.grad;
            // std::cout << "*: backward for a=" << a.repr() << ", b=" << b.repr() << ", out=" << out.repr() << std::endl;
        };
        return out;
    }

    Value operator*(value_type other)
    {
        auto o = Value(other);
        return *this * o;
    }

    Value operator/(Value &other)
    {
        auto b = other.pow(-1);
        return *this * b;
    }

    Value operator/(const value_type &other)
    {
        auto b = Value(other).pow(-1);
        return *this * b;
    }

    Value tanh()
    {
        value_type e2x = std::exp(2 * data);
        value_type th = (e2x - 1) / (e2x + 1);
        Value out(th, {*this}, "tanh");
        // std::cout << "forward for a=" << repr() << " tanh, out=" << out.repr() << ", out.id()=" << out.id() << std::endl;

        backward_ = [&out, this]()
        {
            grad += (1 - out.data * out.data) * out.grad;
            // std::cout << "tanh: backward for a=" << repr() << ", out=" << out.repr() << ", out.id()=" << out.id() << std::endl;
        };
        return out;
    }

    Value exp()
    {
        Value out(std::exp(data), {*this}, "exp");
        backward_ = [&out, this]()
        {
            grad += out.data * out.grad;
        };
        return out;
    }

    Value pow(Value &other)
    {
        value_type n = std::pow(data, other.data);
        Value out(n, {*this, other}, "**");

        backward_ = [&out, &other, this]()
        {
            grad += other.data * std::pow(data, other.data - 1) * out.grad;
        };
        return out;
    }

    Value pow(value_type other)
    {
        Value o(other);
        return pow(o);
    }

    void backward()
    {
        std::unordered_set<Value, IdHasher<Value>, IdEqual<Value>> visited;
        std::queue<Value> q;

        q.push(*this);
        visited.insert(*this);

        grad = 1;
        while (q.size() > 0)
        {
            const auto size = q.size();
            for (size_t i = 0; i < size; i++)
            {
                auto node = q.front();
                q.pop();
                node.backward_();

                for (auto &child : node.prev_)
                {
                    if (!visited.contains(child))
                    {
                        q.push(child);
                        visited.insert(child);
                    }
                }
            }
        }
    }
};

Value operator+(float a, Value &b)
{
    auto av = Value(a);
    return av + b;
}

Value operator-(float a, Value &b)
{
    auto av = Value(a);
    return av - b;
}

Value operator*(float a, Value &b)
{
    auto av = Value(a);
    return av * b;
}

std::ostream &operator<<(std::ostream &out, const Value &v)
{
    out << v.repr();
    return out;
};

