#pragma once
#include <functional>
#include <vector>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>

struct Context
{
    using value_type = float;
    value_type data;
    value_type grad = 0;
    std::string label;
    std::string op;
    std::function<void()> backward = []() {};
    std::vector<std::shared_ptr<Context>> prev;

    Context(value_type data) : data(data) {}
    Context(value_type data, const std::string &label) : data(data), label(label) {}
    Context(value_type data, std::initializer_list<std::shared_ptr<Context>> &&prev, const std::string &op)
        : data(data), op(op), prev(prev)
    {
    }

    std::string repr() const
    {
        std::stringstream ss;
        ss << "(" << label << "," << this << ", data=" << data << ", grad=" << grad << ")";
        return ss.str();
    }
};

auto operator+(std::shared_ptr<Context> &lhs, std::shared_ptr<Context> &rhs)
{
    auto ptr = new Context(lhs->data + rhs->data, {lhs, rhs}, "+");
    auto out = std::shared_ptr<Context>(ptr);
    out->backward = [out, lhs, rhs]()
    {
        lhs->grad += out->grad;
        rhs->grad += out->grad;
    };
    return out;
}

auto operator*(std::shared_ptr<Context> &lhs, std::shared_ptr<Context> &rhs)
{
    auto ptr = new Context(lhs->data * rhs->data, {lhs, rhs}, "*");
    auto out = std::shared_ptr<Context>(ptr);
    out->backward = [out, lhs, rhs]()
    {
        lhs->grad += rhs->data * out->grad;
        rhs->grad += lhs->data * out->grad;
    };
    return out;
}

auto tanh(std::shared_ptr<Context> &lhs)
{
    auto ptr = new Context(std::tanh(lhs->data), {lhs}, "tanh");
    auto out = std::shared_ptr<Context>(ptr);
    out->backward = [out, lhs]()
    {
        lhs->grad += (1 - out->data * out->data) * out->grad;
    };
    return out;
}

auto exp(std::shared_ptr<Context> &lhs)
{
    auto ptr = new Context(std::exp(lhs->data), {lhs}, "exp");
    auto out = std::shared_ptr<Context>(ptr);
    out->backward = [out, lhs]()
    {
        lhs->grad += out->data * out->grad;
    };
    return out;
}

auto pow(std::shared_ptr<Context> &lhs, std::shared_ptr<Context> &rhs)
{
    auto ptr = new Context(std::pow(lhs->data, rhs->data), {lhs, rhs}, "pow");
    auto out = std::shared_ptr<Context>(ptr);
    out->backward = [out, lhs, rhs]()
    {
        lhs->grad += rhs->data * std::pow(lhs->data, rhs->data - 1) * out->grad;
    };
    return out;
}

void backward(std::shared_ptr<Context> &root)
{
    std::unordered_set<std::shared_ptr<Context>> visited;
    std::queue<std::shared_ptr<Context>> q;

    q.push(root);
    visited.insert(root);
    root->grad = 1;

    while (q.size() > 0)
    {
        auto ctx = q.front();
        q.pop();
        ctx->backward();

        for (auto &child : ctx->prev)
        {
            if (!visited.contains(child))
            {
                q.push(child);
                visited.insert(child);
            }
        }
    }
}

struct Value
{
    using value_type = Context::value_type;
    std::shared_ptr<Context> ctx_;

    Value(value_type data) : ctx_(std::make_shared<Context>(data)) {}
    Value(value_type data, const std::string &label) : ctx_(std::make_shared<Context>(data, label)) {}
    Value(std::shared_ptr<Context> &&ctx) : ctx_(ctx) {}

    Value operator+(Value &rhs)
    {
        return Value(ctx_ + rhs.ctx_);
    }

    Value operator+(Value &&rhs)
    {
        return *this + rhs;
    }

    Value operator+(value_type rhs)
    {
        return *this + Value(rhs);
    }

    Value operator+(value_type &&rhs)
    {
        return *this + rhs;
    }

    Value &operator+=(Value &rhs)
    {
        if (this != &rhs)
        {
            ctx_ = std::shared_ptr<Context>(ctx_ + rhs.ctx_);
        }
        return *this;
    }

    Value &operator+=(Value &&rhs)
    {
        return *this += rhs;
    }

    Value &operator+=(value_type rhs)
    {
        return *this += Value(rhs);
    }

    Value &operator+=(value_type &&rhs)
    {
        return *this += rhs;
    }

    Value operator-()
    {
        auto minus_1 = Value(-1);
        return Value(ctx_ * minus_1.ctx_);
    }

    Value operator-(Value &rhs)
    {
        auto n = -rhs;
        return Value(ctx_ + n.ctx_);
    }

    Value operator-(Value &&rhs)
    {
        return *this - rhs;
    }

    Value operator-(value_type rhs)
    {
        return *this - Value(rhs);
    }

    Value operator-(value_type &&rhs)
    {
        return *this - rhs;
    }

    Value &operator-=(Value &rhs)
    {
        if (this != &rhs)
        {
            auto out = *this - rhs;
            ctx_ = out.ctx_;
        }
        return *this;
    }

    Value &operator-=(Value &&rhs)
    {
        return *this -= rhs;
    }

    Value &operator-=(value_type rhs)
    {
        return *this -= Value(rhs);
    }

    Value &operator-=(value_type &&rhs)
    {
        return *this -= rhs;
    }

    Value operator*(Value &rhs)
    {
        return Value(ctx_ * rhs.ctx_);
    }

    Value operator*(Value &&rhs)
    {
        return *this * rhs;
    }

    Value operator*(value_type rhs)
    {
        return *this * Value(rhs);
    }

    Value operator*(value_type &&rhs)
    {
        return *this * rhs;
    }

    Value &operator*=(Value &rhs)
    {
        if (this != &rhs)
        {
            ctx_ = ctx_ * rhs.ctx_;
        }
        return *this;
    }

    Value &operator*=(Value &&rhs)
    {
        return *this *= rhs;
    }

    Value &operator*=(value_type rhs)
    {
        return *this *= Value(rhs);
    }

    Value &operator*=(value_type &&rhs)
    {
        return *this *= rhs;
    }

    Value operator/(Value &rhs)
    {
        Value n(-1);
        auto b = rhs.pow(n);
        return *this * b;
    }

    Value operator/(Value &&rhs)
    {
        return *this / rhs;
    }

    Value operator/(const value_type &rhs)
    {
        return *this / Value(rhs);
    }

    Value operator/(const value_type &&rhs)
    {
        return *this / rhs;
    }

    Value tanh()
    {
        return Value(::tanh(ctx_));
    }

    Value exp()
    {
        return Value(::exp(ctx_));
    }

    Value pow(Value &lhs)
    {
        return Value(::pow(ctx_, lhs.ctx_));
    }

    Value pow(Value &&lhs)
    {
        return pow(lhs);
    }

    Value pow(value_type lhs)
    {
        return pow(Value(lhs));
    }

    Value pow(value_type &&lhs)
    {
        return pow(Value(lhs));
    }

    void backward()
    {
        return ::backward(ctx_);
    }

    std::string repr() const
    {
        return ctx_->repr();
    }

    // Convenience methods to make updating context easy with similar api as original micrograd
    value_type &data() { return ctx_->data; }
    const value_type &data() const { return ctx_->data; }
    value_type &grad() { return ctx_->grad; }
    const value_type &grad() const { return ctx_->grad; }
    std::string &label() { return ctx_->label; }
    const std::string &label() const { return ctx_->label; }
    std::string &op() { return ctx_->op; }
    const std::string &op() const { return ctx_->op; }
};

template <template <typename> class Container, typename T, typename = std::enable_if_t<std::is_floating_point_v<T> || std::is_integral_v<T>>>
std::vector<Value> to_values(const Container<T> &values)
{
    std::vector<Value> out;
    out.reserve(values.size());
    for (auto &v : values)
    {
        out.emplace_back(Value(static_cast<float>(v)));
    }
    return out;
}

Value dot(std::vector<Value> &a, std::vector<Value> &b)
{
    assert(a.size() == b.size());
    auto out = Value(0, "zero");

    for (size_t i = 0; i < a.size(); i++)
    {
        out += a[i] * b[i];
    }

    return out;
}

std::ostream &operator<<(std::ostream &out, const Value &v)
{
    out << v.repr();
    return out;
};