#include <algorithm>
#include <random>
#include <sstream>
#include <vector>

#include <micrograd/engine.hpp>

template <typename T>
std::string join(std::string sep, std::vector<T> &vec)
{
    std::stringstream ss;
    for (size_t i = 0; i < vec.size(); i++)
    {
        ss << vec[i].repr();
        if (i < vec.size() - 1)
        {
            ss << sep;
        }
    }
    return ss.str();
}

struct Module
{
    void zero_grad()
    {
        for (auto &p : parameters())
        {
            p.grad() = 0;
        }
    }

    virtual std::vector<Value> parameters()
    {
        return std::vector<Value>();
    }
};

struct Neuron : Module
{
    Value b;
    bool nonlin;
    std::vector<Value> w;

    Neuron(size_t nin, bool nonlin = true) : b(0), nonlin(nonlin)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<Value::value_type> dist(-1.0, 1.0);
        w.reserve(nin);
        for (size_t i = 0; i < nin; i++)
        {
            w.emplace_back(Value(dist(gen)));
            w.back().label() = "w[" + std::to_string(i) + "]";
        }
        b = Value(dist(gen), "b");
    }

    virtual ~Neuron() {}

    auto operator()(std::vector<Value> &x)
    {
        auto act = dot(w, x) + b;
        if (nonlin)
        {
            return act.tanh();
        }
        return act;
    }

    template <typename Container>
    auto operator()(const Container &values)
    {
        return operator()(to_values(values));
    }

    std::vector<Value> parameters()
    {
        std::vector<Value> out;
        out.reserve(w.size() + 1);
        for (auto &p : w)
        {
            out.push_back(p);
        }
        out.push_back(b);
        return out;
    }

    auto repr()
    {
        std::stringstream ss;
        ss << (nonlin ? "'ReLU'" : "'Linear'") << "Neuron(" << w.size() << ")";
        return ss.str();
    }
};

struct Layer : Module
{
    std::vector<Neuron> neurons;

    Layer(size_t nin, size_t nout, bool nonlin = true)
    {
        std::generate_n(std::back_inserter(neurons), nout, [&]()
                        { return Neuron(nin, nonlin); });
    }

    virtual ~Layer() {}

    auto operator()(std::vector<Value> &x)
    {
        std::vector<Value> out;
        out.reserve(neurons.size());
        for (auto &n : neurons)
        {
            out.emplace_back(n(x));
        }
        return out;
    }

    std::vector<Value> parameters()
    {
        std::vector<Value> out;
        for (auto &n : neurons)
        {
            for (auto &p : n.parameters())
            {
                out.push_back(p);
            }
        }
        return out;
    }

    auto repr()
    {
        std::stringstream ss;
        ss << "Layer of [" << join(", ", neurons) << "]";
        return ss.str();
    }
};

struct MLP : Module
{
    std::vector<Layer> layers;

    MLP(size_t nin, std::vector<size_t> nouts)
    {
        std::vector<size_t> sz;
        sz.push_back(nin);
        for (auto v : nouts)
        {
            sz.push_back(v);
        }

        for (size_t i = 0; i < nouts.size(); i++)
        {
            layers.push_back(Layer(sz[i], sz[i + 1]));
        }
    }

    virtual ~MLP() {}

    auto operator()(std::vector<Value> &x)
    {
        for (auto layer : layers)
        {
            x = layer(x);
        }

        return x;
    }

    auto operator()(std::vector<float> &x)
    {
        auto xv = to_values(x);
        return (*this)(xv);
    }

    template <template <typename> class Container, typename T, typename = std::enable_if_t<std::is_floating_point_v<T> || std::is_integral_v<T>>>
    auto operator()(const Container<T> &values)
    {
        return operator()(to_values(values));
    }

    std::vector<Value> parameters()
    {
        std::vector<Value> out;
        for (auto &layer : layers)
        {
            for (auto &p : layer.parameters())
            {
                out.push_back(p);
            }
        }
        return out;
    }

    auto repr()
    {
        std::stringstream ss;
        ss << "MLP of [" << join(", ", layers) << "]";
        return ss.str();
    }
};
