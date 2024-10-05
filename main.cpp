#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <ostream>
#include <set>
#include <string>
#include <sstream>
#include <vector>
#include <memory>

using namespace std;

struct Value;
using ValueRef = std::reference_wrapper<Value>;
using Linear = vector<ValueRef>;

struct RefWrapperComparator
{
    template <typename T>
    bool operator()(const std::reference_wrapper<T> &lhs, const std::reference_wrapper<T> &rhs) const
    {
        return lhs.get() < rhs.get(); // Compare the underlying objects
    }
};

// Define a custom comparator for pairs of reference_wrappers
struct PairRefWrapperComparator
{
    template <typename T1, typename T2>
    bool operator()(const std::pair<std::reference_wrapper<T1>, std::reference_wrapper<T2>> &lhs,
                    const std::pair<std::reference_wrapper<T1>, std::reference_wrapper<T2>> &rhs) const
    {
        // Compare first elements, if they are equal compare second elements
        if (lhs.first.get() != rhs.first.get())
        {
            return lhs.first.get() < rhs.first.get();
        }
        return lhs.second.get() < rhs.second.get();
    }
};

struct Value
{
    using value_type = float;
    using linear_type = Linear;
    value_type data;
    value_type grad = 0;
    std::string label;
    Linear _prev;
    std::string _op;
    std::function<void()> _backward = []() {};

    Value(const value_type &data, const string &label) : data(data), label(label) {}
    Value(const value_type &d, linear_type &prev) : data(d), _prev(prev) {}
    Value(const value_type &d, linear_type &prev, const string &op) : data(d), _prev(prev), _op(op) {}

    string repr() const
    {
        stringstream ss;
        ss << "Value(data=" << data << ")";
        return ss.str();
    }

    Value operator+(Value &other)
    {
        linear_type prev = {*this, other};
        auto out = Value(data + other.data, prev, "+");
        out._backward = [&out, &other, this]()
        {
            grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };
        return out;
    }

    Value operator*(Value &other)
    {
        // auto out = Value(data * other.data, {this, &other}, "*");
        linear_type prev = {*this, other};
        auto out = Value(data * other.data, prev, "*");
        out._backward = [&other, this]()
        {
            grad = other.data;
            other.grad = data;
        };
        return out;
    }

    void backward(void)
    {
        _backward();
    }

    bool operator<(const Value &other) const
    {
        return this < &other;
    }

    bool operator==(const Value &other) const
    {
        return this == &other;
    }

    bool operator!=(const Value &other) const
    {
        return this != &other;
    }

    string graphviz_str() const
    {
        string node_label = "{ " + label + " | data " + std::to_string(data) + " | grad " + std::to_string(grad) + " }";

        stringstream ss;
        ss << "\"" << this << "\" [label=\"" << node_label << "\", shape=record];\n";

        if (!_op.empty()) {
            ss << "\"" << this << "_op\" [label=\"" << _op << "\"];\n";
            ss << "\"" << this << "_op\" -> \"" << this << "\";\n";
        }

        return ss.str();
    }
};

ostream &operator<<(ostream &out, const Value &v)
{
    out << v.repr();
    return out;
};

using ValueRefSet = std::set<ValueRef, RefWrapperComparator>;
using PairValueRefSet = std::set<std::pair<ValueRef, ValueRef>, PairRefWrapperComparator>;


pair<ValueRefSet, PairValueRefSet> trace(ValueRef root)
{
    ValueRefSet nodes;
    PairValueRefSet edges;

    function<void(ValueRef)> build = [&](ValueRef node)
    {
        if (nodes.find(node) == nodes.end())
        {
            nodes.insert(node);
            for (auto &child : node.get()._prev)
            {
                edges.insert({child, node});
                build(child);
            }
        }
    };

    build(root);
    return {nodes, edges};
}

// Function to generate the .dot file from the traced nodes and edges
void draw_dot(ValueRef root, const std::string &filename, const std::string &rankdir = "LR")
{
    // Validate rankdir input
    if (rankdir != "LR" && rankdir != "TB")
    {
        std::cerr << "Invalid rankdir. Use 'LR' or 'TB'." << std::endl;
        return;
    }

    auto [nodes, edges] = trace(root);

    // Create the .dot file
    std::ofstream file(filename);

    file << "digraph G {\n";
    file << "rankdir=" << rankdir << ";\n";

    // Add nodes to the dot file
    for (const auto &n : nodes)
    {
        file << n.get().graphviz_str();
    }

    // Add edges to the dot file
    for (const auto &edge : edges)
    {
        const auto &n1 = edge.first;
        const auto &n2 = edge.second;
        file << "\"" << &(n1.get()) << "\" -> \"" << &(n2.get()) << "_op\";\n";
    }

    file << "}\n";
    file.close();

    std::cout << "Graphviz .dot file generated: " << filename << std::endl;
}

void generate_png_from_dot(const std::string &dot_filename, const std::string &output_png_filename)
{
    // Construct the command to convert .dot to .png using Graphviz's dot tool
    std::string command = "dot -Tpng " + dot_filename + " -o " + output_png_filename;

    // Execute the command
    int result = system(command.c_str());

    // Check if the command succeeded
    if (result == 0)
    {
        std::cout << "PNG image successfully created: " << output_png_filename << std::endl;
    }
    else
    {
        std::cerr << "Failed to create PNG image. Make sure Graphviz is installed and 'dot' is in your PATH." << std::endl;
    }
}

int main(void)
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

    Linear model = {
        L, d, f, e, c, b, a};

    L.grad = 1;

    for (auto layer : model)
    {
        layer.get().backward();
    }

    draw_dot(L, "graph.dot", "TB");
    generate_png_from_dot("graph.dot", "graph.png");
    cout << a + b << endl;
    return 0;
}