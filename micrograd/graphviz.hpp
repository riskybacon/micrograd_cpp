#pragma once
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <set>
#include <sstream>
#include <tuple>

#include <micrograd/engine.hpp>

std::string context_to_graphviz(const std::shared_ptr<Context> &ctx)
{
    std::stringstream node_label;
    node_label << "{" << ctx->label << " | data " << std::fixed << std::setprecision(4) << ctx->data << " | grad " << ctx->grad << "}";
    std::stringstream ss;
    ss << "\"" << ctx.get() << "\" [label=\"" << node_label.str() << "\", shape=record];\n";

    if (!ctx->op.empty()) {
        ss << "\"" << ctx.get() << "_op\" [label=\"" << ctx->op << "\"];\n";
        ss << "\"" << ctx.get() << "_op\" -> \"" << ctx.get() << "\";\n";
    }

    return ss.str();
}

auto trace(Value &root)
{
    std::unordered_set<std::shared_ptr<Context>> nodes;
    std::vector<std::pair<std::shared_ptr<Context>, std::shared_ptr<Context>>> edges;

    std::function<void(std::shared_ptr<Context>&)> build = [&](std::shared_ptr<Context> &node)
    {
        if (!nodes.contains(node))
        {
            nodes.insert(node);
            for (auto &child : node->prev)
            {
                edges.push_back({child, node});
                build(child);
            }
        }
    };

    build(root.ctx_);
    return std::make_pair(nodes, edges);
}

void draw_dot(Value &root, const std::string &filename, const std::string &rankdir = "LR")
{
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
    for (auto &n : nodes)
    {
        file << context_to_graphviz(n);
    }

    // Add edges to the dot file
    for (const auto &edge : edges)
    {
        const auto &n1 = edge.first;
        const auto &n2 = edge.second;
        file << "\"" << n1.get() << "\" -> \"" << n2.get() << "_op\";\n";
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
