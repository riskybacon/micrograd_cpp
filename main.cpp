
#include <micrograd/engine.hpp>
#include <micrograd/graphviz.hpp>
#include <micrograd/nn.hpp>

void train()
{
    std::vector<std::vector<float>> xs = {
        {2.0f, 3.0f, -1.0f},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0},
    };

    std::vector<float> ys = {1.0, -1.0, -1.0, 1.0};

    auto n = MLP(3, {4, 4, 1});

    assert(n.parameters().size() == 41);
    const float lr = 0.05;
    const size_t num_steps = 500;

    for (size_t step = 0; step < num_steps; step++)
    {
        std::vector<std::vector<Value>> ypred;

        for (auto &x : xs)
        {
            ypred.push_back(n(x));
        }

        Value loss(0.0);

        for (size_t i = 0; i < ypred.size(); i++)
        {
            float ygt = ys[i];
            Value yout = ypred[i][0];
            Value sub = yout - ygt;
            loss += (sub * sub);
        }

#ifndef NO_GRAPHVIZ
        if (step == 0)
        {
            draw_dot(loss, "mlp.dot", "LR");
            generate_png_from_dot("mlp.dot", "mlp.png");
        }
#endif

        if (step % 20 == 0 || step == num_steps - 1)
        {
            std::cout << step << ": " << loss << std::endl;
        }

        n.zero_grad();
        loss.backward();

        for (auto &p : n.parameters())
        {
            p.data() += -(p.grad() * lr);
        }
    }

    std::vector<std::vector<Value>> ypred;
    for (auto &x : xs)
    {
        ypred.push_back(n(x));
    }
    std::cout << "\nypred:\n"
              << ypred << "\n";
}

int main(void)
{
    train();
    return 0;
}