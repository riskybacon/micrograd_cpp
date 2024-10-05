
#include <micrograd/engine.hpp>
#include <micrograd/graphviz.hpp>

void test1(void) {
    auto a = Value(2.0f, "a");
    auto b = Value(-3.0f, "b");
    auto c = Value(10.0f, "c");
    auto e = a + b;
    e.label = "e";
    auto d = e + c;
    d.label = "d";
    auto f = Value(-2.0f, "f");
    auto L = d * f;
    L.label = "L";

    L.backward();

#ifndef NO_GRAPHVIZ
    draw_dot(L, "L.dot", "TB");
    generate_png_from_dot("L.dot", "L.png");
#endif
}

int main(void)
{
    test1();
    return 0;
}