#include <iostream>
#include <vector>
#include <cmath>

#ifdef FLOAT
typedef float T;
#else
typedef double T;
#endif

int main() {
    int size = 10000000;
    std::vector<T> vec(size);
    T sum{};
    for (int i = 0; i < size; ++i) {
#ifdef FLOAT
        vec[i] = sinf(M_PI * 2 / size * i);
#else
        vec[i] = sin(M_PI * 2/ size * i);
#endif
        sum += vec[i];
    }
    std::cout << sum << std::endl;

    return 0;
}
