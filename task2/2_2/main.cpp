#include <iostream>
#include <cmath>
#include <numbers>

using std::size_t;
using std::sin;

static size_t maxThreadsCount;
const size_t nSteps = 40'000'000;

double integrate(double (*func)(double), double from, double to) {
    const double step = (to - from) / nSteps;
    double sum = 0.;

#pragma omp parallel for num_threads(maxThreadsCount)
    for (size_t i = 0; i < nSteps; ++i) {
        const double left = static_cast<double>(i) * step;
        const double right = left + step;
        double const segment = (right - left) / 6. * (func(left) + 4 * func((left + right) / 2.) + func(right));
#pragma omp atomic
        sum += segment;
    }

    return sum;
}

int main(int argc, char *argv[]) {

    maxThreadsCount = std::stoi(argv[1]);

    auto integrableFunction = [](double x) {
        return sin(x);
    };

    std::cout << integrate(integrableFunction, 0., std::numbers::pi) << std::endl;

    return 0;
}
