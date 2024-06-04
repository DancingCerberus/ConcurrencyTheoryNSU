#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <numbers>

const double a = -4.0;
const double b = 4.0;
const int nsteps = 40000000;

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double) ts.tv_sec + (double) ts.tv_nsec * 1.e-9);
}

double func(double x) {
    return exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));
    sum *= h;
    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int threads) {
    double h = (b - a) / n;
    double sum = 0.0;
#pragma omp parallel num_threads(threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double sumlock = 0.0;
        for (int i = lb; i <= ub; i++)
            sumlock += func(a + h * (i + 0.5));
#pragma omp atomic
        sum += sumlock;
    }
    sum *= h;
    return sum;
}


double run_parallel(int threads) {
    double t = cpuSecond();
    double res = integrate_omp(func, a, b, nsteps, threads);
    t = cpuSecond() - t;
    std::cout << "Result (parallel): " << res << "; error " << fabs(res - std::sqrt(std::numbers::pi)) << std::endl;
    return t;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_threads>" << std::endl;
        return 1;
    }

    int threads = std::atoi(argv[1]);
    std::cout << "Integration f(x) on [" << a << ", " << b << "], nsteps = " << nsteps << std::endl;
    double tparallel = run_parallel(threads);

    std::cout << "Execution time (parallel): " << tparallel << " sec" << std::endl;


    return 0;
}
