#include <iostream>
#include <vector>
#include <omp.h>

void matrix_vector_product(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void matrix_vector_product_omp(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int n, int threads) {
#pragma omp parallel num_threads(threads)
    {
        int nThreads = omp_get_num_threads();
        int threadId = omp_get_thread_num();
        int items_per_thread = n / nThreads;
        int lb = threadId * items_per_thread;
        int ub = (threadId == nThreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_serial(int n, int iterations) {
    std::vector<double> a(n * n);
    std::vector<double> b(n);
    std::vector<double> c(n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;

    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = omp_get_wtime();
    for (int i = 0; i < iterations; i++)
        matrix_vector_product(a, b, c, n);
    t = omp_get_wtime() - t;

    std::cout << "Elapsed time (serial): " << t / iterations << " sec." << std::endl;
}

void run_parallel(int n, int threads, int iterations) {
    std::vector<double> a(n * n);
    std::vector<double> b(n);
    std::vector<double> c(n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;

    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = omp_get_wtime();
    for (int i = 0; i < iterations; i++)
        matrix_vector_product_omp(a, b, c, n, threads);
    t = omp_get_wtime() - t;

    std::cout << "Elapsed time (parallel): " << t / iterations << " sec." << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 4)
        return 1;

    int n = std::atoi(argv[1]);
    int threads = std::atoi(argv[2]);
    int iterations = std::atoi(argv[3]);

    std::cout << "Matrix-vector product (c[n] = a[n, n] * b[n]; n = " << n << ")\n";
    std::cout << "Memory used: " << ((n * n + n + n) * sizeof(double)) / (1024 * 1024) << " MiB\n";

    run_serial(n, iterations);
    run_parallel(n, threads, iterations);

    return 0;
}
