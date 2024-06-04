#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <iomanip>

size_t N;
size_t threads;

void MatrixVectorMultiply(const std::vector<double> &matrix, const std::vector<double> &vector,
                          std::vector<double> &resultVector, int lb, int ub) {
    for (int i = lb; i <= ub; ++i) {
        resultVector[i] = 0;
        for (int j = 0; j < N; ++j)
            resultVector[i] += matrix[i * N + j] * vector[j];
    }
}

void VectorSubtract(const std::vector<double> &vector0, const std::vector<double> &vector1,
                    std::vector<double> &resultVector, int lb, int ub) {
    for (int i = lb; i <= ub; ++i)
        resultVector[i] = vector0[i] - vector1[i];
}

void ScalarVectorMultiply(double scalar, const std::vector<double> &vector, std::vector<double> &resultVector,
                          int lb, int ub) {
    for (int i = lb; i <= ub; ++i)
        resultVector[i] = scalar * vector[i];
}

double SquaredNorm(const std::vector<double> &vector, int lb, int ub) {
    double res{};
    for (int i = lb; i <= ub; ++i)
        res += vector[i] * vector[i];
    return res;
}

void Algorithm(const std::vector<double> &A, const std::vector<double> &b, std::vector<double> &X, double tau) {
    std::vector<double> bufferVector(N);
    double numerator{}, denominator{};
#pragma omp parallel num_threads(threads)
    {
        int numThreads = omp_get_num_threads();
        int threadNumber = omp_get_thread_num();
        int items_per_thread = N / numThreads;
        int lb = threadNumber * items_per_thread;
        int ub = (threadNumber == numThreads - 1) ? (N - 1) : (lb + items_per_thread - 1);
        double numBuffer{}, denomBuffer{};
        while (true) {
            MatrixVectorMultiply(A, X, bufferVector, lb, ub);
            VectorSubtract(bufferVector, b, bufferVector, lb, ub);
            numBuffer = SquaredNorm(bufferVector, lb, ub);
            denomBuffer = SquaredNorm(b, lb, ub);
#pragma omp single
            {
                numerator = 0;
                denominator = 0;
            }
#pragma omp atomic
            numerator += numBuffer;
#pragma omp atomic
            denominator += denomBuffer;
#pragma omp single
            {
                numerator = sqrt(numerator);
                denominator = sqrt(denominator);
            }
            if (numerator < 0.00001 * denominator)
                break;
            ScalarVectorMultiply(tau, bufferVector, bufferVector, lb, ub);
            VectorSubtract(X, bufferVector, X, lb, ub);
        }
    }
}

int main(int argc, char **argv) {
    N = atoi(argv[1]);
    double tau = std::stod(argv[2]);
    threads = atoi(argv[3]);

    std::vector<double> A(N * N, 1);
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < N; ++i)
        A[i * N + i] = 2;

    const std::vector<double> b(N, N + 1);
    std::vector<double> X(N, 0);

    const auto start{std::chrono::steady_clock::now()};
    Algorithm(A, b, X, tau);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout << "Time: " << std::fixed << std::setprecision(4) << elapsed_seconds.count() << std::endl;
    return 0;
}
