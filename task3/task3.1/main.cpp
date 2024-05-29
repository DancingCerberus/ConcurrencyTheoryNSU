#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <functional>

std::chrono::duration<double> MatrixVectorMultiply(int size, int numThreads) {
    std::vector<std::thread> workerThreads(numThreads);
    std::vector<double> matrix(size * size);
    std::vector<double> vector(size);

    int chunkSize = size / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startIdx = i * chunkSize;
        int endIdx = (i == numThreads - 1) ? size : (startIdx + chunkSize);

        workerThreads[i] = std::thread([](std::vector<double> &mat, std::vector<double> &vec, int start, int end) {
            for (int row = start; row < end; ++row) {
                for (int col = 0; col < vec.size(); ++col) {
                    mat[row * vec.size() + col] = row + col;
                }
                vec[row] = row;
            }
        }, std::ref(matrix), std::ref(vector), startIdx, endIdx);
    }
    std::for_each(workerThreads.begin(), workerThreads.end(), std::mem_fn(&std::thread::join));

    std::vector<double> result(size, 0.0);

    const auto startTime = std::chrono::steady_clock::now();

    for (int i = 0; i < numThreads; ++i) {
        int startIdx = i * chunkSize;
        int endIdx = (i == numThreads - 1) ? size : (startIdx + chunkSize);

        workerThreads[i] = std::thread(
                [&result](const std::vector<double> &mat, const std::vector<double> &vec, int start, int end) {
                    for (int row = start; row < end; ++row) {
                        for (int col = 0; col < vec.size(); ++col) {
                            result[row] += mat[row * vec.size() + col] * vec[col];
                        }
                    }
                }, std::cref(matrix), std::cref(vector), startIdx, endIdx);
    }
    std::for_each(workerThreads.begin(), workerThreads.end(), std::mem_fn(&std::thread::join));

    const auto endTime = std::chrono::steady_clock::now();

    return (endTime - startTime);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <number_of_threads>\n";
        return 1;
    }

    int matrixSize = std::stoi(argv[1]);
    int numThreads = std::stoi(argv[2]);

    auto duration = MatrixVectorMultiply(matrixSize, numThreads);

    std::cout << "Duration: " << std::fixed << duration.count() << " seconds\n";

    return 0;
}
