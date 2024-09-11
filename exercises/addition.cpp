#include <cassert>
#include <iostream>
#include <functional>
#include <chrono>
#include <omp.h>

#define NUM_THREADS 5

template<typename T>
void iterative_sum(T a[], T b[], T c[], T n)
{
    for (T i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

template<typename T>
void parallel_sum(T a[], T b[], T c[], T n)
{
    #pragma omp parallel for
    for (T i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int reduction(int a[], int b[], std::function<int(int, int)> op, int n)
{
    if (n == 0) {
        return 0;
    }
    int res = op(a[0], b[0]);
    return res + reduction(a + 1, b + 1, op, n - 1);
}

int main()
{
    using namespace std::chrono_literals;

    int a[NUM_THREADS] = {1, 2, 3, 4, 5};
    int b[NUM_THREADS] = {6, 7, 8, 9, 10};
    int c[NUM_THREADS];

    auto start = std::chrono::high_resolution_clock::now();
    
    iterative_sum<int>(a, b, c, NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        assert(c[i] == a[i] + b[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Iterative sum took " << elapsed.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    
    int d[NUM_THREADS];
    parallel_sum<int>(a, b, d, NUM_THREADS);
    for(int i = 0; i < NUM_THREADS; i++) {
        assert(d[i] == c[i]);
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Parallel sum took " << elapsed.count() << " ms" << std::endl;

    int res = reduction(a, b, [](int x, int y) { return x + y; }, NUM_THREADS);
    assert(res == 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10);

    return 0;
}
