#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>
#include <numa.h>
#include <cstring>  // For memset
#include <iomanip>  // For std::setprecision

using namespace std;

// Helper to print usage message
void usage(const char *name) {
    cout << "usage: " << name << " matrix-size nworkers" << endl;
    exit(-1);
}

// Compute L2,1 norm of residual PA - LU
double computeResidualNorm(const vector<vector<double>>& A,
                           const vector<vector<double>>& L,
                           const vector<vector<double>>& U,
                           const vector<int>& pi) {
    int n = A.size();
    double norm = 0.0;

    #pragma omp parallel for reduction(+:norm)
    for (int j = 0; j < n; ++j) {
        double colNorm = 0.0;
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += L[i][k] * U[k][j];
            }
            double diff = A[pi[i]][j] - sum;
            colNorm += diff * diff;
        }
        norm += sqrt(colNorm);
    }

    return norm;
}

int main(int argc, char **argv) {
    const char *progName = argv[0];

    if (argc < 3) usage(progName);

    int n = atoi(argv[1]);
    int nthreads = atoi(argv[2]);

    cout << "Running " << progName << " with matrix size " << n << " and " << nthreads << " threads." << endl;

    omp_set_num_threads(nthreads);

    // Allocate and initialize matrix A with random numbers (NUMA-aware)
    vector<vector<double>> A(n, vector<double>(n));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937_64 rng(42 + tid);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            void* mem = numa_alloc_local(sizeof(double) * n);
            memcpy(A[i].data(), mem, sizeof(double) * n);  // Using numa alloc
            for (int j = 0; j < n; ++j) {
                A[i][j] = dist(rng);
            }
        }
    }

    // Permutation vector π, Lower L, Upper U
    vector<int> pi(n);
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    vector<vector<double>> U(n, vector<double>(n, 0.0));

    // Initialize permutation π and L (identity)
    for (int i = 0; i < n; ++i) {
        pi[i] = i;
        L[i][i] = 1.0;
    }

    double t1 = omp_get_wtime();

    // LU Decomposition with partial pivoting
    for (int k = 0; k < n; ++k) {
        // Pivot selection (serial)
        double maxA = 0.0;
        int k_ = -1;
        for (int i = k; i < n; ++i) {
            double val = fabs(A[i][k]);
            if (val > maxA) {
                maxA = val;
                k_ = i;
            }
        }

        if (maxA == 0.0) {
            cerr << "Singular matrix detected!" << endl;
            exit(-1);
        }

        // Swap rows k and k_ in A
        if (k != k_) {
            swap(pi[k], pi[k_]);
            swap(A[k], A[k_]);
            if (k > 0) {
                for (int j = 0; j < k; ++j) {
                    swap(L[k][j], L[k_][j]);
                }
            }
        }

        U[k][k] = A[k][k];

        // Compute L[i][k] and U[k][i]
        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; ++i) {
            L[i][k] = A[i][k] / U[k][k];
        }

        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; ++i) {
            U[k][i] = A[k][i];
        }

        // Update A[i][j] in trailing submatrix
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    double t2 = omp_get_wtime();

    double elapsed = t2 - t1;

    // Compute residual norm
    double residualNorm = computeResidualNorm(A, L, U, pi);

    // Print results
    cout << fixed << setprecision(6);
    cout << "LU decomposition completed in " << elapsed << " seconds." << endl;
    cout << "Residual L2,1 norm = " << residualNorm << endl;

    return 0;
}
