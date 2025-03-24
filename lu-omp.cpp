/*************************************************************
 * lu-omp.cpp
 *
 * Improved OpenMP-based LU Decomposition with Partial Pivoting.
 * Key features:
 *  - Single-thread pivot search to reduce overhead.
 *  - One parallel region with minimal barriers.
 *  - Vectorized trailing submatrix update with #pragma omp simd.
 *  - No manual tiling (blocking).
 *  - NUMA-aware allocations for large arrays (A, L, U).
 *  - Valid for n=1000..8000 and 1..32 threads, but may need
 *    increased walltime for largest cases on few threads.
 *************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <numa.h>

// Print usage message.
static void usage(const char *progName) {
    std::cerr << "Usage: " << progName << " <matrix_size> <num_threads>\n";
    exit(EXIT_FAILURE);
}

// Utility for indexing a matrix stored in row-major layout as a 1D array.
inline double& elem(double *matrix, int n, int i, int j) {
    return matrix[(long)i * n + j];
}

// Generates an n x n matrix of random doubles in [0,1) using drand48_r().
// Each thread has its own seed/state to avoid data races.
double* generate_random_matrix(int n, int nthreads) {
    double *mat = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!mat) {
        std::cerr << "ERROR: numa_alloc_local failed for matrix.\n";
        exit(EXIT_FAILURE);
    }
    #pragma omp parallel num_threads(nthreads)
    {
        struct drand48_data randBuf;
        srand48_r(2023 + 37 * omp_get_thread_num(), &randBuf);

        #pragma omp for schedule(static)
        for (int i = 0; i < n*n; i++) {
            double x;
            drand48_r(&randBuf, &x);
            mat[i] = x;
        }
    }
    return mat;
}

// Compute the L2,1 norm (sum of Euclidean norms of columns) of (P*A - L*U).
// Steps:
//   1) Form PA by permuting rows of A according to piv.
//   2) Compute R = PA - L*U.
//   3) Sum over columns: sum_{j=0}^{n-1} sqrt( sum_{i=0}^{n-1} (R(i,j))^2 ).
double compute_residual_l21_norm(double *Aorig, int n, int *piv, double *L, double *U) {
    double *PA = (double*) calloc((size_t)n * n, sizeof(double));
    if (!PA) {
        std::cerr << "ERROR: could not allocate memory for PA.\n";
        exit(EXIT_FAILURE);
    }
    // Form PA
    for (int i = 0; i < n; i++) {
        int srcRow = piv[i];
        memcpy(&PA[(long)i * n], &Aorig[(long)srcRow * n], n * sizeof(double));
    }
    // Compute R = PA - L*U
    double *R = (double*) calloc((size_t)n * n, sizeof(double));
    if (!R) {
        std::cerr << "ERROR: could not allocate memory for R.\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sumLU = 0.0;
            for (int k = 0; k < n; k++) {
                sumLU += L[(long)i * n + k] * U[(long)k * n + j];
            }
            R[(long)i * n + j] = PA[(long)i * n + j] - sumLU;
        }
    }
    // Compute L2,1 norm
    double l21 = 0.0;
    for (int j = 0; j < n; j++) {
        double sumSq = 0.0;
        for (int i = 0; i < n; i++) {
            double val = R[(long)i * n + j];
            sumSq += val * val;
        }
        l21 += std::sqrt(sumSq);
    }
    free(PA);
    free(R);
    return l21;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        usage(argv[0]);
    }
    int n = std::atoi(argv[1]);       // matrix size
    int nthreads = std::atoi(argv[2]); // number of threads

    if (n <= 0 || nthreads <= 0) {
        usage(argv[0]);
    }
    omp_set_num_threads(nthreads);

    std::cout << "Running LU Decomposition with row pivoting on a " 
              << n << " x " << n << " matrix using " 
              << nthreads << " threads.\n";

    // 1) Generate random matrix (and keep a copy for residual check).
    double *A0 = generate_random_matrix(n, nthreads);

    // 2) Copy A0 to A for factorization.
    double *A = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!A) {
        std::cerr << "ERROR: numa_alloc_local failed for A.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(A, A0, (size_t)n * n * sizeof(double));

    // 3) Allocate L and U using NUMA local allocation.
    double *L = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    double *U = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!L || !U) {
        std::cerr << "ERROR: failed to allocate L or U.\n";
        exit(EXIT_FAILURE);
    }
    memset(L, 0, n * n * sizeof(double));
    memset(U, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        L[i * n + i] = 1.0;
    }

    // 4) Allocate pivot array.
    int *piv = (int*) malloc(n * sizeof(int));
    if (!piv) {
        std::cerr << "ERROR: failed to allocate pivot array.\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // 5) LU factorization with row pivoting (single-thread pivot search).
    bool singular = false;
    double t_start = omp_get_wtime();

    #pragma omp parallel default(none) shared(A, L, U, piv, n, singular)
    {
        for (int k = 0; k < n; k++) {
            if (singular) {
                // If pivot is zero, skip rest quickly.
                #pragma omp barrier
                continue;
            }
            // --- Single-thread pivot search + row swap ---
            #pragma omp single
            {
                // Find pivot row in [k..n-1]
                double maxval = 0.0;
                int maxrow = k;
                for (int i = k; i < n; i++) {
                    double val = std::fabs(A[(long)i * n + k]);
                    if (val > maxval) {
                        maxval = val;
                        maxrow = i;
                    }
                }
                if (maxval == 0.0) {
                    // Singular matrix
                    singular = true;
                } else {
                    // Swap pivot array
                    std::swap(piv[k], piv[maxrow]);
                    // Swap rows in A
                    if (maxrow != k) {
                        for (int j = 0; j < n; j++) {
                            std::swap(A[k*n + j], A[maxrow*n + j]);
                        }
                        // Swap the first k entries of L
                        for (int j = 0; j < k; j++) {
                            std::swap(L[k*n + j], L[maxrow*n + j]);
                        }
                    }
                    // Set U(k,k)
                    U[k*n + k] = A[k*n + k];
                }
            } // end single

            // Implicit barrier at end of single
            if (!singular) {
                // --- Update L(i,k) and U(k,i) in parallel ---
                #pragma omp for
                for (int i = k+1; i < n; i++) {
                    L[i*n + k] = A[i*n + k] / A[k*n + k];
                    U[k*n + i] = A[k*n + i];
                }

                // --- Update trailing submatrix: A(i,j) -= L(i,k)*U(k,j) ---
                #pragma omp for
                for (int i = k+1; i < n; i++) {
                    double lik = L[i*n + k];
                    double *Arow = &A[i*n];
                    double *Urow = &U[k*n];
                    #pragma omp simd
                    for (int j = k+1; j < n; j++) {
                        Arow[j] -= lik * Urow[j];
                    }
                }
            }
            // Implicit barrier at end of for
        } // end for k
    } // end parallel region

    double t_end = omp_get_wtime();
    double factor_time = t_end - t_start;

    if (singular) {
        std::cerr << "ERROR: Factorization failed: matrix is singular (pivot = 0).\n";
    }

    // 6) Compute residual L2,1 norm
    double l21_norm = 0.0;
    if (!singular) {
        l21_norm = compute_residual_l21_norm(A0, n, piv, L, U);
    }

    std::cout << "LU factorization time: " << factor_time << " seconds.\n";
    if (!singular) {
        std::cout << "Residual L2,1 norm = " << l21_norm << "\n";
    }

    // 7) Free memory
    numa_free(A0, (size_t)n * n * sizeof(double));
    numa_free(A,  (size_t)n * n * sizeof(double));
    numa_free(L,  (size_t)n * n * sizeof(double));
    numa_free(U,  (size_t)n * n * sizeof(double));
    free(piv);

    return 0;
}
