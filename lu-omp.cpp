/*************************************************************
 * lu-omp.cpp
 *
 * Further Optimized OpenMP-based LU Decomposition with Partial Pivoting.
 * Improvements:
 * - Uses drand48_r() for thread-safe random number generation.
 * - Uses a single parallel region with careful synchronization.
 * - Applies blocking (tiling) in the trailing submatrix update for better cache locality.
 * - Uses collapse(2) and simd directives to improve vectorization and load balance.
 * - Allocates all large matrices (A, L, U) using NUMA-aware allocation.
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
    for (int i = 0; i < n; i++) {
        int srcRow = piv[i];
        memcpy(&PA[(long)i * n], &Aorig[(long)srcRow * n], n * sizeof(double));
    }
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
    int n = std::atoi(argv[1]);      // matrix size
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
    double * __restrict__ A = (double* __restrict__) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!A) {
        std::cerr << "ERROR: numa_alloc_local failed for A.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(A, A0, (size_t)n * n * sizeof(double));

    // 3) Allocate L and U using NUMA local allocation.
    double * __restrict__ L = (double* __restrict__) numa_alloc_local((size_t)n * n * sizeof(double));
    double * __restrict__ U = (double* __restrict__) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!L || !U) {
        std::cerr << "ERROR: failed to allocate L or U.\n";
        exit(EXIT_FAILURE);
    }
    // Initialize L to 0 and U to 0, then set L's diagonal to 1.
    memset(L, 0, n * n * sizeof(double));
    memset(U, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        L[(long)i * n + i] = 1.0;
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

    // Allocate temporary arrays for per-thread pivot reduction.
    double *thread_maxvals = new double[nthreads];
    int    *thread_maxrows = new int[nthreads];

    // 5) LU factorization with row pivoting.
    bool singular = false;  // flag to detect singular matrix.
    double t_start = omp_get_wtime();

    // Block sizes for the trailing submatrix update.
    const int block_size_i = 64;
    const int block_size_j = 64;

    #pragma omp parallel default(none) shared(A, L, U, piv, n, singular, thread_maxvals, thread_maxrows, block_size_i, block_size_j)
    {
        for (int k = 0; k < n; k++) {
            if (singular) {
                #pragma omp barrier
                continue;
            }
            int tid = omp_get_thread_num();
            double local_maxval = 0.0;
            int local_maxrow = k;
            // Parallel pivot search over rows k...n-1.
            #pragma omp for nowait
            for (int i = k; i < n; i++) {
                double val = std::fabs(elem(A, n, i, k));
                if (val > local_maxval) {
                    local_maxval = val;
                    local_maxrow = i;
                }
            }
            thread_maxvals[tid] = local_maxval;
            thread_maxrows[tid] = local_maxrow;
            #pragma omp barrier

            double maxval;
            int maxrow;
            #pragma omp single
            {
                maxval = 0.0;
                maxrow = k;
                int num_threads = omp_get_num_threads();
                for (int t = 0; t < num_threads; t++) {
                    if (thread_maxvals[t] > maxval) {
                        maxval = thread_maxvals[t];
                        maxrow = thread_maxrows[t];
                    }
                }
                if (maxval == 0.0) {
                    singular = true;
                } else {
                    // Swap pivot entries.
                    int tmpP = piv[k];
                    piv[k] = piv[maxrow];
                    piv[maxrow] = tmpP;
                    // Swap rows in A.
                    if (maxrow != k) {
                        for (int j = 0; j < n; j++) {
                            double tmpA = A[k * n + j];
                            A[k * n + j] = A[maxrow * n + j];
                            A[maxrow * n + j] = tmpA;
                        }
                        // Swap the first k entries of L.
                        for (int j = 0; j < k; j++) {
                            double tmpL = L[k * n + j];
                            L[k * n + j] = L[maxrow * n + j];
                            L[maxrow * n + j] = tmpL;
                        }
                    }
                    U[k * n + k] = A[k * n + k];
                }
            }
            #pragma omp barrier

            if (!singular) {
                double pivotVal = A[k * n + k];
                // Update L and U for the kth column/row.
                #pragma omp for
                for (int i = k+1; i < n; i++) {
                    L[i * n + k] = A[i * n + k] / pivotVal;
                    U[k * n + i] = A[k * n + i];
                }
                // Trailing submatrix update: update A(i,j) for i,j > k.
                // Using collapse(2) to combine the two outer loops and simd for the inner loop.
                #pragma omp for collapse(2) schedule(static)
                for (int ii = k+1; ii < n; ii += block_size_i) {
                    for (int jj = k+1; jj < n; jj += block_size_j) {
                        int iend = std::min(n, ii + block_size_i);
                        int jend = std::min(n, jj + block_size_j);
                        for (int i = ii; i < iend; i++) {
                            double lik = L[i * n + k];
                            double* __restrict__ Arow = &A[i * n];
                            double* __restrict__ Urow = &U[k * n];
                            #pragma omp simd
                            for (int j = jj; j < jend; j++) {
                                Arow[j] -= lik * Urow[j];
                            }
                        }
                    }
                }
            }
            #pragma omp barrier
        } // end for k
    } // end parallel region

    double t_end = omp_get_wtime();
    double factor_time = t_end - t_start;

    delete[] thread_maxvals;
    delete[] thread_maxrows;

    if (singular) {
        std::cerr << "ERROR: Factorization failed: matrix is singular (pivot = 0).\n";
    }

    double l21_norm = 0.0;
    if (!singular) {
        l21_norm = compute_residual_l21_norm(A0, n, piv, L, U);
    }

    std::cout << "LU factorization time: " << factor_time << " seconds.\n";
    if (!singular) {
        std::cout << "Residual L2,1 norm = " << l21_norm << "\n";
    }

    numa_free(A0, (size_t)n * n * sizeof(double));
    numa_free(A, (size_t)n * n * sizeof(double));
    numa_free(L, (size_t)n * n * sizeof(double));
    numa_free(U, (size_t)n * n * sizeof(double));
    free(piv);

    return 0;
}
