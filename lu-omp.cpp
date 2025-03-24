/*************************************************************
 * lu-omp.cpp
 *
 * Improved OpenMP-based LU Decomposition with Partial Pivoting.
 * - Uses drand48_r() for thread-safe random number generation.
 * - Single parallel region for improved performance (avoids
 *   repeated creation/teardown of parallel teams).
 * - Each thread computes its local maximum for the pivot search,
 *   and the reduction is performed by a single thread.
 * - Temporary reduction arrays are allocated outside the parallel
 *   region to avoid cross-thread stack accesses.
 *************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <numa.h>

/*
  Print usage message.
*/
static void usage(const char *progName) {
    std::cerr << "Usage: " << progName << " <matrix_size> <num_threads>\n";
    exit(EXIT_FAILURE);
}

/*
  Utility for indexing a matrix stored in row-major layout as a 1D array.
*/
inline double& elem(double *matrix, int n, int i, int j) {
    return matrix[(long)i * n + j];
}

/*
  Generates an n x n matrix of random doubles in [0,1) using drand48_r().
  Each thread has its own seed/state, which avoids data races.
*/
double* generate_random_matrix(int n, int nthreads) {
    // NUMA local allocation
    double *mat = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!mat) {
        std::cerr << "ERROR: numa_alloc_local failed for matrix.\n";
        exit(EXIT_FAILURE);
    }

    // Parallel initialization with thread-local random state
    #pragma omp parallel num_threads(nthreads)
    {
        struct drand48_data randBuf;
        // Seed each thread's generator differently
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

/*
  Compute the L2,1 norm (sum of Euclidean norms of columns) of (P*A - L*U).

  Steps:
    1) Form PA by permuting rows of A according to piv.
    2) Compute R = PA - L*U.
    3) Sum_{j=0..n-1} sqrt( sum_{i=0..n-1} R(i,j)^2 ).
*/
double compute_residual_l21_norm(double *Aorig, int n, int *piv, double *L, double *U) {
    // 1) Form P*A
    double *PA = (double*) calloc((size_t)n * n, sizeof(double));
    if (!PA) {
        std::cerr << "ERROR: could not allocate memory for PA.\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        int srcRow = piv[i];
        // Copy row srcRow of Aorig into row i of PA
        memcpy(&PA[(long)i*n], &Aorig[(long)srcRow*n], n * sizeof(double));
    }

    // 2) Compute R = PA - (L*U)
    double *R = (double*) calloc((size_t)n * n, sizeof(double));
    if (!R) {
        std::cerr << "ERROR: could not allocate memory for R.\n";
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sumLU = 0.0;
            for (int k = 0; k < n; k++) {
                sumLU += L[(long)i*n + k] * U[(long)k*n + j];
            }
            R[(long)i*n + j] = PA[(long)i*n + j] - sumLU;
        }
    }

    // 3) L2,1 norm = sum of Euclidean norms of columns
    double l21 = 0.0;
    for (int j = 0; j < n; j++) {
        double sumSq = 0.0;
        for (int i = 0; i < n; i++) {
            double val = R[(long)i*n + j];
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
    int n         = std::atoi(argv[1]);  // matrix size
    int nthreads  = std::atoi(argv[2]);  // number of threads

    if (n <= 0 || nthreads <= 0) {
        usage(argv[0]);
    }

    // Set number of threads for OpenMP
    omp_set_num_threads(nthreads);

    std::cout << "Running LU Decomposition with row pivoting on a "
              << n << " x " << n 
              << " matrix using " << nthreads << " threads.\n";

    // 1) Generate random matrix (keep copy for residual check)
    double *A0 = generate_random_matrix(n, nthreads);

    // 2) Copy A0 to A for factorization
    double *A = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!A) {
        std::cerr << "ERROR: numa_alloc_local failed for A.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(A, A0, (size_t)n * n * sizeof(double));

    // 3) Allocate L, U
    double *L = (double*) calloc((size_t)n * n, sizeof(double));
    double *U = (double*) calloc((size_t)n * n, sizeof(double));
    if (!L || !U) {
        std::cerr << "ERROR: failed to allocate L or U.\n";
        exit(EXIT_FAILURE);
    }
    // Initialize L’s diagonal to 1.0
    for (int i = 0; i < n; i++) {
        L[(long)i*n + i] = 1.0;
    }

    // 4) Allocate pivot array
    int *piv = (int*) malloc(n * sizeof(int));
    if (!piv) {
        std::cerr << "ERROR: failed to allocate pivot array.\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // Allocate temporary reduction arrays for pivot search.
    // These arrays are sized to the number of threads and shared by all threads.
    double *thread_maxvals = new double[nthreads];
    int    *thread_maxrows = new int[nthreads];

    // 5) Perform LU factorization with row pivoting
    bool singular = false;  // shared flag to detect singular matrix
    double t_start = omp_get_wtime();

    #pragma omp parallel default(none) shared(A, L, U, piv, n, singular, thread_maxvals, thread_maxrows)
    {
        for (int k = 0; k < n; k++)
        {
            // If matrix is singular, synchronize and skip further work.
            if (singular) {
                #pragma omp barrier
                continue;
            }

            int tid = omp_get_thread_num();
            double local_maxval = 0.0;
            int local_maxrow = k;

            // Each thread searches its assigned chunk of rows in column k.
            #pragma omp for nowait
            for (int i = k; i < n; i++) {
                double val = std::fabs(elem(A, n, i, k));
                if (val > local_maxval) {
                    local_maxval = val;
                    local_maxrow = i;
                }
            }
            // Write each thread’s local result to the shared arrays.
            thread_maxvals[tid] = local_maxval;
            thread_maxrows[tid] = local_maxrow;

            #pragma omp barrier  // Ensure all threads have written their results

            // One thread performs the reduction.
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
                    singular = true; // pivot is zero -> singular matrix
                } else {
                    // Swap pivot array entries.
                    int tmpP = piv[k];
                    piv[k]   = piv[maxrow];
                    piv[maxrow] = tmpP;

                    // Swap rows k and maxrow of A, if needed.
                    if (maxrow != k) {
                        for (int j = 0; j < n; j++) {
                            double tmpA = elem(A, n, k, j);
                            elem(A, n, k, j) = elem(A, n, maxrow, j);
                            elem(A, n, maxrow, j) = tmpA;
                        }
                        // Swap the partial row of L.
                        for (int j = 0; j < k; j++) {
                            double tmpL = L[(long)k*n + j];
                            L[(long)k*n + j] = L[(long)maxrow*n + j];
                            L[(long)maxrow*n + j] = tmpL;
                        }
                    }
                    // Set U(k,k) = A(k,k)
                    U[(long)k*n + k] = elem(A, n, k, k);
                }
            }
            #pragma omp barrier  // Ensure pivot decision is visible

            // If singular, skip further updates.
            if (!singular)
            {
                double pivotVal = elem(A, n, k, k);
                // Update L(i,k) and U(k,i) for rows i = k+1..n-1.
                #pragma omp for
                for (int i = k+1; i < n; i++) {
                    L[(long)i*n + k] = elem(A, n, i, k) / pivotVal;
                    U[(long)k*n + i] = elem(A, n, k, i);
                }
                // Update the trailing submatrix.
                #pragma omp for
                for (int i = k+1; i < n; i++) {
                    double lik = L[(long)i*n + k];
                    for (int j = k+1; j < n; j++) {
                        elem(A, n, i, j) -= lik * U[(long)k*n + j];
                    }
                }
            }
            #pragma omp barrier  // Synchronize at end of iteration k
        } // end for k
    } // end parallel region

    double t_end = omp_get_wtime();
    double factor_time = t_end - t_start;

    // Free the temporary reduction arrays.
    delete[] thread_maxvals;
    delete[] thread_maxrows;

    // If singular, print error.
    if (singular) {
        std::cerr << "ERROR: Factorization failed: matrix is singular (pivot = 0).\n";
    }

    // 6) Compute the residual L2,1 norm (only if not singular)
    double l21_norm = 0.0;
    if (!singular) {
        l21_norm = compute_residual_l21_norm(A0, n, piv, L, U);
    }

    // 7) Print results
    std::cout << "LU factorization time: " << factor_time << " seconds.\n";
    if (!singular) {
        std::cout << "Residual L2,1 norm = " << l21_norm << "\n";
    }

    // Cleanup
    numa_free(A0, (size_t)n * n * sizeof(double));
    numa_free(A,  (size_t)n * n * sizeof(double));
    free(L);
    free(U);
    free(piv);

    return 0;
}
