/*************************************************************
 * lu-omp.cpp
 *
 * OpenMP-based LU Decomposition with Partial Pivoting.
 * - Uses drand48_r() for thread-safe random number generation.
 * - Single parallel region for improved performance (avoids
 *   repeated creation/teardown of parallel teams).
 * - Avoids referencing std::cerr inside parallel with default(none).
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
        // seed each thread's generator differently
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
  Compute the L2,1 norm (sum of Euclidian norms of columns) of (P*A - L*U).

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

    // 3) L2,1 norm = sum of Euclidian norms of columns
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

    // 5) Perform LU factorization with row pivoting
    bool singular = false;  // shared flag to detect singular matrix
    double t_start = omp_get_wtime();

    #pragma omp parallel default(none) \
                         shared(A, L, U, piv, n, singular)
    {
        // These static variables are used for pivot search
        static double maxval;
        static int    maxrow;

        // Per-thread local pivot search
        double local_maxval;
        int    local_maxrow;

        for (int k = 0; k < n; k++)
        {
            // If matrix is already found singular, skip rest
            if (singular) {
                #pragma omp barrier
                continue;
            }

            // Let a single thread reset pivot info
            #pragma omp single
            {
                maxval = 0.0;
                maxrow = -1;
            }
            #pragma omp barrier

            // Each thread searches portion of column k for largest absolute value
            local_maxval = 0.0;
            local_maxrow = k;

            #pragma omp for nowait
            for (int i = k; i < n; i++) {
                double val = std::fabs(elem(A, n, i, k));
                if (val > local_maxval) {
                    local_maxval = val;
                    local_maxrow = i;
                }
            }

            // Combine local results into global pivot choice
            #pragma omp critical
            {
                if (local_maxval > maxval) {
                    maxval = local_maxval;
                    maxrow = local_maxrow;
                }
            }
            #pragma omp barrier

            // One thread checks for singular, performs pivot row swap
            #pragma omp single
            {
                if (maxval == 0.0) {
                    singular = true; // set the flag
                } else {
                    // Swap pivot array entries
                    int tmpP = piv[k];
                    piv[k]   = piv[maxrow];
                    piv[maxrow] = tmpP;

                    // Swap rows k and maxrow of A
                    if (maxrow != k) {
                        for (int j = 0; j < n; j++) {
                            double tmpA = elem(A, n, k, j);
                            elem(A, n, k, j) = elem(A, n, maxrow, j);
                            elem(A, n, maxrow, j) = tmpA;
                        }
                        // Swap partial row of L
                        for (int j = 0; j < k; j++) {
                            double tmpL = L[(long)k*n + j];
                            L[(long)k*n + j] = L[(long)maxrow*n + j];
                            L[(long)maxrow*n + j] = tmpL;
                        }
                    }

                    // U(k,k) = A(k,k)
                    U[(long)k*n + k] = elem(A, n, k, k);
                }
            }
            #pragma omp barrier

            // If singular found, skip updating
            if (!singular)
            {
                // For i=k+1..n-1, set L(i,k) and U(k,i)
                double pivotVal = elem(A, n, k, k);
                #pragma omp for
                for (int i = k+1; i < n; i++) {
                    L[(long)i*n + k] = elem(A, n, i, k) / pivotVal;
                    U[(long)k*n + i] = elem(A, n, k, i);
                }

                // Update the trailing submatrix
                #pragma omp for
                for (int i = k+1; i < n; i++) {
                    double lik = L[(long)i*n + k];
                    for (int j = k+1; j < n; j++) {
                        elem(A,n,i,j) -= lik * U[(long)k*n + j];
                    }
                }
            }
            #pragma omp barrier
        } // end for k
    } // end parallel region

    double t_end = omp_get_wtime();
    double factor_time = t_end - t_start;

    // If singular, print error and optionally exit
    if (singular) {
        std::cerr << "ERROR: Factorization failed: matrix is singular (pivot = 0). \n";
        // If you'd like to stop the program here:
        // exit(EXIT_FAILURE);
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
