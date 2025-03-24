/*************************************************************
 * lu-omp.cpp
 * 
 * OpenMP-based LU Decomposition with Partial Pivoting.
 * - Uses drand48_r() to avoid data-races in random generation.
 * - Uses one parallel region for the main loop to reduce overhead.
 * - Proper synchronization in pivot search and row swapping.
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
  Utility to index a matrix stored in row-major layout in 1D array.
*/
inline double& elem(double *matrix, int n, int i, int j) {
    return matrix[(long)i * n + j];
}

/*
  Generate an n x n matrix of random doubles in [0,1) using drand48_r.
  Each thread has its own seed/state to avoid data races.
*/
double* generate_random_matrix(int n, int nthreads) {
    // Allocate matrix in NUMA local memory
    double *mat = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!mat) {
        std::cerr << "ERROR: numa_alloc_local failed for matrix.\n";
        exit(EXIT_FAILURE);
    }

    // Parallel initialization
    #pragma omp parallel num_threads(nthreads)
    {
        struct drand48_data randBuf;
        // Seed each thread differently (example: base seed + threadID).
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
  1) Form P*A by permuting rows of A according to piv.
  2) R = PA - (L*U).
  3) For each column j, compute sqrt( sum_i R(i,j)^2 ), then sum over j.
*/
double compute_residual_l21_norm(double *Aorig, int n, int *piv, double *L, double *U) {
    // 1) Form PA
    double *PA = (double*) calloc((size_t)n * n, sizeof(double));
    if (!PA) {
        std::cerr << "ERROR: could not allocate memory for PA.\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        int srcRow = piv[i];
        // copy row 'srcRow' of Aorig into row 'i' of PA
        memcpy(&PA[(long)i*n], &Aorig[(long)srcRow*n], n * sizeof(double));
    }

    // 2) Compute R = PA - L*U
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

    // 3) Sum of Euclidian norms of columns
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
    int n         = std::atoi(argv[1]);   // matrix size
    int nthreads  = std::atoi(argv[2]);   // number of threads

    if (n <= 0 || nthreads <= 0) {
        usage(argv[0]);
    }

    // Set number of threads for OpenMP
    omp_set_num_threads(nthreads);

    std::cout << "Running LU Decomposition with row pivoting on a "
              << n << " x " << n 
              << " matrix using " << nthreads << " threads.\n";

    // 1) Generate original random matrix A0
    double *A0 = generate_random_matrix(n, nthreads);

    // 2) Copy A0 into A for factorization
    double *A = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!A) {
        std::cerr << "ERROR: numa_alloc_local failed for A.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(A, A0, (size_t)n * n * sizeof(double));

    // 3) Allocate L and U
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
        piv[i] = i; // initial identity permutation
    }

    // 5) Perform LU with row pivoting, timed
    double t_start = omp_get_wtime();

    #pragma omp parallel default(none) \
                         shared(A, L, U, piv, n)
    {
        // These variables will track per-thread local maxima
        double local_maxval;
        int    local_maxrow;

        // Shared pivot stats - must be updated carefully
        static double maxval;
        static int    maxrow;

        for (int k = 0; k < n; k++) 
        {
            // Let exactly one thread reset pivot info before searching
            #pragma omp single
            {
                maxval = 0.0;
                maxrow = -1;
            }
            #pragma omp barrier

            // Each thread searches part of the column to find local max
            local_maxval = 0.0;
            local_maxrow = k;

            #pragma omp for nowait
            for (int i = k; i < n; i++) {
                double val = std::fabs( elem(A, n, i, k) );
                if (val > local_maxval) {
                    local_maxval = val;
                    local_maxrow = i;
                }
            }

            // Combine local maxima into global pivot choice
            #pragma omp critical
            {
                if (local_maxval > maxval) {
                    maxval = local_maxval;
                    maxrow = local_maxrow;
                }
            }
            #pragma omp barrier

            // Pivot row swapping done by a single thread
            #pragma omp single
            {
                if (maxval == 0.0) {
                    std::cerr << "ERROR: Factorization failed (matrix is singular)\n";
                    // If truly singular, you could exit or set a flag...
                }

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

                // Set U(k,k) = A(k,k)
                U[(long)k*n + k] = elem(A, n, k, k);
            }
            #pragma omp barrier

            // Each thread updates L(i,k) and U(k,i) for i=k+1..n-1
            double pivotVal = elem(A, n, k, k);
            #pragma omp for
            for (int i = k+1; i < n; i++) {
                L[(long)i*n + k] = elem(A, n, i, k) / pivotVal;
                U[(long)k*n + i] = elem(A, n, k, i);
            }

            // Update trailing submatrix A(i,j) -= L(i,k)*U(k,j)
            #pragma omp for
            for (int i = k+1; i < n; i++) {
                double lik = L[(long)i*n + k];
                for (int j = k+1; j < n; j++) {
                    elem(A, n, i, j) -= lik * U[(long)k*n + j];
                }
            }

            #pragma omp barrier
        } // end for k
    } // end parallel region

    double t_end = omp_get_wtime();
    double factor_time = t_end - t_start;

    // 6) Compute residual
    double l21_norm = compute_residual_l21_norm(A0, n, piv, L, U);

    // 7) Print results
    std::cout << "LU factorization time: " << factor_time << " seconds.\n"
              << "Residual L2,1 norm = " << l21_norm << "\n";

    // Cleanup
    numa_free(A0, (size_t)n * n * sizeof(double));
    numa_free(A,  (size_t)n * n * sizeof(double));
    free(L);
    free(U);
    free(piv);

    return 0;
}
