/*************************************************************
 * lu-omp.cpp
 *
 * OpenMP-based LU Decomposition with Partial Pivoting.
 * - Uses drand48_r() for thread-safe random number generation.
 * - Single parallel region to minimize overhead.
 * - All shared variables properly protected with barriers/critical.
 * - No static variables that might confuse data-race analysis.
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
  Utility to index into a matrix stored row-major in a 1D array.
*/
inline double& elem(double *matrix, int n, int i, int j) {
    return matrix[(long)i * n + j];
}

/*
  Generate an n x n matrix of random doubles in [0,1) using drand48_r().
  Each thread has a private seed, avoiding data races.
*/
double* generate_random_matrix(int n, int nthreads) {
    double *mat = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!mat) {
        std::cerr << "ERROR: numa_alloc_local failed for matrix.\n";
        exit(EXIT_FAILURE);
    }

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
  Compute the L2,1 norm (sum of Euclidian norms of columns) of (P*A - L*U).

  Steps:
    1) Form P*A by permuting rows of A.
    2) R = PA - L*U.
    3) Sum_{j=0..n-1} sqrt( sum_{i=0..n-1} [R(i,j)]^2 ).
*/
double compute_residual_l21_norm(double *Aorig, int n, int *piv, double *L, double *U) {
    double *PA = (double*) calloc((size_t)n*n, sizeof(double));
    if (!PA) {
        std::cerr << "ERROR: could not allocate memory for PA.\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        int srcRow = piv[i];
        memcpy(&PA[(long)i*n], &Aorig[(long)srcRow*n], n*sizeof(double));
    }

    // R = PA - (L * U)
    double *R = (double*) calloc((size_t)n*n, sizeof(double));
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

    // L2,1 norm
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
    int n         = std::atoi(argv[1]);
    int nthreads  = std::atoi(argv[2]);
    if (n <= 0 || nthreads <= 0) {
        usage(argv[0]);
    }

    omp_set_num_threads(nthreads);

    std::cout << "Running LU Decomposition with row pivoting on a "
              << n << " x " << n
              << " matrix using " << nthreads << " threads.\n";

    // 1. Generate random matrix A0
    double *A0 = generate_random_matrix(n, nthreads);

    // 2. Copy A0 -> A
    double *A = (double*) numa_alloc_local((size_t)n*n*sizeof(double));
    if (!A) {
        std::cerr << "ERROR: numa_alloc_local failed for A.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(A, A0, (size_t)n*n*sizeof(double));

    // 3. Allocate L, U
    double *L = (double*) calloc((size_t)n*n, sizeof(double));
    double *U = (double*) calloc((size_t)n*n, sizeof(double));
    if (!L || !U) {
        std::cerr << "ERROR: failed to allocate L or U.\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        L[(long)i*n + i] = 1.0;  // L's diagonal
    }

    // 4. Pivot array
    int *piv = (int*) malloc(n*sizeof(int));
    if (!piv) {
        std::cerr << "ERROR: failed to allocate pivot array.\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // 5. LU factorization
    bool singular = false;
    double t_start = omp_get_wtime();

    // Single parallel region
    #pragma omp parallel default(none) \
                         shared(A,L,U,piv,n,singular)
    {
        // Shared pivot variables (re-initialized each iteration in #pragma omp single)
        double pivotVal;
        int pivotRow;

        for (int k = 0; k < n; k++) 
        {
            // If singular found already, skip remaining steps
            if (singular) {
                #pragma omp barrier
                continue;
            }

            // 5a. Re-initialize pivotVal, pivotRow in single
            #pragma omp single
            {
                pivotVal = 0.0;
                pivotRow = -1;
            }
            #pragma omp barrier // ensure pivotVal/pivotRow are set to 0 / -1

            // 5b. Parallel pivot search
            double local_maxval = 0.0;
            int    local_maxrow = k;

            #pragma omp for nowait
            for (int i = k; i < n; i++) {
                double val = std::fabs(elem(A,n,i,k));
                if (val > local_maxval) {
                    local_maxval = val;
                    local_maxrow = i;
                }
            }

            // 5c. Update global pivot in critical
            #pragma omp critical
            {
                if (local_maxval > pivotVal) {
                    pivotVal = local_maxval;
                    pivotRow = local_maxrow;
                }
            }
            #pragma omp barrier

            // 5d. One thread checks pivotVal and does row swap
            #pragma omp single
            {
                if (pivotVal == 0.0) {
                    singular = true;
                } else {
                    // Swap pivot array
                    int tmp = piv[k];
                    piv[k] = piv[pivotRow];
                    piv[pivotRow] = tmp;

                    // Swap rows k and pivotRow in A
                    if (pivotRow != k) {
                        for (int j = 0; j < n; j++) {
                            double tmpA = elem(A,n,k,j);
                            elem(A,n,k,j) = elem(A,n,pivotRow,j);
                            elem(A,n,pivotRow,j) = tmpA;
                        }
                        // Swap partial row of L
                        for (int j = 0; j < k; j++) {
                            double tmpL = L[(long)k*n + j];
                            L[(long)k*n + j] = L[(long)pivotRow*n + j];
                            L[(long)pivotRow*n + j] = tmpL;
                        }
                    }

                    // U(k,k) = A(k,k)
                    U[(long)k*n + k] = elem(A,n,k,k);
                }
            }
            #pragma omp barrier

            // If singular, skip further updates
            if (!singular) 
            {
                // 5e. Fill L(i,k) and U(k,i)
                double diagVal = elem(A,n,k,k);
                #pragma omp for
                for (int i = k+1; i < n; i++) {
                    L[(long)i*n + k] = elem(A,n,i,k) / diagVal;
                    U[(long)k*n + i] = elem(A,n,k,i);
                }

                // 5f. Update trailing submatrix
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

    // 6. If singular, print error
    if (singular) {
        std::cerr << "ERROR: Factorization failed (matrix is singular: pivot=0)\n";
        // You could return or exit
    }

    // 7. Compute residual if not singular
    double l21_norm = 0.0;
    if (!singular) {
        l21_norm = compute_residual_l21_norm(A0, n, piv, L, U);
    }

    // 8. Print
    std::cout << "LU factorization time: " << factor_time << " seconds.\n";
    if (!singular) {
        std::cout << "Residual L2,1 norm = " << l21_norm << "\n";
    }

    // Cleanup
    numa_free(A0, (size_t)n*n*sizeof(double));
    numa_free(A,  (size_t)n*n*sizeof(double));
    free(L);
    free(U);
    free(piv);

    return 0;
}
