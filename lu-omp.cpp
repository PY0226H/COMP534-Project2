#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <numa.h>

static void usage(const char *progName) {
    std::cerr << "Usage: " << progName << " <matrix_size> <num_threads>\n";
    exit(EXIT_FAILURE);
}

/*
  Utility function to return a reference to matrix(i,j) if stored in row-major
  in a 1D array "matrix" of size n*n.
*/
inline double& elem(double *matrix, int n, int i, int j) {
    return matrix[(long)i * n + j];
}

/*
  Generates an n x n matrix of random doubles in [0,1).
  We allocate with numa_alloc_local. You can remove or adapt if needed.
*/
double* generate_random_matrix(int n) {
    double *mat = (double*) numa_alloc_local( (size_t)n * n * sizeof(double) );
    if (!mat) {
        std::cerr << "ERROR: numa_alloc_local failed for matrix.\n";
        exit(EXIT_FAILURE);
    }

    srand48(2023);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n*n; i++) {
        mat[i] = drand48(); 
    }

    return mat;
}

/*
  We will compute the residual's L2,1 norm: sum of Euclidian norms of each column of (P*A - L*U).

  Steps to do this safely:
  1. Form P*A into a temporary matrix PA (by reordering rows of the original A).
  2. Compute R = PA - L*U.
  3. For each column j, compute sqrt( sum_i (R(i,j))^2 ), and sum that over j.
*/
double compute_residual_l21_norm(
    double *Aorig, int n, 
    int *piv,     // pivot array (i.e., pi)
    double *L, 
    double *U
) {
    // 1. Compute PA into a temporary matrix
    double *PA = (double*) calloc((size_t)n * n, sizeof(double));
    if (!PA) {
        std::cerr << "ERROR: could not allocate memory for PA.\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        int srcRow = piv[i];
        memcpy(&PA[(long)i * n], &Aorig[(long)srcRow * n], n * sizeof(double));
    }

    // 2. Allocate R = PA - L*U
    double *R = (double*) calloc((size_t)n * n, sizeof(double));
    if (!R) {
        std::cerr << "ERROR: could not allocate memory for R.\n";
        exit(EXIT_FAILURE);
    }

    // R = PA - L*U
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sumLU = 0.0;
            for (int k = 0; k < n; k++) {
                sumLU += L[(long)i*n + k] * U[(long)k*n + j];
            }
            R[(long)i*n + j] = PA[(long)i*n + j] - sumLU;
        }
    }

    // 3. Now compute the sum of Euclidian norms of columns of R
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

    int n = std::atoi(argv[1]);       // matrix size
    int nthreads = std::atoi(argv[2]); // number of threads

    omp_set_num_threads(nthreads);

    std::cout << "Running LU Decomposition with row pivoting on a " 
              << n << " x " << n 
              << " matrix using " << nthreads << " threads.\n";

    // 1. Generate random matrix A0, which we will keep for computing residual
    double *A0 = generate_random_matrix(n);

    // 2. Copy A0 into A for factorization
    double *A = (double*) numa_alloc_local( (size_t)n * n * sizeof(double) );
    if (!A) {
        std::cerr << "ERROR: numa_alloc_local failed for A.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(A, A0, (size_t)n * n * sizeof(double));

    // 3. Allocate L and U (each n x n)
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

    // 4. Allocate the pivot array
    int *piv = (int*) malloc(n * sizeof(int));
    if (!piv) {
        std::cerr << "ERROR: failed to allocate pivot array.\n";
        exit(EXIT_FAILURE);
    }
    // Initialize pi[i] = i
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // 5. Perform the row-pivoted LU factorization
    double t_start = omp_get_wtime();

    for (int k = 0; k < n; k++) {
        // Pivot search
        double maxval = 0.0;
        int maxrow = -1;
        double local_max, local_row;
        #pragma omp parallel default(none) \
                     private(local_max, local_row) \
                     shared(A, n, k, maxval, maxrow)
        {
            double local_max = 0.0;
            int local_row = -1;

            #pragma omp for
            for (int i = k; i < n; i++) {
                double val = std::fabs(elem(A, n, i, k));
                if (val > local_max) {
                    local_max = val;
                    local_row = i;
                }
            }
            #pragma omp critical
            {
                if (local_max > maxval) {
                    maxval = local_max;
                    maxrow = local_row;
                }
            }
        }

        if (maxval == 0.0) {
            std::cerr << "ERROR: Factorization failed: matrix is singular or nearly singular.\n";
            exit(EXIT_FAILURE);
        }

        // Swap pivot array entries
        {
            int tmp = piv[k];
            piv[k] = piv[maxrow];
            piv[maxrow] = tmp;
        }

        // Swap rows k and maxrow of A
        if (maxrow != k) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < n; j++) {
                double tmp = elem(A, n, k, j);
                elem(A, n, k, j) = elem(A, n, maxrow, j);
                elem(A, n, maxrow, j) = tmp;
            }
            // Also swap L’s partial row
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < k; j++) {
                double tmp = L[(long)k*n + j];
                L[(long)k*n + j] = L[(long)maxrow*n + j];
                L[(long)maxrow*n + j] = tmp;
            }
        }

        // U(k,k) = A(k,k)
        double pivotVal = elem(A, n, k, k);
        U[(long)k*n + k] = pivotVal;

        // For i = k+1..n-1:
        #pragma omp parallel for schedule(static)
        for (int i = k+1; i < n; i++) {
            L[(long)i*n + k] = elem(A, n, i, k) / pivotVal;
            U[(long)k*n + i] = elem(A, n, k, i);
        }

        // Update trailing submatrix
        #pragma omp parallel for schedule(static)
        for (int i = k+1; i < n; i++) {
            double lik = L[(long)i*n + k];
            for (int j = k+1; j < n; j++) {
                elem(A, n, i, j) -= lik * U[(long)k*n + j];
            }
        }
    }

    double t_end = omp_get_wtime();
    double factor_time = t_end - t_start;

    // 6. Compute the residual
    double l21_norm = compute_residual_l21_norm(A0, n, piv, L, U);

    // 7. Print results
    std::cout << "LU factorization time: " << factor_time << " seconds.\n";
    std::cout << "Residual L2,1 norm = " << l21_norm << "\n";

    // Cleanup
    numa_free(A0, (size_t)n * n * sizeof(double));
    numa_free(A,  (size_t)n * n * sizeof(double));
    free(L);
    free(U);
    free(piv);

    return 0;
}
