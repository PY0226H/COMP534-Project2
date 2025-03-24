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

// Utility function to access A(i,j)
inline double& elem(double *matrix, int n, int i, int j) {
    return matrix[(long)i * n + j];
}

// Generate an n x n matrix of random doubles in [0,1).
double* generate_random_matrix(int n, int nthreads)
{
    // NUMA local allocation for big matrix
    double *mat = (double*) numa_alloc_local( (size_t)n * n * sizeof(double) );
    if (!mat) {
        std::cerr << "ERROR: numa_alloc_local failed for matrix.\n";
        exit(EXIT_FAILURE);
    }

    // Parallel initialization with per-thread random state
    #pragma omp parallel num_threads(nthreads)
    {
        struct drand48_data randBuf;
        // seed each thread's generator differently 
        // (just an example: 2023 + threadID)
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

// Compute the residual's L2,1 norm: sum of the Euclidian norm of columns of (P*A - L*U)
double compute_residual_l21_norm(
    double *Aorig, int n, 
    int *piv,     // pivot array
    double *L, 
    double *U
) {
    // 1. Form P*A into a temporary matrix
    double *PA = (double*) calloc((size_t)n * n, sizeof(double));
    if (!PA) {
        std::cerr << "ERROR: could not allocate memory for PA.\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++) {
        int srcRow = piv[i];
        memcpy(&PA[(long)i*n], &Aorig[(long)srcRow*n], n * sizeof(double));
    }

    // 2. Compute R = PA - L*U
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

    // 3. Compute sum of Euclidian norms of columns
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

    int n        = std::atoi(argv[1]); // matrix size
    int nthreads = std::atoi(argv[2]); // number of threads

    omp_set_num_threads(nthreads);

    std::cout << "Running LU Decomposition with row pivoting on a " 
              << n << " x " << n 
              << " matrix using " << nthreads << " threads.\n";

    // 1. Generate random matrix A0, used later for residual check
    double *A0 = generate_random_matrix(n);

    // 2. Copy A0 into A for factorization
    double *A = (double*) numa_alloc_local( (size_t)n * n * sizeof(double) );
    if (!A) {
        std::cerr << "ERROR: numa_alloc_local failed for A.\n";
        exit(EXIT_FAILURE);
    }
    memcpy(A, A0, (size_t)n * n * sizeof(double));

    // 3. Allocate L and U
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

    // 5. Perform LU factorization with row pivoting
    double t_start = omp_get_wtime();

    for (int k = 0; k < n; k++) {

        // === Pivot search ===
        double maxval = 0.0;
        int    maxrow = -1;

        // Parallel max search in column k
        #pragma omp parallel
        {
            double my_maxval = 0.0;
            int    my_maxrow = -1;

            #pragma omp for nowait
            for (int i = k; i < n; i++) {
                double val = std::fabs(elem(A, n, i, k));
                if (val > my_maxval) {
                    my_maxval = val;
                    my_maxrow = i;
                }
            }

            // Combine partial maxima
            #pragma omp critical
            {
                if (my_maxval > maxval) {
                    maxval = my_maxval;
                    maxrow = my_maxrow;
                }
            }
        }

        if (maxval == 0.0) {
            std::cerr << "ERROR: Factorization failed: matrix is singular.\n";
            exit(EXIT_FAILURE);
        }

        // === Row swap in pivot array ===
        {
            int tmp = piv[k];
            piv[k] = piv[maxrow];
            piv[maxrow] = tmp;
        }

        // === Swap rows k and maxrow of A ===
        if (maxrow != k) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < n; j++) {
                double tmp = elem(A, n, k, j);
                elem(A, n, k, j)     = elem(A, n, maxrow, j);
                elem(A, n, maxrow, j) = tmp;
            }

            // also swap partial rows of L
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < k; j++) {
                double tmp = L[(long)k*n + j];
                L[(long)k*n + j]       = L[(long)maxrow*n + j];
                L[(long)maxrow*n + j]  = tmp;
            }
        }

        // U(k,k) = A(k,k)
        double pivotVal = elem(A, n, k, k);
        U[(long)k*n + k] = pivotVal;

        // === Compute L(i,k) and U(k,i) for i>k
        #pragma omp parallel for schedule(static)
        for (int i = k+1; i < n; i++) {
            L[(long)i*n + k] = elem(A, n, i, k) / pivotVal;
            U[(long)k*n + i] = elem(A, n, k, i);
        }

        // === Update trailing submatrix A(i,j) for i>k, j>k
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = k+1; i < n; i++) {
            for (int j = k+1; j < n; j++) {
                elem(A, n, i, j) -= L[(long)i*n + k] * U[(long)k*n + j];
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
