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
  Utility function to return a reference to A(i,j) if stored in row-major
  in a 1D array "A" of size n*n.
*/
inline double& A(double *matrix, int n, int i, int j) {
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

    // Initialize the random number generator. For repeatability,
    // you might want a fixed seed. For truly random, seed with time.
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
        // row i of P*A is pivot[i]-th row of Aorig
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
    // We'll do: for each i,j
    //    R(i,j) = PA(i,j) - sum_{k=0..n-1} L(i,k)*U(k,j)
    // But L, U are both triangular. We can do a triple nested loop carefully.
    // For correctness, let's do the straightforward approach.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sumLU = 0.0;
            for (int k = 0; k < n; k++) {
                // If k>i, L(i,k)=0. If k<j, U(k,j)=0. But let's skip that optimization for clarity.
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

    int n = std::atoi(argv[1]);    // matrix size
    int nthreads = std::atoi(argv[2]); // number of threads

    omp_set_num_threads(nthreads);

    std::cout << "Running LU Decomposition with row pivoting on a " 
              << n << " x " << n 
              << " matrix using " << nthreads << " threads.\n";

    // 1. Generate random matrix A0, which we will keep for computing residual
    double *A0 = generate_random_matrix(n);

    // 2. Copy A0 into A for factorization, because we must not destroy the original
    double *A = (double*) numa_alloc_local( (size_t)n * n * sizeof(double) );
    if (!A) {
        std::cerr << "ERROR: numa_alloc_local failed for A.\n";
        exit(EXIT_FAILURE);
    }

    memcpy(A, A0, (size_t)n * n * sizeof(double));

    // 3. Allocate L and U (each n x n). 
    //    We'll store them fully as n*n, but only use lower-triangle or upper-triangle as needed.
    double *L = (double*) calloc((size_t)n * n, sizeof(double));
    double *U = (double*) calloc((size_t)n * n, sizeof(double));
    if (!L || !U) {
        std::cerr << "ERROR: failed to allocate L or U.\n";
        exit(EXIT_FAILURE);
    }
    // Initialize L’s diagonal to 1.0
    for (int i = 0; i < n; i++) {
        L[(long)i * n + i] = 1.0;
    }

    // 4. Allocate the pivot array pi, which will track row permutations
    //    We'll store the row index that ends up in the i-th row after pivoting.
    int *piv = (int*) malloc(n * sizeof(int));
    if (!piv) {
        std::cerr << "ERROR: failed to allocate pivot array.\n";
        exit(EXIT_FAILURE);
    }
    // Initialize pi[i] = i
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // 5. Perform the row-pivoted LU factorization using the pseudocode structure
    double t_start = omp_get_wtime();

    for (int k = 0; k < n; k++) {
        // Pivot search: find row r >= k that has the largest absolute A(r, k)
        double maxval = 0.0;
        int maxrow = -1;

        // We can parallelize the pivot search
        #pragma omp parallel
        {
            double local_max = 0.0;
            int local_row = -1;

            #pragma omp for nowait
            for (int i = k; i < n; i++) {
                double val = std::fabs(A(A, n, i, k));
                if (val > local_max) {
                    local_max = val;
                    local_row = i;
                }
            }
            // Reduce the local maxima
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

        // 6. Swap pivot array entries piv[k] and piv[maxrow]
        {
            int tmp = piv[k];
            piv[k] = piv[maxrow];
            piv[maxrow] = tmp;
        }

        // 7. Swap entire row k and row maxrow of A 
        if (maxrow != k) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < n; j++) {
                double tmp = A(A, n, k, j);
                A(A, n, k, j) = A(A, n, maxrow, j);
                A(A, n, maxrow, j) = tmp;
            }
            // Also swap L’s partial row (k, 0..k-1) with (maxrow, 0..k-1)
            // but L is stored separately. We only swap columns < k in L 
            // because columns >= k haven’t been set yet.
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < k; j++) {
                double tmp = L[(long)k*n + j];
                L[(long)k*n + j] = L[(long)maxrow*n + j];
                L[(long)maxrow*n + j] = tmp;
            }
        }

        // 8. Set U(k,k) = A(k,k)
        double pivotVal = A(A, n, k, k);
        U[(long)k*n + k] = pivotVal;

        // 9. For i = k+1..n-1:
        //       L(i,k) = A(i,k)/U(k,k)
        //       U(k,i) = A(k,i)
        #pragma omp parallel for schedule(static)
        for (int i = k+1; i < n; i++) {
            L[(long)i*n + k] = A(A, n, i, k) / pivotVal;
            U[(long)k*n + i] = A(A, n, k, i);
        }

        // 10. For i = k+1..n-1:
        //        For j = k+1..n-1:
        //           A(i,j) -= L(i,k) * U(k,j)
        #pragma omp parallel for schedule(static)
        for (int i = k+1; i < n; i++) {
            double lik = L[(long)i*n + k];
            for (int j = k+1; j < n; j++) {
                A(A, n, i, j) -= lik * U[(long)k*n + j];
            }
        }
    }

    double t_end = omp_get_wtime();
    double factor_time = t_end - t_start;

    // 11. Now we have L and U. Let's check the residual. 
    //     Because we destroyed A while factoring, we use the original A0
    //     but re-order rows according to piv[] to compute (P*A0 - L*U).

    double l21_norm = compute_residual_l21_norm(A0, n, piv, L, U);

    // 12. Print results
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
