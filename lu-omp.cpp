/*************************************************************
 * improved_lu_omp.cpp
 *
 * Improved OpenMP-based LU Decomposition with Partial Pivoting.
 *
 * Key improvements:
 *  - Uses an array-of-pointers (2D layout) for matrices so that
 *    row swapping is done in O(1) time.
 *  - Performs a serial pivot search in each iteration to reduce
 *    synchronization overhead.
 *  - For each iteration, uses a small parallel region with:
 *       • a single nowait section to swap parts of L and copy the pivot row to U,
 *       • a parallel for loop (with schedule(static,1)) to update the trailing submatrix.
 *  - Uses memcpy for fast copying of the pivot row.
 *  - Computes and prints the residual L2,1 norm (i.e. the sum over columns of the
 *    Euclidean norms of \(PA - LU\)) as required.
 *
 * This code conforms with the project requirements and is designed to
 * run efficiently for matrix sizes from 1000 to 8000 and thread counts from 1 to 32.
 *************************************************************/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <numa.h>
#include <sched.h> // for sched_getcpu()

// --------------------
// Utility Functions
// --------------------

// Print usage and exit.
void usage(const char *name) {
    std::cout << "Usage: " << name << " <matrix_size> <num_workers>\n";
    exit(EXIT_FAILURE);
}

// Timer functions using OpenMP wall time.
double timer_start_time = 0.0;
void timer_start() {
    timer_start_time = omp_get_wtime();
}
double timer_elapsed() {
    return omp_get_wtime() - timer_start_time;
}

// Initialize matrices A, L, U and pivot vector pi.
// Matrices are allocated as an array-of-pointers so that each row is allocated
// separately using numa_alloc_local. This allows row swaps via pointer swaps.
void init_matrix(double **A, double **L, double **U, int *pi, int n, int nworkers) {
    #pragma omp parallel for schedule(static) num_threads(nworkers) default(none) shared(A, L, U, pi, n)
    for (int i = 0; i < n; i++) {
        A[i] = (double*) numa_alloc_local(sizeof(double) * n);
        L[i] = (double*) numa_alloc_local(sizeof(double) * n);
        U[i] = (double*) numa_alloc_local(sizeof(double) * n);
        // Use srand48 with seed = i for reproducibility.
        srand48(i);
        for (int j = 0; j < n; j++) {
            A[i][j] = drand48();
        }
        // Initialize L as identity, U as zeros.
        memset(L[i], 0, sizeof(double) * n);
        memset(U[i], 0, sizeof(double) * n);
        L[i][i] = 1.0;
        pi[i] = i;
    }
}

// Create a deep copy of matrix A (allocated as an array-of-pointers).
double** deep_copy_matrix(double **A, int n) {
    double **copy = new double*[n];
    for (int i = 0; i < n; i++) {
        copy[i] = new double[n];
        memcpy(copy[i], A[i], sizeof(double) * n);
    }
    return copy;
}

// Free a matrix allocated as an array-of-pointers using numa_alloc_local.
void free_matrix(double **A, int n) {
    for (int i = 0; i < n; i++) {
        numa_free(A[i], sizeof(double) * n);
    }
    delete[] A;
}

// Free a deep-copied matrix allocated with new.
void free_deep_matrix(double **A, int n) {
    for (int i = 0; i < n; i++) {
        delete[] A[i];
    }
    delete[] A;
}

// Compute the residual L2,1 norm of (PA - LU).
// PA is formed by applying the pivot vector pi to A_orig, i.e. PA[i] = A_orig[pi[i]].
// The L2,1 norm is defined as the sum over columns j of the Euclidean norm of column j.
double compute_residual_norm(int n, int *pi, double **A_orig, double **L, double **U) {
    double norm = 0.0;
    // For each column j:
    for (int j = 0; j < n; j++) {
        double col_sum = 0.0;
        for (int i = 0; i < n; i++) {
            // PA[i][j] is the pivoted row from A_orig.
            double PA_ij = A_orig[pi[i]][j];
            // Compute the (i,j) entry of L*U.
            double LU_ij = 0.0;
            for (int k = 0; k < n; k++) {
                LU_ij += L[i][k] * U[k][j];
            }
            double diff = PA_ij - LU_ij;
            col_sum += diff * diff;
        }
        norm += std::sqrt(col_sum);
    }
    return norm;
}

// --------------------
// Main LU Decomposition
// --------------------

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
    }
    int n = std::atoi(argv[1]);         // Matrix size.
    int nworkers = std::atoi(argv[2]);    // Number of threads.
    if(n <= 0 || nworkers <= 0) {
        usage(argv[0]);
    }
    omp_set_num_threads(nworkers);
    std::cout << "Running improved LU Decomposition on a " << n << "x" << n 
              << " matrix using " << nworkers << " threads.\n";

    // Allocate arrays-of-pointers for matrices A, L, U and pivot vector.
    double **A = new double*[n];
    double **L = new double*[n];
    double **U = new double*[n];
    int *pi = new int[n];

    // Initialize matrices.
    init_matrix(A, L, U, pi, n, nworkers);

    // Create a deep copy of A for the residual norm check.
    double **A_orig = deep_copy_matrix(A, n);

    timer_start();

    // LU Factorization Loop.
    for (int k = 0; k < n; k++) {
        double max_val = 0.0;
        int k_ = k;
        // --- Serial pivot search ---
        for (int i = k; i < n; i++) {
            double abs_val = fabs(A[i][k]);
            if (abs_val > max_val) {
                max_val = abs_val;
                k_ = i;
            }
        }
        if (max_val == 0.0) {
            std::cerr << "Matrix is singular at column " << k << "\n";
            exit(EXIT_FAILURE);
        }
        // --- Swap rows in A (pointer swap) and update pivot vector ---
        std::swap(A[k], A[k_]);
        std::swap(pi[k], pi[k_]);

        // --- Parallel region for updating L, U and trailing submatrix ---
        #pragma omp parallel default(none) shared(k, k_, n, A, L, U, nworkers)
        {
            // Step 2: Swap the first k elements of L's rows (for row k and k_).
            #pragma omp single nowait
            {
                std::swap_ranges(L[k], L[k] + k, L[k_]);
            }
            // Step 3: Copy A[k][k:] into U[k][k:].
            #pragma omp single nowait
            {
                memcpy(&U[k][k], &A[k][k], sizeof(double) * (n - k));
            }
            // Step 4: Update trailing submatrix rows from i = n-1 downto k+1.
            #pragma omp for nowait schedule(static,1)
            for (int i = n - 1; i > k; i--) {
                double tmp = A[i][k] / A[k][k];
                L[i][k] = tmp;
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= tmp * A[k][j];
                }
            }
        } // end parallel region
    }
    double elapsed = timer_elapsed();
    std::cout << "LU decomposition time: " << elapsed << " seconds.\n";

    // Compute the residual L2,1 norm.
    double residual = compute_residual_norm(n, pi, A_orig, L, U);
    std::cout << "Residual L2,1 norm = " << residual << "\n";

    // Free allocated memory.
    free_matrix(A, n);
    free_matrix(L, n);
    free_matrix(U, n);
    delete[] pi;
    free_deep_matrix(A_orig, n);

    return 0;
}
