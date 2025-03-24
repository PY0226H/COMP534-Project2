/*************************************************************
 * improved_lu_omp.cpp
 *
 * Improved OpenMP-based LU Decomposition with Partial Pivoting.
 * (Now including residual verification)
 *
 * Key improvements:
 *  - Uses an array-of-pointers (2D layout) for matrices so that
 *    row swapping is done in O(1) time.
 *  - Performs the pivot search serially to reduce synchronization.
 *  - In each iteration, uses a new parallel region with minimal barriers:
 *       • #pragma omp single nowait to swap parts of L and copy pivot row to U.
 *       • A parallel for loop (with schedule(static,1)) to update the trailing submatrix.
 *  - Uses memcpy to quickly copy the pivot row.
 *  - After the decomposition, computes the residual L2,1 norm of PA - LU.
 *
 * This code is designed to run efficiently for matrix sizes from 1000 to 8000
 * and thread counts from 1 to 32.
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

// --- Utility Functions ---

// Print usage and exit.
void usage(const char *name) {
    std::cout << "Usage: " << name << " <matrix_size> <num_workers>\n";
    exit(EXIT_FAILURE);
}

// Initialize matrices A, L, U, and pivot array pi.
// A, L, U are allocated as arrays-of-pointers; each row is allocated with numa_alloc_local.
void init_matrix(double **A, double **L, double **U, int *pi, int n, int nworkers) {
    #pragma omp parallel for schedule(static) num_threads(nworkers) default(none) shared(A, L, U, pi, n)
    for (int i = 0; i < n; i++) {
        A[i] = (double*) numa_alloc_local(sizeof(double) * n);
        L[i] = (double*) numa_alloc_local(sizeof(double) * n);
        U[i] = (double*) numa_alloc_local(sizeof(double) * n);
        // Initialize A with random numbers; use srand48 with seed = i for reproducibility.
        srand48(i);
        for (int j = 0; j < n; j++) {
            A[i][j] = drand48();
        }
        // Initialize L as identity and U as zero.
        memset(L[i], 0, sizeof(double) * n);
        memset(U[i], 0, sizeof(double) * n);
        L[i][i] = 1.0;
        // Initialize pivot array.
        pi[i] = i;
    }
}

// Free a matrix allocated as an array-of-pointers.
void free_matrix(double **A, int n) {
    for (int i = 0; i < n; i++) {
        numa_free(A[i], sizeof(double) * n);
    }
    delete[] A;
}

// Simple timer functions using OpenMP wall time.
double timer_start_time = 0.0;
void timer_start() {
    timer_start_time = omp_get_wtime();
}
double timer_elapsed() {
    return omp_get_wtime() - timer_start_time;
}

// --- Main LU Decomposition Code ---

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

    // Allocate arrays-of-pointers for matrices and pivot vector.
    double **A = new double*[n];
    double **L = new double*[n];
    double **U = new double*[n];
    int *pi = new int[n];

    // Initialize matrices A, L, U, and pivot array.
    init_matrix(A, L, U, pi, n, nworkers);

    // *** New: Create a copy of the original A for residual verification ***
    double **A_orig = new double*[n];
    for (int i = 0; i < n; i++) {
        A_orig[i] = new double[n];  // standard allocation
        memcpy(A_orig[i], A[i], sizeof(double) * n);
    }

    // Start the timer.
    timer_start();

    // LU Decomposition with partial pivoting.
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
        // --- Swap rows in A (O(1) pointer swap) ---
        std::swap(A[k], A[k_]);
        // Also swap pivot indices.
        std::swap(pi[k], pi[k_]);

        // --- Parallel region for updating L, U, and trailing submatrix ---
        #pragma omp parallel default(none) shared(k, k_, n, A, L, U)
        {
            // Step 2: Swap the first k elements of L's rows k and k_.
            #pragma omp single nowait
            {
                std::swap_ranges(L[k], L[k] + k, L[k_]);
            }
            // Step 3: Copy A[k][k:] into U[k][k:] using memcpy.
            #pragma omp single nowait
            {
                memcpy(&U[k][k], &A[k][k], sizeof(double) * (n - k));
            }
            // Step 4: Update rows i = n-1 downto k+1.
            #pragma omp for nowait schedule(static,1)
            for (int i = n - 1; i > k; i--) {
                double tmp = A[i][k] / A[k][k];
                L[i][k] = tmp;
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= tmp * A[k][j];
                }
            }
        } // end parallel region
    } // end for k

    double elapsed = timer_elapsed();
    std::cout << "LU decomposition time: " << elapsed << " seconds.\n";

    // --- Residual Verification ---
    // Compute the residual matrix R = PA - L*U and calculate its L2,1 norm.
    // PA is obtained by permuting the rows of the original matrix A_orig according to the pivot vector pi.
    double l21_norm = 0.0;
    for (int j = 0; j < n; j++) {
        double col_norm_sq = 0.0;
        for (int i = 0; i < n; i++) {
            double prod = 0.0;
            int k_end = std::min(i, j);
            for (int k = 0; k <= k_end; k++) {
                prod += L[i][k] * U[k][j];
            }
            // A_orig[pi[i]] is the original row that ended up as row i of PA.
            double diff = A_orig[pi[i]][j] - prod;
            col_norm_sq += diff * diff;
        }
        l21_norm += sqrt(col_norm_sq);
    }
    std::cout << "Residual L2,1 norm: " << l21_norm << "\n";

    // Free allocated memory for A_orig.
    for (int i = 0; i < n; i++) {
        delete[] A_orig[i];
    }
    delete[] A_orig;

    // Free allocated memory for matrices and pivot vector.
    free_matrix(A, n);
    free_matrix(L, n);
    free_matrix(U, n);
    delete[] pi;

    return 0;
}
