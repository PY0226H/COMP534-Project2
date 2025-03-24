/******************************************************************************
 * File:    lu_omp.cpp
 * Author:  (Your Name)
 *
 * Description:
 *   This program performs LU decomposition with partial (row) pivoting on an
 *   n x n matrix A in parallel using OpenMP.
 *   Usage:  ./lu_omp <n> <t>
 *   - n : matrix dimension
 *   - t : number of OpenMP threads
 *
 * Steps:
 *   1) Allocate and initialize A with random doubles in [0,1).
 *   2) Factor A into L and U using partial pivoting (keep track in permutation pi).
 *   3) Compute L2,1 norm of residual (P*A - L*U) to validate correctness.
 *   4) Print residual norm and timing for the factorization.
 *
 ******************************************************************************/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <omp.h>     // for OpenMP
#include <random>    // for C++11 random generator

/******************************************************************************
 * Function: initMatrix
 * --------------------
 * Initializes matrix A (size n x n) with uniform random numbers in [0, 1).
 * A is stored in row-major 1D array of length n*n.
 *****************************************************************************/
void initMatrix(double* A, int n, unsigned int seed)
{
    // Create a Mersenne Twister pseudo-random generator for stable, repeatable results
    // Each thread or each call can have a different seed to reduce correlation if needed.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Fill A row by row
    for(int i = 0; i < n*n; i++) {
        A[i] = dist(gen);
    }
}

/******************************************************************************
 * Function: computeResidualL21
 * ----------------------------
 * Computes the L2,1 norm of the residual (P*A - L*U).
 *
 * Inputs:
 *   A   : Original matrix (row-major)
 *   L   : Lower-triangular matrix from factorization (row-major)
 *   U   : Upper-triangular matrix from factorization (row-major)
 *   pi  : Permutation array of length n; pi[k] gives the row in A that was
 *         permuted to row k in the factorization.
 *   n   : dimension
 *
 * Return:
 *   double - the sum of Euclidean norms of each column of (P*A - L*U).
 *****************************************************************************/
double computeResidualL21(const double* A,
                          const double* L,
                          const double* U,
                          const std::vector<int>& pi,
                          int n)
{
    double norm_sum = 0.0;

    // For each column j, compute the Euclidean norm of column j of (P*A - L*U).
    for(int j = 0; j < n; j++) {
        double col_norm_sq = 0.0;
        // Compute each entry i of column j in the residual
        for(int i = 0; i < n; i++) {
            // P*A row i is actually row pi[i] of A
            double pa_ij = A[ pi[i]*n + j ];

            // (L*U)(i,j) = sum_{k=0..n-1} L(i,k)*U(k,j)
            // But effectively, L is lower-triangular and U is upper-triangular
            // so many terms are zero. For correctness, we do the full sum.
            double lu_val = 0.0;
            for(int k = 0; k < n; k++) {
                lu_val += L[i*n + k] * U[k*n + j];
            }
            double diff = pa_ij - lu_val;
            col_norm_sq += diff * diff;
        }
        double col_norm = std::sqrt(col_norm_sq);
        norm_sum += col_norm;
    }

    return norm_sum;
}

/******************************************************************************
 * Function: lu_decomposition
 * --------------------------
 * Performs LU decomposition with partial (row) pivoting on matrix A in-place,
 * producing L and U, plus a permutation array pi representing P.
 *
 *   A (in/out): row-major array of size n*n. Overwritten during process.
 *   L        : row-major array for L (n*n)
 *   U        : row-major array for U (n*n)
 *   pi       : vector<int> of length n for pivot info
 *   n        : dimension
 *
 * The final factorization is: P*A = L*U.
 *
 * Steps:
 *   1) Initialize pi, L (identity on diagonal), U (0).
 *   2) For k = 0..n-1:
 *       a) Find pivot row (k') with max |A(i,k)|, i in [k..n-1].
 *       b) If max == 0 => singular matrix => error.
 *       c) Swap pi[k] and pi[k'], swap row k and k' in A, and row k and k' of L up to k-1.
 *       d) U(k,k) = A(k,k).
 *       e) For i>k: L(i,k) = A(i,k)/U(k,k), U(k,i) = A(k,i).
 *       f) Update trailing submatrix of A.
 *****************************************************************************/
void lu_decomposition(double* A,
                      double* L,
                      double* U,
                      std::vector<int>& pi,
                      int n)
{
    // 1) Initialize pi, L, U
    for(int i = 0; i < n; i++) {
        pi[i] = i;
    }

    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            L[i*n + j] = (i == j) ? 1.0 : 0.0; // identity for L
            U[i*n + j] = 0.0;                 // zero for U
        }
    }

    // 2) Main loop over columns k
    for(int k = 0; k < n; k++) {
        // a) Partial pivot search for row with max |A(i,k)|
        double max_val = 0.0;
        int kprime = k;

        {
            // Parallel region to find pivot
            #pragma omp parallel
            {
                double local_max = 0.0;
                int local_index = k;

                #pragma omp for nowait
                for(int i = k; i < n; i++){
                    double val = std::fabs(A[i*n + k]);
                    if(val > local_max) {
                        local_max  = val;
                        local_index = i;
                    }
                }
                // Combine results
                #pragma omp critical
                {
                    if(local_max > max_val) {
                        max_val = local_max;
                        kprime = local_index;
                    }
                }
            }
        }

        if(max_val == 0.0) {
            std::cerr << "Error: singular matrix (zero pivot found)\n";
            std::exit(EXIT_FAILURE);
        }

        // b) Swap rows in pi, A, and L (up to column k-1)
        if(kprime != k) {
            // swap pi
            int tmp_pi = pi[k];
            pi[k] = pi[kprime];
            pi[kprime] = tmp_pi;

            // swap row k and row kprime in A
            #pragma omp parallel for
            for(int col = 0; col < n; col++) {
                double tmp = A[k*n + col];
                A[k*n + col] = A[kprime*n + col];
                A[kprime*n + col] = tmp;
            }

            // swap L up to column k-1
            #pragma omp parallel for
            for(int col = 0; col < k; col++) {
                double tmp = L[k*n + col];
                L[k*n + col] = L[kprime*n + col];
                L[kprime*n + col] = tmp;
            }
        }

        // c) U(k,k) = A(k,k)
        double pivotVal = A[k*n + k];
        U[k*n + k] = pivotVal;

        // d) Fill in L(i,k) and U(k,i) for i in [k+1..n-1]
        #pragma omp parallel for
        for(int i = k+1; i < n; i++) {
            L[i*n + k] = A[i*n + k] / pivotVal; // below diagonal
            U[k*n + i] = A[k*n + i];           // on/above diagonal
        }

        // e) Update trailing submatrix of A
        //    A(i,j) -= L(i,k) * U(k,j), for i>k, j>k
        #pragma omp parallel for
        for(int i = k+1; i < n; i++){
            for(int j = k+1; j < n; j++){
                A[i*n + j] -= L[i*n + k] * U[k*n + j];
            }
        }
    } // end for k
}

int main(int argc, char* argv[])
{
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    int t = std::atoi(argv[2]);

    // 1) Set number of OMP threads
    omp_set_num_threads(t);

    // 2) Allocate memory for A, L, U
    //    We'll store them in row-major form as 1D arrays of length n*n.
    double* A = new double[n * n];
    double* L = new double[n * n];
    double* U = new double[n * n];

    // 3) Initialize A with random values
    initMatrix(A, n, /*seed=*/12345);

    // 4) Prepare permutation array pi
    std::vector<int> pi(n);

    // 5) Time the LU decomposition
    double start_time = omp_get_wtime();
    lu_decomposition(A, L, U, pi, n);
    double end_time = omp_get_wtime();

    double factor_time = end_time - start_time;

    // 6) Compute L2,1 norm of residual (P*A - L*U)
    //    This step is for correctness verification. Typically done in serial.
    double residual_norm = computeResidualL21(A, L, U, pi, n);

    // 7) Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Matrix size (n): " << n << "\n";
    std::cout << "Number of threads: " << t << "\n";
    std::cout << "LU Decomposition Time (sec): " << factor_time << "\n";
    std::cout << "Residual L2,1 Norm: " << residual_norm << "\n";

    // 8) Cleanup
    delete[] A;
    delete[] L;
    delete[] U;

    return 0;
}
