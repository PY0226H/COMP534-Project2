#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <numa.h>
#include <iostream>

//---------------------------------------------------------------------
// Helper for indexing a row-major matrix stored in a 1D array
//---------------------------------------------------------------------
inline double& elem(double *A, int n, int i, int j) {
    return A[(long)i * n + j];
}

//---------------------------------------------------------------------
// Generate a random n x n matrix in [0,1).
// Each thread seeds a drand48_r() generator to avoid data races.
//---------------------------------------------------------------------
double* generate_random_matrix(int n, int nthreads) {
    double *mat = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    if (!mat) {
        std::cerr << "ERROR: numa_alloc_local failed for matrix.\n";
        std::exit(EXIT_FAILURE);
    }

    #pragma omp parallel num_threads(nthreads)
    {
        struct drand48_data randBuf;
        // seed each thread’s generator differently
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

//---------------------------------------------------------------------
// Compute the L2,1 norm of (P*A - L*U).
//   1) Form PA by permuting rows of Aorig according to piv.
//   2) R = PA - L*U
//   3) L2,1 = sum_{j=0..n-1} sqrt( sum_{i=0..n-1} R(i,j)^2 )
//---------------------------------------------------------------------
double compute_residual_l21_norm(double *Aorig, int n, int *piv,
                                 double *L, double *U)
{
    double *PA = (double*) calloc((size_t)n*n, sizeof(double));
    if (!PA) {
        std::cerr << "ERROR: could not allocate PA.\n";
        std::exit(EXIT_FAILURE);
    }

    // 1) Form P*A
    for (int i = 0; i < n; i++) {
        int src = piv[i];
        std::memcpy(&PA[(long)i*n], &Aorig[(long)src*n], n*sizeof(double));
    }

    // 2) R = PA - L*U
    double *R = (double*) calloc((size_t)n*n, sizeof(double));
    if (!R) {
        std::cerr << "ERROR: could not allocate R.\n";
        std::exit(EXIT_FAILURE);
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

    // 3) L2,1 norm
    double l21 = 0.0;
    for (int j = 0; j < n; j++) {
        double sumSq = 0.0;
        for (int i = 0; i < n; i++) {
            double val = R[(long)i*n + j];
            sumSq += val*val;
        }
        l21 += std::sqrt(sumSq);
    }

    free(PA);
    free(R);
    return l21;
}

//---------------------------------------------------------------------
// Factor a small "panel" block [k..k+kb-1, k..k+kb-1] using partial pivoting.
// This is a standard "unblocked" LU factorization of a submatrix.
// We store pivot information into the global pivot array "piv" and
// also perform row swaps in the entire matrix A (for columns k..n-1).
// L and U are updated for the block columns as well.
//---------------------------------------------------------------------
bool factor_block(double *A, double *L, double *U, int *piv,
                  int n, int k, int kb)
{
    // We'll do partial pivoting in the sub-block columns from k..k+kb-1
    // but the pivot search goes from row k..n-1, since the entire trailing
    // submatrix might be affected by pivoting.
    for (int col = k; col < k+kb; col++) {
        // 1) Pivot search in column col
        double maxval = 0.0;
        int maxrow = col;
        for (int i = col; i < n; i++) {
            double val = std::fabs(elem(A, n, i, col));
            if (val > maxval) {
                maxval = val;
                maxrow = i;
            }
        }
        if (maxval == 0.0) {
            // singular
            return false;
        }

        // 2) Swap pivot array
        int tmpP = piv[col];
        piv[col] = piv[maxrow];
        piv[maxrow] = tmpP;

        // 3) Swap row col and maxrow of A for columns col..n-1
        if (maxrow != col) {
            for (int j = 0; j < n; j++) {
                double tmpA = elem(A, n, col, j);
                elem(A, n, col, j) = elem(A, n, maxrow, j);
                elem(A, n, maxrow, j) = tmpA;
            }
            // Also swap partial row of L for columns k..col-1
            for (int j = k; j < col; j++) {
                double tmpL = elem(L, n, col, j);
                elem(L, n, col, j) = elem(L, n, maxrow, j);
                elem(L, n, maxrow, j) = tmpL;
            }
        }

        // 4) U(col,col) = A(col,col)
        elem(U, n, col, col) = elem(A, n, col, col);

        // 5) For i=col+1..k+kb-1, fill in L(i,col), U(col,i)
        double pivotVal = elem(A, n, col, col);
        for (int i = col+1; i < k+kb; i++) {
            elem(L, n, i, col) = elem(A, n, i, col) / pivotVal;
            elem(U, n, col, i) = elem(A, n, col, i);
        }

        // 6) Update the sub-block A(col+1..k+kb-1, col+1..k+kb-1)
        for (int i = col+1; i < k+kb; i++) {
            double lval = elem(L, n, i, col);
            for (int j = col+1; j < k+kb; j++) {
                elem(A, n, i, j) -= lval * elem(U, n, col, j);
            }
        }
    }
    return true;
}

//---------------------------------------------------------------------
// Blocked LU Factorization with partial pivoting (right-looking).
// Outer loop increments k by block_size, factor a small block, then
// apply the factor to the trailing submatrix with a matrix-multiply
// approach.  This approach is standard in HPC libraries (like LAPACK).
//---------------------------------------------------------------------
bool blocked_lu_factor(double *A, double *L, double *U, int *piv,
                       int n, int block_size)
{
    for (int k = 0; k < n; k += block_size) {
        int kb = std::min(block_size, n - k);

        // 1) Factor the block A[k..n-1, k..k+kb-1] using partial pivoting
        //    but store the factor into A, L, U for rows/cols k..k+kb-1
        if (!factor_block(A, L, U, piv, n, k, kb)) {
            return false;  // singular
        }

        // 2) The block [k..k+kb-1, k..k+kb-1] is now factored.  We need
        //    to update the trailing columns k+kb..n-1 in the panel A
        //    (the "Triangular Solve" step):
        //    L(k..k+kb-1, k..k+kb-1) is known, so for each row in [k..n-1]
        //    and col in [k+kb..n-1], we do:
        //        A(i,j) -= L(i,k..k+kb-1)*U(k..k+kb-1,j)
        //    but we first do the "panel solve" for rows [k+kb..n-1].
        //    i.e., for i in [k+kb..n-1], col in [k..k+kb-1]:
        for (int col = k; col < k+kb; col++) {
            double pivotVal = elem(U, n, col, col); // the diagonal
            // For i=k+kb..n-1, fill L(i,col)
            #pragma omp parallel for schedule(static)
            for (int i = k+kb; i < n; i++) {
                elem(L, n, i, col) = elem(A, n, i, col) / pivotVal;
            }
            // For i=k+kb..n-1, also set U(col,i) = A(col,i)
            #pragma omp parallel for schedule(static)
            for (int j = k+kb; j < n; j++) {
                elem(U, n, col, j) = elem(A, n, col, j);
            }
        }

        // 3) Now do the trailing submatrix update:  A(i,j) -= L(i,k..k+kb-1)*U(k..k+kb-1,j)
        //    for i in [k+kb..n-1], j in [k+kb..n-1].
        //    This is basically a matrix multiply of the form:
        //        A(i,j) -= (L block) * (U block).
        //    We can tile this loop again for better cache reuse.
        #pragma omp parallel for schedule(static)
        for (int i = k+kb; i < n; i++) {
            for (int j = k+kb; j < n; j++) {
                double sumVal = 0.0;
                for (int x = k; x < k+kb; x++) {
                    sumVal += elem(L, n, i, x) * elem(U, n, x, j);
                }
                elem(A, n, i, j) -= sumVal;
            }
        }
    }
    return true;
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>\n";
        return 1;
    }
    int n = std::atoi(argv[1]);
    int nthreads = std::atoi(argv[2]);

    if (n <= 0 || nthreads <= 0) {
        std::cerr << "Error: invalid n or nthreads.\n";
        return 1;
    }

    omp_set_num_threads(nthreads);

    std::cout << "Blocked LU Factorization (partial pivot) on " 
              << n << " x " << n 
              << " matrix with " << nthreads << " threads.\n";

    // 1) Generate random matrix
    double *A0 = generate_random_matrix(n, nthreads);

    // 2) Allocate A, L, U
    double *A = (double*) numa_alloc_local((size_t)n * n * sizeof(double));
    double *L = (double*) calloc((size_t)n * n, sizeof(double));
    double *U = (double*) calloc((size_t)n * n, sizeof(double));
    if (!A || !L || !U) {
        std::cerr << "Error: allocation failed.\n";
        return 1;
    }
    std::memcpy(A, A0, (size_t)n*n*sizeof(double));

    // Initialize L’s diagonal
    for (int i = 0; i < n; i++) {
        elem(L, n, i, i) = 1.0;
    }

    // 3) Allocate pivot array
    int *piv = (int*) malloc(n * sizeof(int));
    if (!piv) {
        std::cerr << "Error: pivot allocation failed.\n";
        return 1;
    }
    for (int i = 0; i < n; i++) {
        piv[i] = i;
    }

    // 4) Perform blocked LU factorization
    //    Tune block_size for your architecture
    const int block_size = 64;

    double t1 = omp_get_wtime();
    bool ok = blocked_lu_factor(A, L, U, piv, n, block_size);
    double t2 = omp_get_wtime();

    if (!ok) {
        std::cerr << "ERROR: matrix is singular (pivot=0)\n";
    }

    // 5) Compute residual
    double l21_norm = 0.0;
    if (ok) {
        l21_norm = compute_residual_l21_norm(A0, n, piv, L, U);
    }

    // 6) Print results
    std::cout << "LU factorization time: " << (t2 - t1) << " seconds.\n";
    if (ok) {
        std::cout << "Residual L2,1 norm = " << l21_norm << "\n";
    }

    // 7) Cleanup
    numa_free(A0, (size_t)n*n*sizeof(double));
    numa_free(A,  (size_t)n*n*sizeof(double));
    free(L);
    free(U);
    free(piv);

    return 0;
}
