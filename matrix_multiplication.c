/* Parallelized Matrix Multiplication using OpenMP
 * Inputs: Number of threads, Number of loops to parallelize
 * Outputs: Execution time and result array
*/

#include<stdio.h>
#include<omp.h>
#include<stdlib.h>

#define DIM 1000 /* Size of matrix */

/* Global variables */
int num_threads;
__uint16_t A[DIM][DIM];
__uint16_t B[DIM][DIM];
__uint32_t C[DIM][DIM];

/* Initialize matricies */
void initialize_matricies() {
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            A[i][j] = rand();
            B[i][j] = rand();
            C[i][j] = 0;
        }
    }
}

// Function to perform matrix multiplication with outer loop parallelized
void matmul_outer_parallel(__uint16_t A[DIM][DIM], __uint16_t B[DIM][DIM], __uint32_t C[DIM][DIM], int num_threads) {
    int i, j, k;
    omp_set_num_threads(num_threads);

#pragma omp parallel for private(i, j, k) shared(A, B, C)
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            for (k = 0; k < DIM; k++) {
                if (k == 0) C[i][j] = 0;
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to perform matrix multiplication with outer and middle loops parallelized
void matmul_outer_middle_parallel(__uint16_t A[DIM][DIM], __uint16_t B[DIM][DIM], __uint32_t C[DIM][DIM], int num_threads) {
    int i, j, k;
    omp_set_num_threads(num_threads);

#pragma omp parallel for collapse(2) private(i, j, k) shared(A, B, C)
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            for (k = 0; k < DIM; k++) {
                if (k == 0) C[i][j] = 0;
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to perform matrix multiplication with all loops parallelized
void matmul_outer_middle_inner_parallel(__uint16_t A[DIM][DIM], __uint16_t B[DIM][DIM], __uint32_t C[DIM][DIM], int num_threads) {
    int i, j, k;
    omp_set_num_threads(num_threads);

#pragma omp parallel for collapse(3) private(i, j, k) shared(A, B, C)
    for (i = 0; i < DIM; i++) {
        for (j = 0; j < DIM; j++) {
            for (k = 0; k < DIM; k++) {
                if (k == 0) C[i][j] = 0;
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <num_threads> <loops_to_parallelize (1, 2, or 3)>\n", argv[0]);
        return -1;
    }

    int num_threads = atoi(argv[1]);
    int loops_to_parallelize = atoi(argv[2]);

    if (loops_to_parallelize < 1 || loops_to_parallelize > 3) {
        printf("Error: loops_to_parallelize should be 1, 2, or 3.\n");
        return -1;
    }

    // Initialize A, B and C matricies
    initialize_matricies();

    double start_time, end_time;

    // Start timing here
    start_time = omp_get_wtime();

    // Call the appropriate function based on loops_to_parallelize
    if (loops_to_parallelize == 1) {
        matmul_outer_parallel(A, B, C, num_threads);
    } else if (loops_to_parallelize == 2) {
        matmul_outer_middle_parallel(A, B, C, num_threads);
    } else {
        matmul_outer_middle_inner_parallel(A, B, C, num_threads);
    }

    // End timing here
    end_time = omp_get_wtime();
    
    // Print execution time
    printf("Execution time with %d threads and %d loops parallelized: %f seconds\n", num_threads, loops_to_parallelize, end_time - start_time);
}
