#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h> // Include math.h for fabs

#define SIZE 42000 /* Array size */
#define TOLERANCE 1e-6 // Define a tolerance for comparison

void row_oriented_back_substitution(double **A, double *b, double *x) {
    int row,col;

    /* Parallelisation starts here */
    #pragma omp parallel for schedule(runtime) default(shared) private(row,col)
        for(row = SIZE-1; row >= 0; row--) {
                x[row] = b[row];
                for(col = row+1; col < SIZE; col++) {
                        x[row] -= A[row][col] * x[col];
                }
                x[row] /= A[row][row];
        }
}

void column_oriented_back_substitution(double **A, double *b, double *x) {
    int row, col;

    /* Parallelisation starts here */
    #pragma omp parallel default(shared) private(row,col)
    {
        #pragma omp single  // Only one thread initializes x[row] = b[row];
        {
            for(row = 0; row < SIZE; row++) {
                    x[row] = b[row];
            }
        }
        
        // The outer loop (col) must remain sequential for correct back substitution
        for (col = SIZE - 1; col >= 0; col--) {
            #pragma omp single  // Ensure only one thread performs the division
            {
                x[col] /= A[col][col];
            }

            // Parallelize the inner loop (row)
            #pragma omp for schedule(runtime)
            for (row = 0; row < col; row++) {
                x[row] -= A[row][col] * x[col];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return -1;
    }

    int num_threads = atoi(argv[1]);
    int i, j;
    double start_time, end_time;

    /* Shared Variables */
    double **A = malloc(SIZE * sizeof(double *));  // Upper triangular matrix
    double *b = malloc(SIZE * sizeof(double));     // Right-hand side vector
    double *x_row = malloc(SIZE * sizeof(double)); // Solution vector for row-oriented back substitution
    double *x_col = malloc(SIZE * sizeof(double)); // Solution vector for column-oriented back substitution

    // Allocate each row of the matrix
    for (int i = 0; i < SIZE; ++i) {
        A[i] = malloc(SIZE * sizeof(double));
    }

    // Seed the random number generator with the current time
    srand(time(0));  
    /* Initializes the A and b with random numbers between 1 and 10*/
    for(i = 0; i < SIZE; i++) {
        for(j = 0; j < SIZE; j++) {
                A[i][j] = (i <= j) ? (rand() % 10) + 1 : 0; // Upper triangular matrix
        }
        b[i] = (rand() % 10) + 1;
    }

    omp_set_num_threads(num_threads);

    // Measure the execution time for row-oriented back substitution
    start_time = omp_get_wtime();
    row_oriented_back_substitution(A, b, x_row);
    end_time = omp_get_wtime();
    printf("Row-oriented back substitution execution time: %.15f seconds\n", (end_time - start_time));

    // Measure the execution time for column-oriented back substitution
    start_time = omp_get_wtime();
    column_oriented_back_substitution(A, b, x_col);
    end_time = omp_get_wtime();
    printf("Column-oriented back substitution execution time: %.15f seconds\n",(end_time - start_time));

    // Compare the results
    int mismatch = 0;
    for (int i = 0; i < SIZE; ++i) {
        if (fabs(x_row[i] - x_col[i]) >= TOLERANCE) {
            mismatch = 1;
            break;
        }
    }
    if (mismatch == 1) {
        printf("Mismatch found between the row oriented and column oriented solutions\n");
    }

    // Free allocated memory
    for (int i = 0; i < SIZE; ++i) {
        free(A[i]);
    }
    free(A);
    free(b);
    free(x_row);
    free(x_col);

    return 0;
}
