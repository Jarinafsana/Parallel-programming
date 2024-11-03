#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MAX 1000000
#define MAX_NAME_SIZE 42

int main(int argc, char* argv[]) {
    int sqrt_max = (int)sqrt(MAX);
    int rank, size, len, count = 0;
    char name[MAX_NAME_SIZE];
    int mpi_root = 0;  // Rank 0 is the master

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(name, &len);

    double start_time, end_time, local_elapsed, global_elapsed;

    // Initialize seeds
    int* seeds = (int*)malloc((sqrt_max + 1) * sizeof(int));

    if (rank == mpi_root) {
        // Master initializes the seeds
        for (int i = 0; i <= sqrt_max; i++) seeds[i] = 1;
        seeds[0] = seeds[1] = 0; // 0 and 1 are not prime

        // Sieve of Eratosthenes for primes up to sqrt(MAX)
        for (int i = 2; i * i <= sqrt_max; ++i) {
            if (seeds[i]) {
                for (int j = i * i; j <= sqrt_max; j += i) {
                    seeds[j] = 0;
                }
            }
        }
    }

    // Broadcast the seeds to all processes
    MPI_Bcast(seeds, sqrt_max + 1, MPI_INT, mpi_root, MPI_COMM_WORLD);

    // Start timing after receiving the seeds
    start_time = MPI_Wtime();

    // Divide the remaining range [sqrt(MAX) + 1, MAX] among processes
    int total_range_size = MAX - sqrt_max;
    int base_range_size = total_range_size / size;
    int remainder = total_range_size % size;

    int start, end;
    if (rank < remainder) {
        start = sqrt_max + rank * (base_range_size + 1) + 1;
        end = start + base_range_size;
    } else {
        start = sqrt_max + remainder * (base_range_size + 1) + (rank - remainder) * base_range_size + 1;
        end = start + base_range_size - 1;
    }

    // Mark non-primes in the assigned range
    int range_size = end - start + 1;
    int* range_prime_array = (int*)malloc(range_size * sizeof(int));
    for (int i = 0; i < range_size; i++) {
        range_prime_array[i] = 1;
    }

    // Use primes up to sqrt(MAX) to mark non-primes in the range
    for (int i = 2; i <= sqrt_max; ++i) {
        if (seeds[i]) {
            int p = i;
            int first_multiple = ((start + p - 1) / p) * p;
            if (first_multiple < start) first_multiple = p * p;
            for (int j = first_multiple; j <= end; j += p) {
                range_prime_array[j - start] = 0;
            }
        }
    }

    // Gather the results at process 0 using MPI_Gather
    int* final_primes = NULL;
    if (rank == mpi_root) {
        final_primes = (int*)malloc((MAX + 1) * sizeof(int));
        // Initialize the final primes array up to sqrt(MAX)
        for (int i = 0; i <= sqrt_max; i++) {
            final_primes[i] = seeds[i];
        }
    }

    // Gather the results from all processes
    MPI_Gather(range_prime_array, range_size, MPI_INT, &final_primes[start], range_size, MPI_INT, mpi_root, MPI_COMM_WORLD);

    // Stop the timer
    end_time = MPI_Wtime();
    local_elapsed = end_time - start_time;

    // Reduce the timing results to the master process
    MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, mpi_root, MPI_COMM_WORLD);

    if (rank == mpi_root) {
        // Print the final list of primes (a subset for demonstration)
        printf("Printing a sub-part of primes starting from %d up to %d:\n", sqrt_max, MAX / (sqrt_max / 5));
        for (int i = sqrt_max; i <= MAX / (sqrt_max / 5); ++i) {
            if (final_primes[i]) printf("%d ", i);
        }
        printf("\n");
        printf("Max Execution Time among all processes: %0.12f seconds\n", global_elapsed);

        free(final_primes);
    }

    // Finalize MPI and free allocated memory
    MPI_Finalize();
    free(seeds);
    free(range_prime_array);

    return 0;
}
