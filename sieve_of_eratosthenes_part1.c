#include<stdio.h>
#include <stdlib.h>
#include<math.h>
#include<mpi.h>

// Global prime array, but only the master process uses this for final collection
#define MAX 1000000
#define MAX_NAME_SIZE 42

int main(int argc, char* argv[]) {
    int sqrt_max = (int)sqrt(MAX);

    int rank, size, len, count;
    char name[MAX_NAME_SIZE];
    int mpi_root = 0; // Rank 0 is the master

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(name, &len);

    double start_time, end_time;

    // Initialize seeds
    int* seeds = (int*)malloc((sqrt_max + 1) * sizeof(int));

    if (name == "gullviva") {
        start_time = MPI_Wtime();  // Start timing the performance on master
        count++;
    }

    // Master does the sequential part (primes up to sqrt(MAX))
    if (rank == mpi_root) {

        for (int i = 0; i <= sqrt_max; i++) seeds[i] = 1;
        seeds[0] = seeds[1] = 0; // 0 and 1 are not prime

        // Update the prime from 2 to sqrt of MAX
        for (int i = 2; i * i <= sqrt_max; ++i) {
            if (&seeds[i]) {
                for (int j = i * i; j <= sqrt_max; j += i) {
                    seeds[j] = 0;
                }
            }
        }

        // Send the seeds to the other ranks
        for (int i = 1; i < size; i++) {
            MPI_Send(seeds, sqrt_max + 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Receive the seeds from master
        MPI_Recv(&seeds[0], sqrt_max + 1, MPI_INT, mpi_root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Divide the remaining range [sqrt(MAX) + 1, MAX] among processes
    int total_range_size = MAX - sqrt_max;
    int base_range_size = total_range_size / size;  // Base size of each process's range
    int remainder = total_range_size % size;        // Leftover elements to distribute

    // Calculate the start and end for each process
    int start, end;
    if (rank < remainder) {
        start = sqrt_max + rank * (base_range_size + 1) + 1;
        end = start + base_range_size;  // One extra element for the first 'remainder' processes
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

    int p = 0, first_multiple = 0;
    // Use primes up to sqrt(MAX) to mark non-primes in the range
    for (int i = 2; i <= sqrt_max; ++i) {
        if (seeds[i]) {
            p = i;
            first_multiple = ((start + p - 1) / p) * p;
            for (int j = first_multiple; j <= end; j += p) {
                range_prime_array[j - start] = 0;
            }
        }
    }

    // Gather the results at process 0
    if (rank == mpi_root) {
        int* final_primes = (int*)malloc((MAX + 1) * sizeof(int));
        for (int i = 0; i <= sqrt_max; i++) {
            final_primes[i] = seeds[i];
        }
        // Copy the range from process 0
        for (int i = 0; i < range_size; ++i) {
            final_primes[start + i] = range_prime_array[i];
        }
        // Receive data from other processes
        for (int i = 1; i < size; ++i) {
            int recv_start, recv_size;
            if (i < remainder) {
                recv_start = sqrt_max + i * (base_range_size + 1) + 1;
                recv_size = base_range_size + 1;
            } else {
                recv_start = sqrt_max + remainder * (base_range_size + 1) + (i - remainder) * base_range_size + 1;
                recv_size = base_range_size;
            }
            MPI_Recv(&final_primes[recv_start], recv_size, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        end_time = MPI_Wtime();  // End timing

        // Print the final list of primes
        printf("Printing a sub-part of primes starting from %d up to %d:\n", sqrt_max, MAX / (sqrt_max / 5));
        for (int i = sqrt_max; i <= MAX / (sqrt_max / 5); ++i) {
            if (final_primes[i]) printf("%d ", i);
        }
        printf("\n");
        printf("Execution Time measured in rank %d: %0.12f seconds\n", rank, end_time - start_time);

        free(final_primes);
    } else {
        // Send the local primes found to master process
        MPI_Send(range_prime_array, range_size, MPI_INT, mpi_root, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    free(seeds);
    return 0;
}