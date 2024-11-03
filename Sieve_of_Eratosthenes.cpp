
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>

// Global prime array and mutex for thread safety
std::vector<bool> prime_array;
std::mutex mtx;

void mark_non_primes(int start, int end, const std::vector<int>& seeds) {
    for (int p : seeds) {
        // Find the first multiple of p in this range that is >= start
        int first_multiple = std::max(p * p, ((start + p - 1) / p) * p);

        // Mark multiples of p in the range [start, end] as non-prime
        for (int j = first_multiple; j <= end; j += p) {
            prime_array[j] = false; // Mark as not prime
        }
    }
}

// Sequentially compute primes up to sqrt(Max)
std::vector<int> compute_primes_up_to_sqrt(int sqrt_max) {
    std::vector<int> seeds;
    for (int i = 2; i <= sqrt_max; ++i) {
        if (prime_array[i]) {
            seeds.push_back(i); // i is prime
            for (int j = i * i; j <= sqrt_max; j += i) {
                prime_array[j] = false; // Mark multiples of i as not prime
            }
        }
    }
    return seeds;
}

int main() {
    const int MAX = 10000000; // Maximum number to check for primes
    const int NUM_THREADS = 4; // Number of threads
    int sqrt_max = std::sqrt(MAX);

    // Initialize the prime array (true means prime)
    prime_array.assign(MAX + 1, true);
    prime_array[0] = prime_array[1] = false; // 0 and 1 are not prime

    // Step 1: Sequentially compute primes up to sqrt(Max)
    std::vector<int> seeds = compute_primes_up_to_sqrt(sqrt_max);

    // Step 2: Divide the range from sqrt(Max)+1 to Max among threads
    std::vector<std::thread> threads;
    int chunk_size = (MAX - sqrt_max) / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = sqrt_max + 1 + i * chunk_size;
        int end = (i == NUM_THREADS - 1) ? MAX : start + chunk_size - 1;
        threads.emplace_back(mark_non_primes, start, end, std::cref(seeds));
    }

    // Step 3: Wait for all threads to complete
    for (auto& th : threads) {
        th.join();
    }

    // Step 4: Collect and print the unmarked numbers (which are primes)
    std::cout << "Primes up to " << MAX << " are:\n";
    for (int i = 2; i <= MAX; ++i) {
        if (prime_array[i]) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;

    return 0;
}
