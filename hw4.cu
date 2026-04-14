/*
 Name: Chanakya Thirumala Setty
 Email: cthirumalasetty@crimson.ua.edu
 Course Section: CS 481
 Homework #: 4
 To Compile: nvcc -O3 -o hw4 hw4.cu
 To Run: ./hw4 <board_size> <max_iterations> <output_file>
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define BLOCK_SIZE 16

/* Get wall clock time */
double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
}

/*
 * CUDA kernel for one generation of Game of Life using shared memory.
 * Each thread block loads a (BLOCK_SIZE+2)x(BLOCK_SIZE+2) tile into shared
 * memory (including a 1-cell halo on every side) to reduce global memory
 * traffic from the 8-neighbor stencil.
 *
 * current : input flat array of size (N+2)*(N+2) — ghost cells at index 0
 *           and N+1 in both dimensions are always 0 and never written.
 * next    : output flat array of the same size
 * N       : board side length; active cells occupy indices [1..N] x [1..N]
 */
__global__ void game_of_life_kernel(const int *current, int *next, int N) {
    /* Shared memory tile: center + 1-cell halo on each side */
    __shared__ int s[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /* Global position in the (N+2)x(N+2) array */
    int row = blockIdx.y * BLOCK_SIZE + ty + 1;
    int col = blockIdx.x * BLOCK_SIZE + tx + 1;

    int stride = N + 2;

    /* ---- Load shared memory tile ---- */

    /* Center cell */
    s[ty + 1][tx + 1] = (row <= N && col <= N) ? current[row * stride + col] : 0;

    /* Top halo — only first thread row; row-1 is always >= 0 */
    if (ty == 0)
        s[0][tx + 1] = (col <= N) ? current[(row - 1) * stride + col] : 0;

    /* Bottom halo — only last thread row */
    if (ty == BLOCK_SIZE - 1) {
        int r = row + 1;
        s[BLOCK_SIZE + 1][tx + 1] = (r <= N + 1 && col <= N) ? current[r * stride + col] : 0;
    }

    /* Left halo — only first thread column; col-1 is always >= 0 */
    if (tx == 0)
        s[ty + 1][0] = (row <= N) ? current[row * stride + (col - 1)] : 0;

    /* Right halo — only last thread column */
    if (tx == BLOCK_SIZE - 1) {
        int c = col + 1;
        s[ty + 1][BLOCK_SIZE + 1] = (c <= N + 1 && row <= N) ? current[row * stride + c] : 0;
    }

    /* Four corners — one thread each */
    if (tx == 0 && ty == 0)
        s[0][0] = current[(row - 1) * stride + (col - 1)];

    if (tx == BLOCK_SIZE - 1 && ty == 0) {
        int c = col + 1;
        s[0][BLOCK_SIZE + 1] = (c <= N + 1) ? current[(row - 1) * stride + c] : 0;
    }

    if (tx == 0 && ty == BLOCK_SIZE - 1) {
        int r = row + 1;
        s[BLOCK_SIZE + 1][0] = (r <= N + 1) ? current[r * stride + (col - 1)] : 0;
    }

    if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1) {
        int r = row + 1, c = col + 1;
        s[BLOCK_SIZE + 1][BLOCK_SIZE + 1] =
            (r <= N + 1 && c <= N + 1) ? current[r * stride + c] : 0;
    }

    __syncthreads();

    /* ---- Compute next generation for this cell ---- */
    if (row <= N && col <= N) {
        int n_count = s[ty][tx]       + s[ty][tx + 1]       + s[ty][tx + 2]
                    + s[ty + 1][tx]                          + s[ty + 1][tx + 2]
                    + s[ty + 2][tx]   + s[ty + 2][tx + 1]   + s[ty + 2][tx + 2];
        int alive = s[ty + 1][tx + 1];
        /* Standard GoL rules: born at 3 neighbors, survives at 2 or 3 */
        next[row * stride + col] = (n_count == 3) || (alive && n_count == 2);
    }
}

int main(int argc, char **argv) {
    int N, K;
    char *output_file;
    double starttime, endtime;

    if (argc < 4) {
        printf("Usage: %s <board_size> <max_iterations> <output_file>\n", argv[0]);
        exit(-1);
    }

    N = atoi(argv[1]);
    K = atoi(argv[2]);
    output_file = argv[3];

    int stride      = N + 2;
    int total_cells = stride * stride;

    /* Allocate host array (flat 1-D representation of (N+2)x(N+2) grid) */
    int *h_current = (int *)malloc(total_cells * sizeof(int));
    if (h_current == NULL) {
        printf("Error allocating host memory\n");
        exit(-1);
    }

    /* Zero out everything so ghost cells stay 0 */
    memset(h_current, 0, total_cells * sizeof(int));

    /* Initialize active cells with random values using the default seed,
       matching the sequential version for output comparison */
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            h_current[i * stride + j] = rand() % 2;

    /* Allocate two device arrays: one for each generation */
    int *d_current, *d_next;
    cudaMalloc((void **)&d_current, total_cells * sizeof(int));
    cudaMalloc((void **)&d_next,    total_cells * sizeof(int));

    /* Copy initial board to device; d_next ghost cells must be 0 */
    cudaMemcpy(d_current, h_current, total_cells * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_next, 0, total_cells * sizeof(int));

    /* One thread per active cell; grid rounds up to cover all N x N cells */
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Ensure all transfers are done before starting the timer */
    cudaDeviceSynchronize();
    starttime = gettime();

    /* Main simulation: run exactly K iterations (no early-exit check) */
    for (int k = 0; k < K; k++) {
        game_of_life_kernel<<<gridDim, blockDim>>>(d_current, d_next, N);
        /* Swap pointers so the output of this step becomes the input of the next */
        int *temp = d_current;
        d_current = d_next;
        d_next    = temp;
    }

    /* Wait for all GPU work to complete before stopping the timer */
    cudaDeviceSynchronize();
    endtime = gettime();

    printf("Time taken for size %d = %lf seconds\n", N, endtime - starttime);

    /* Copy final board back to host */
    cudaMemcpy(h_current, d_current, total_cells * sizeof(int), cudaMemcpyDeviceToHost);

    /* Write final board to the output file (active cells only, no ghost cells) */
    FILE *fp = fopen(output_file, "w");
    if (fp == NULL) {
        printf("Error opening output file: %s\n", output_file);
        exit(-1);
    }
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++)
            fprintf(fp, "%d ", h_current[i * stride + j]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    /* Free host and device memory */
    free(h_current);
    cudaFree(d_current);
    cudaFree(d_next);

    return 0;
}
