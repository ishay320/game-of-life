#include <cuda.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

#define BOARD_WIDTH 30
#define BOARD_HEIGHT 20

bool should_run = true;
void signal_handler(int) { should_run = false; }

typedef enum _States
{
    DEAD,
    ALIVE,
    STATES_LEN
} States;

#define POS(arr, i, j, width) arr[(i)*width + (j)]
#define cudaCheckErrors(msg)                                                                                        \
    do                                                                                                              \
    {                                                                                                               \
        cudaError_t __err = cudaGetLastError();                                                                     \
        if (__err != cudaSuccess)                                                                                   \
        {                                                                                                           \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n");                                                             \
            exit(1);                                                                                                \
        }                                                                                                           \
    } while (0)

void print_board(States *board, size_t width, size_t height)
{
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            switch (POS(board, i, j, width))
            {
                case 0:
                    putchar(' ');
                    break;
                case 1:
                    putchar('#');
                    break;

                default:
                    printf("\n%s:%d %s ERROR: unreachable code, state: %d\n", __FILE__, __LINE__, __FUNCTION__, board[i * width + j]);
                    exit(1);
                    break;
            }
            putchar(' ');
        }
        putchar('\n');
    }
}

#define ROUND_POS(arr, i, j, width, height) POS(arr, ((i + height) % height), ((j + width) % width), width)
__device__ uint8_t get_neighbors(States *board, size_t width, size_t height, size_t i, size_t j)
{
    uint8_t count = 0;

    for (int row = i - 1; row <= (int)i + 1; row++)
    {
        for (int column = j - 1; column <= (int)j + 1; column++)
        {
            if (!(row == (int)i && column == (int)j) && ROUND_POS(board, row, column, width, height) == 1)
            {
                count++;
            }
        }
    }
    return count;
}

__global__ void step(States *board, States *board_out, size_t width, size_t height)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (width * height))
    {
        printf("%d, idx %d", width * height, idx);
        return;
    }

    int j = idx % width;
    int i = (idx - j) / width;

    uint8_t neighbors = get_neighbors(board, width, height, i, j);

    if (POS(board, i, j, width) == ALIVE && (2 == neighbors || neighbors == 3))
    {
        POS(board_out, i, j, width) = ALIVE;
    }
    else if (neighbors == 3)
    {
        POS(board_out, i, j, width) = ALIVE;
    }
    else
    {
        POS(board_out, i, j, width) = DEAD;
    }
}

void clear_screen()
{
    const char *CLEAR_SCREEN_ANSI = "\e[1;1H\e[2J";
    write(STDOUT_FILENO, CLEAR_SCREEN_ANSI, 11);
}

void switch_board(States **board_a, States **board_b)
{
    States *tmp = *board_b;
    *board_b    = *board_a;
    *board_a    = tmp;
}

int main(void)
{
    // Signal for ending the loop peacefully
    if (signal(SIGINT, signal_handler) == SIG_ERR)
    {
        perror("signal");
        return 1;
    }

    size_t size       = BOARD_WIDTH * BOARD_HEIGHT;
    States *board_cpu = (States *)malloc(size * sizeof(States));

    States *board_gpu;
    cudaMalloc((void **)&board_gpu, size * sizeof(States));
    cudaCheckErrors("cudaMalloc fail");
    States *board_gpu_out;
    cudaMalloc((void **)&board_gpu_out, size * sizeof(States));
    cudaCheckErrors("cudaMalloc fail");

    // Initialize host array
    for (size_t i = 0; i < BOARD_WIDTH * BOARD_HEIGHT; i++)
    {
        board_cpu[i] = DEAD;
    }

    {  // R-Pentomino
        POS(board_cpu, (BOARD_HEIGHT / 2) + 0, (BOARD_WIDTH / 2) + 1, BOARD_WIDTH) = ALIVE;
        POS(board_cpu, (BOARD_HEIGHT / 2) + 0, (BOARD_WIDTH / 2) + 0, BOARD_WIDTH) = ALIVE;
        POS(board_cpu, (BOARD_HEIGHT / 2) + 1, (BOARD_WIDTH / 2) + 0, BOARD_WIDTH) = ALIVE;
        POS(board_cpu, (BOARD_HEIGHT / 2) + 2, (BOARD_WIDTH / 2) + 0, BOARD_WIDTH) = ALIVE;
        POS(board_cpu, (BOARD_HEIGHT / 2) + 1, (BOARD_WIDTH / 2) - 1, BOARD_WIDTH) = ALIVE;
    }

    // Copy to CUDA device
    cudaMemcpy(board_gpu, board_cpu, size * sizeof(States), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy 1 fail");

    // Size of blocks and threads
    int block_size = 1;
    int n_blocks   = size / block_size + (size % block_size == 0 ? 0 : 1);
    step<<<n_blocks, block_size>>>(board_gpu, board_gpu_out, BOARD_WIDTH, BOARD_HEIGHT);
    cudaDeviceSynchronize();
    cudaCheckErrors("kernel fail");

    while (should_run)
    {
        // Retrieve result from device and store it in host array
        cudaDeviceSynchronize();
        cudaMemcpy(board_cpu, board_gpu_out, size * sizeof(States), cudaMemcpyDeviceToHost);
        cudaCheckErrors("cudaMemcpy 2 fail");

        step<<<n_blocks, block_size>>>(board_gpu, board_gpu_out, BOARD_WIDTH, BOARD_HEIGHT);
        cudaCheckErrors("kernel fail");

        // Swap boards
        States *tmp   = board_gpu_out;
        board_gpu_out = board_gpu;
        board_gpu     = tmp;

        // Print results
        clear_screen();
        print_board(board_cpu, BOARD_WIDTH, BOARD_HEIGHT);

        usleep(100000);
    }

    // Cleanup
    free(board_cpu);
    cudaFree(board_gpu);
    cudaFree(board_gpu_out);
}