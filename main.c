#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BOARD_WIDTH 30
#define BOARD_HEIGHT 20

bool should_run = true;
void signal_handler(int) { should_run = false; }

enum states
{
    DEAD,
    ALIVE,
    STATES_LEN
};

#define POS(arr, i, j, width) arr[(i)*width + (j)]

void print_board(uint8_t* board, size_t width, size_t height)
{
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            switch (POS(board, i, j, width))
            {
                case DEAD:
                    printf(" ");
                    break;
                case ALIVE:
                    printf("*");
                    break;

                default:
                    printf("\n%s:%d %s ERROR: unreachable code, state: %d\n", __FILE__, __LINE__, __FUNCTION__, board[i * width + j]);
                    exit(1);
                    break;
            }
            printf(" ");
        }
        printf("\n");
    }
}

#define ROUND_POS(arr, i, j, width, height) POS(arr, ((i + height) % height), ((j + width) % width), width)
uint8_t get_neighbors(const uint8_t* board, size_t width, size_t height, size_t i, size_t j)
{
    uint8_t count = 0;
    for (int row = i - 1; row <= (int)i + 1; row++)
    {
        for (int column = j - 1; column <= (int)j + 1; column++)
        {
            if (!(row == (int)i && column == (int)j) && ROUND_POS(board, row, column, width, height) == ALIVE)
            {
                count++;
            }
        }
    }
    return count;
}

void step(const uint8_t* board_a, uint8_t* board_b, size_t width, size_t height)
{
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            uint8_t neighbors = get_neighbors(board_a, width, height, i, j);
            if (POS(board_a, i, j, width) == ALIVE && (2 == neighbors || neighbors == 3))
            {
                POS(board_b, i, j, width) = ALIVE;
            }
            else if (neighbors == 3)
            {
                POS(board_b, i, j, width) = ALIVE;
            }
            else
            {
                POS(board_b, i, j, width) = DEAD;
            }
        }
    }
}

void clear_screen()
{
    const char* CLEAR_SCREEN_ANSI = "\e[1;1H\e[2J";
    write(STDOUT_FILENO, CLEAR_SCREEN_ANSI, 11);
}

void switch_board(uint8_t** board_a, uint8_t** board_b)
{
    uint8_t* tmp = *board_b;
    *board_b     = *board_a;
    *board_a     = tmp;
}

int main(void)
{
    // Signal for ending the loop peacefully
    if (signal(SIGINT, signal_handler) == SIG_ERR)
    {
        perror("signal");
        return 1;
    }

    uint8_t* board_a = malloc(sizeof(uint8_t) * BOARD_HEIGHT * BOARD_WIDTH);
    uint8_t* board_b = malloc(sizeof(uint8_t) * BOARD_HEIGHT * BOARD_WIDTH);

    for (size_t i = 0; i < BOARD_HEIGHT * BOARD_WIDTH; i++)
    {
        board_a[i] = DEAD;
    }

    {  // Create a glider
        POS(board_a, 10, 10, BOARD_WIDTH) = ALIVE;
        POS(board_a, 10, 11, BOARD_WIDTH) = ALIVE;
        POS(board_a, 10, 12, BOARD_WIDTH) = ALIVE;
        POS(board_a, 11, 10, BOARD_WIDTH) = ALIVE;
        POS(board_a, 12, 11, BOARD_WIDTH) = ALIVE;
        POS(board_a, 1, 0, BOARD_WIDTH)   = ALIVE;
    }

    while (should_run)
    {
        step(board_a, board_b, BOARD_WIDTH, BOARD_HEIGHT);
        switch_board(&board_a, &board_b);

        clear_screen();
        print_board(board_a, BOARD_WIDTH, BOARD_HEIGHT);
        usleep(100000);
    }

    free(board_a);
    free(board_b);
    return 0;
}
