#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define USE_X11

#ifdef USE_X11
#define XWRAP_IMPLEMENTATION
#define XWRAP_AUTO_LINK
#include "xwrap.h"
#endif

#define BOARD_WIDTH 40
#define BOARD_HEIGHT 40

bool should_run = true;
void signal_handler(int) { should_run = false; }

typedef enum _States
{
    DEAD,
    ALIVE,
    STATES_LEN
} States;

#define POS(arr, i, j, width) arr[(i)*width + (j)]

#ifdef USE_X11
xw_handle* handle = NULL;
void print_board_x11(States* board, size_t width, size_t height)
{
    size_t mult = 10;
    if (NULL == handle)
    {
        handle = xw_create_window(height * mult, width * mult);
    }
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            uint32_t color = 0;
            switch (POS(board, i, j, width))
            {
                case DEAD:
                    color = 0x00000000;
                    break;
                case ALIVE:
                    color = 0x00FFFFFF;
                    break;

                default:
                    printf("\n%s:%d %s ERROR: unreachable code, state: %d\n", __FILE__, __LINE__, __FUNCTION__, board[i * width + j]);
                    exit(1);
                    break;
            }
            xw_draw_rectangle(handle, i * mult, j * mult, mult, mult, true, color);
        }
    }

    xw_draw(handle);
}
#endif

void print_board(States* board, size_t width, size_t height)
{
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            switch (POS(board, i, j, width))
            {
                case DEAD:
                    putchar(' ');
                    break;
                case ALIVE:
                    putchar('*');
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
uint8_t get_neighbors(const States* board, size_t width, size_t height, size_t i, size_t j)
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

void step(const States* board_a, States* board_b, size_t width, size_t height)
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

void switch_board(States** board_a, States** board_b)
{
    States* tmp = *board_b;
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

    States* board_a = malloc(sizeof(States) * BOARD_HEIGHT * BOARD_WIDTH);
    States* board_b = malloc(sizeof(States) * BOARD_HEIGHT * BOARD_WIDTH);

    for (size_t i = 0; i < BOARD_HEIGHT * BOARD_WIDTH; i++)
    {
        board_a[i] = DEAD;
    }

    {  // R-Pentomino
        POS(board_a, (BOARD_HEIGHT / 2) + 0, (BOARD_WIDTH / 2) + 1, BOARD_WIDTH) = ALIVE;
        POS(board_a, (BOARD_HEIGHT / 2) + 0, (BOARD_WIDTH / 2) + 0, BOARD_WIDTH) = ALIVE;
        POS(board_a, (BOARD_HEIGHT / 2) + 1, (BOARD_WIDTH / 2) + 0, BOARD_WIDTH) = ALIVE;
        POS(board_a, (BOARD_HEIGHT / 2) + 2, (BOARD_WIDTH / 2) + 0, BOARD_WIDTH) = ALIVE;
        POS(board_a, (BOARD_HEIGHT / 2) + 1, (BOARD_WIDTH / 2) - 1, BOARD_WIDTH) = ALIVE;
    }
    while (should_run)
    {
        step(board_a, board_b, BOARD_WIDTH, BOARD_HEIGHT);
        switch_board(&board_a, &board_b);
#ifdef USE_X11
        print_board_x11(board_a, BOARD_WIDTH, BOARD_HEIGHT);
#else
        clear_screen();
        print_board(board_a, BOARD_WIDTH, BOARD_HEIGHT);
#endif

        // clear_screen();
        usleep(100000);
    }

#ifdef USE_X11
    xw_free_window(handle);
#endif

    free(board_a);
    free(board_b);
    return 0;
}
