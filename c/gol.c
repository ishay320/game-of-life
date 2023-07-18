#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef USE_X11
#define XWRAP_IMPLEMENTATION
#define XWRAP_AUTO_LINK
#include "xwrap.h"
#endif

#define BOARD_WIDTH 300
#define BOARD_HEIGHT 200

bool should_run = true;
void signal_handler(int) { should_run = false; }

typedef enum _States
{
    DEAD,
    DYING,
    ALIVE,
    STATES_LEN
} States;

typedef struct _Board
{
    size_t width;
    size_t height;
    States* data;
} Board;

#define POS(arr, i, j, width) arr[(i)*width + (j)]

#ifdef USE_X11
xw_handle* handle = NULL;
void print_board_x11(Board board)
{
    size_t mult = 2;
    if (NULL == handle)
    {
        handle = xw_create_window(board.width * mult, board.height * mult);
    }
    for (size_t i = 0; i < board.height; i++)
    {
        for (size_t j = 0; j < board.width; j++)
        {
            uint32_t color = 0;
            switch (POS(board.data, i, j, board.width))
            {
                case DEAD:
                    color = 0x00000000;
                    break;
                case DYING:
                    color = 0x00FF00FF;
                    break;
                case ALIVE:
                    color = 0x00FFFFFF;
                    break;

                default:
                    printf("\n%s:%d %s ERROR: unreachable code, state: %d\n", __FILE__, __LINE__, __FUNCTION__, board.data[i * board.width + j]);
                    exit(1);
                    break;
            }
            xw_draw_rectangle(handle, j * mult, i * mult, mult, mult, true, color);
        }
    }

    xw_draw(handle);
}
#endif

void print_board(Board board)
{
    for (size_t i = 0; i < board.height; i++)
    {
        for (size_t j = 0; j < board.width; j++)
        {
            switch (POS(board.data, i, j, board.width))
            {
                case DEAD:
                    putchar(' ');
                    break;
                case DYING:
                    putchar('*');
                    break;
                case ALIVE:
                    putchar('@');
                    break;

                default:
                    printf("\n%s:%d %s ERROR: unreachable code, state: %d\n", __FILE__, __LINE__, __FUNCTION__, board.data[i * board.width + j]);
                    exit(1);
                    break;
            }
            putchar(' ');
        }
        putchar('\n');
    }
}

#define ROUND_POS(arr, i, j, width, height) POS(arr, ((i + height) % height), ((j + width) % width), width)
uint8_t get_neighbors(const Board board, size_t i, size_t j)
{
    uint8_t count = 0;
    for (int row = i - 1; row <= (int)i + 1; row++)
    {
        for (int column = j - 1; column <= (int)j + 1; column++)
        {
            if (!(row == (int)i && column == (int)j) && ROUND_POS(board.data, row, column, board.width, board.height) == ALIVE)
            {
                count++;
            }
        }
    }
    return count;
}

void step_bb(const Board board_a, Board board_b)
{
    for (size_t i = 0; i < board_a.height; i++)
    {
        for (size_t j = 0; j < board_a.width; j++)
        {
            uint8_t neighbors = get_neighbors(board_a, i, j);
            if (POS(board_a.data, i, j, board_a.width) == ALIVE && (2 == neighbors || neighbors == 3))
            {
                POS(board_b.data, i, j, board_b.width) = ALIVE;
            }
            else if (POS(board_a.data, i, j, board_a.width) == DEAD && neighbors == 2)
            {
                POS(board_b.data, i, j, board_b.width) = ALIVE;
            }
            else if (POS(board_a.data, i, j, board_a.width) == ALIVE)
            {
                POS(board_b.data, i, j, board_b.width) = DYING;
            }
            else
            {
                POS(board_b.data, i, j, board_b.width) = DEAD;
            }
        }
    }
}

void step_gol(const Board board_a, Board board_b)
{
    for (size_t i = 0; i < board_a.height; i++)
    {
        for (size_t j = 0; j < board_a.width; j++)
        {
            uint8_t neighbors = get_neighbors(board_a, i, j);
            if (POS(board_a.data, i, j, board_a.width) == ALIVE && (2 == neighbors || neighbors == 3))
            {
                POS(board_b.data, i, j, board_b.width) = ALIVE;
            }
            else if (neighbors == 3)
            {
                POS(board_b.data, i, j, board_b.width) = ALIVE;
            }
            else
            {
                POS(board_b.data, i, j, board_b.width) = DEAD;
            }
        }
    }
}

void clear_screen()
{
    const char* CLEAR_SCREEN_ANSI = "\e[1;1H\e[2J";
    write(STDOUT_FILENO, CLEAR_SCREEN_ANSI, 11);
}

void switch_board(Board* board_a, Board* board_b)
{
    Board tmp = *board_b;
    *board_b  = *board_a;
    *board_a  = tmp;
}

Board board_create(size_t width, size_t height)
{
    Board board = {.width = width, .height = height, .data = malloc(sizeof(States) * height * width)};
    return board;
}

void board_destroy(Board board) { free(board.data); }

int main(void)
{
    // Signal for ending the loop peacefully
    if (signal(SIGINT, signal_handler) == SIG_ERR)
    {
        perror("signal");
        return 1;
    }

    Board board_a = board_create(BOARD_WIDTH, BOARD_HEIGHT);
    Board board_b = board_create(BOARD_WIDTH, BOARD_HEIGHT);

    for (size_t i = 0; i < board_a.height * board_a.width; i++)
    {
        board_a.data[i] = DEAD;
    }

    {  // R-Pentomino
        POS(board_a.data, (board_a.height / 2) + 0, (board_a.width / 2) + 1, board_a.width) = ALIVE;
        POS(board_a.data, (board_a.height / 2) + 0, (board_a.width / 2) + 0, board_a.width) = ALIVE;
        POS(board_a.data, (board_a.height / 2) + 1, (board_a.width / 2) + 0, board_a.width) = ALIVE;
        POS(board_a.data, (board_a.height / 2) + 2, (board_a.width / 2) + 0, board_a.width) = ALIVE;
        POS(board_a.data, (board_a.height / 2) + 1, (board_a.width / 2) - 1, board_a.width) = ALIVE;
    }

    while (should_run)
    {
#ifdef BRIAN_BRAIN
        step_bb(board_a, board_b);
#else
        step_gol(board_a, board_b);
#endif
        switch_board(&board_a, &board_b);
#ifdef USE_X11
        print_board_x11(board_a);
#else
        clear_screen();
        print_board(board_a);
#endif

        usleep(100000);
    }

#ifdef USE_X11
    xw_free_window(handle);
#endif

    board_destroy(board_a);
    board_destroy(board_b);
    return 0;
}
