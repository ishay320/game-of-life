from time import sleep

width = 30
height = 25
states = [' ', '#']

def get_neighbors(board: list[list[str]], y: int, x: int) -> int:
    count: int  = 0
    width: int  = len(board[0])
    height: int = len(board)

    for i in range(y-1, y+2):
        for j in range(x-1, x+2):
            if board[(i + height )% height][(j + width) % width] == states[1] and not (i == y and j == x):
                count += 1
    return count

def print_board(board: list[list[str]]) -> None:
    for row in board:
        for col in row:
            print(col,end=' ') 
        print()

def clear_screen() -> None:
    print("\033[1;1H\033[2J")


def step(board_a: list[list[str]], board_b: list[list[str]]) -> None:
    for i in range(height):
        for j in range(width):
            neighbors = get_neighbors(board_a,i,j)
            if board_a[i][j] == states[1] and (neighbors == 2 or 3 ==
                neighbors):
                board_b[i][j] = states[1]
            elif neighbors == 3:
                board_b[i][j] = states[1]
            else:
                board_b[i][j] = states[0]

if __name__ == "__main__":
    board_a: list[list[str]] = [[states[0] for i in range(width)] for j in
        range(height)]
    board_b: list[list[str]] = [[states[0] for i in range(width)] for j in
        range(height)]

    # R-Pentomino
    board_a[(height//2) + 0][(width // 2) + 1] = states[1]
    board_a[(height//2) + 0][(width // 2) + 0] = states[1]
    board_a[(height//2) + 1][(width // 2) + 0] = states[1]
    board_a[(height//2) + 2][(width // 2) + 0] = states[1]
    board_a[(height//2) + 1][(width // 2) - 1] = states[1]

    while(True):
        step(board_a, board_b)
        board_a, board_b = board_b, board_a

        clear_screen()
        print_board(board_a)

        sleep(0.1)


