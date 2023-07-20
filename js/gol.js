
var states = [' ', '#'];
var width = 30;
var height = 20;


const sleep = ms => new Promise(r => setTimeout(r, ms));

function print_board(board) {
    board.forEach(row => {
        row.forEach(s => {
            process.stdout.write(' ' + s)
        });
        console.log();
    });
}

function neighbors(board, x, y) {
    count = 0;
    for (let i = x - 1; i < x + 2; i++) {
        for (let j = y - 1; j < y + 2; j++) {
            if (j == y && i == x) {
                continue;
            }
            if (board[(i + board.length) % board.length][(j + board[0].length) % board[0].length] == states[1]) {
                count++;
            }
        }
    }
    return count;
}

function clear_screen() {
    console.log("\x1b[1;1H\x1b[2J");
}

function step(board_a, board_b) {
    for (let i = 0; i < board_a.length; i++) {
        for (let j = 0; j < board_a[0].length; j++) {
            n = neighbors(board_a, i, j);

            if (board_a[i][j] == states[1] && (n == 2 || 3 ==
                n)) {
                board_b[i][j] = states[1];
            }
            else if (n == 3) {
                board_b[i][j] = states[1];
            }
            else {
                board_b[i][j] = states[0];
            }

        }
    }
}

function create_boards(width, height) {
    board_a = Array(height);
    board_b = Array(height).fill(Array(width).fill(states[0]));
    for (let i = 0; i < height; i++) {
        board_a[i] = Array(width);
        board_b[i] = Array(width);
        for (let j = 0; j < width; j++) {
            board_a[i][j] = states[0];
            board_b[i][j] = states[0];
        }
    }
    return [board_a, board_b]
}

function add_r_pentomino(board) {
    height = board.length
    width = board[0].length
    board[(height / 2) + 0][(width / 2) + 1] = states[1];
    board[(height / 2) + 0][(width / 2) + 0] = states[1];
    board[(height / 2) + 1][(width / 2) + 0] = states[1];
    board[(height / 2) + 2][(width / 2) + 0] = states[1];
    board[(height / 2) + 1][(width / 2) - 1] = states[1];

}

async function main() {
    [board_a, board_b] = create_boards(width, height)
    add_r_pentomino(board_a)

    while (true) {
        step(board_a, board_b)
        var tmp = board_a;
        board_a = board_b;
        board_b = tmp;

        clear_screen()
        print_board(board_a);
        await sleep(100);
    }
}


if (require.main === module) {
    main();
}