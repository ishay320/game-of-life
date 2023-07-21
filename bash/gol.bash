#!/usr/bin/env bash

states=(0 1)
width=30
height=20
board_a=()
board_b=()

clear_screen() {
    echo -e "\e[1;1H\e[2J"
}

print_board() {
    board=("$@")
    for ((i = 0; i < height; i++)); do
        for ((j = 0; j < width; j++)); do
            case "${board[i * width + j]}" in
            "${states[1]}")
                echo -n "# "
                ;;
            "${states[0]}")
                echo -n "  "
                ;;
            *)
                echo "ERROR: state not known"
                ;;
            esac
        done
        echo
    done
}

circular_arr() {
    declare -i row=$1
    declare -i column=$2
    declare -i height=$3
    declare -i width=$4
    echo $(($(($((row + height)) % height)) * width + $(($((column + width)) % width))))
}

for ((i = 0; i < height * width; i++)); do
    board_a[i]=${states[0]}
    board_b[i]=${states[0]}
done

# R-Pentomino
board_a[((height / 2) + 0) * width + ((width / 2) + 1)]=${states[1]}
board_a[((height / 2) + 0) * width + ((width / 2) + 0)]=${states[1]}
board_a[((height / 2) + 1) * width + ((width / 2) + 0)]=${states[1]}
board_a[((height / 2) + 2) * width + ((width / 2) + 0)]=${states[1]}
board_a[((height / 2) + 1) * width + ((width / 2) - 1)]=${states[1]}

while true; do
    for ((i = 0; i < height; i++)); do
        for ((j = 0; j < width; j++)); do

            # neighbors
            count=0
            for ((row = i - 1; row < i + 2; row++)); do
                for ((column = j - 1; column < j + 2; column++)); do
                    if ! { [[ $row == "$i" ]] && [[ $column == "$j" ]]; } &&
                        [[ ${board_a[$(circular_arr $row $column $height $width)]} -eq ${states[1]} ]]; then
                        ((count++))
                    fi
                done
            done
            if [[ ${board_a[$i * $width + $j]} == "${states[1]}" ]] && { [[ $count == 2 ]] || [[ $count == 3 ]]; }; then
                board_b[i * width + j]=${states[1]}
            elif [[ $count == 3 ]]; then
                board_b[i * width + j]=${states[1]}
            else
                board_b[i * width + j]=${states[0]}
            fi
        done
    done

    tmp=("${board_a[@]}")
    board_a=("${board_b[@]}")
    board_b=("${tmp[@]}")

    clear_screen
    print_board "${board_a[@]}"
done
