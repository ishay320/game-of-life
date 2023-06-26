

SRC=main.c
CFLAGS=-Wall -Wextra -g

main: ${SRC}
	${CC} ${CFLAGS} $< -o $@