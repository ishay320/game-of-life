

SRC=gol.c
CFLAGS=-Wall -Wextra -g

gol: ${SRC}
	${CC} ${CFLAGS} $< -o $@