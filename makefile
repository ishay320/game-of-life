

SRC=gol.c
CFLAGS=-Wall -Wextra -g

gol: ${SRC} xwrap.h
	${CC} ${CFLAGS} $< -o $@

xwrap.h:
	wget https://github.com/ishay320/XWrap/raw/main/xwrap.h