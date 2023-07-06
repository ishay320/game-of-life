

SRC=gol.c
CFLAGS=-Wall -Wextra -g

gol_x11: ${SRC} xwrap.h
	${CC} ${CFLAGS} $< -o $@ -DUSE_X11

gol: ${SRC} xwrap.h
	${CC} ${CFLAGS} $< -o $@

xwrap.h:
	wget https://github.com/ishay320/XWrap/raw/main/xwrap.h