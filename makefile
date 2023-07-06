

SRC=gol.c
CFLAGS=-Wall -Wextra -g

gol_x11: ${SRC} xwrap.h
	${CC} ${CFLAGS} $< -o $@ -DUSE_X11

gol: ${SRC}
	${CC} ${CFLAGS} $< -o $@

bb_x11: ${SRC} xwrap.h
	${CC} ${CFLAGS} $< -o $@ -DUSE_X11 -DBRIAN_BRAIN

bb: ${SRC}
	${CC} ${CFLAGS} $< -o $@ -DBRIAN_BRAIN

xwrap.h:
	wget https://github.com/ishay320/XWrap/raw/main/xwrap.h

clean:
	-rm -f gol_x11 gol bb_x11 bb xwrap.h
.PHONY: clean