
SRC=gol.cu
CFLAGS=-w
CC=gcc
NVCC=nvcc

gol_x11: ${SRC} xwrap.o
	${NVCC} ${CFLAGS} xwrap.o $< -o $@ -DUSE_X11

xwrap.o: xwrap.c xwrap.h
	${CC} -c $< -o $@

gol: ${SRC}
	${NVCC} ${CFLAGS} $< -o $@

xwrap.h:
	wget https://github.com/ishay320/XWrap/raw/main/xwrap.h

clean:
	-rm -f gol xwrap.h
.PHONY: clean