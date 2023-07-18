
SRC=gol.cu
CFLAGS=-w -g
CC=nvcc

gol: ${SRC}
	${CC} ${CFLAGS} $< -o $@

clean:
	-rm -f gol
.PHONY: clean