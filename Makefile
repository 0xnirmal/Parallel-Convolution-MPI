CC=mpicc

all: gol
gol:
	$(CC) conv.c -o conv
clean:
	rm -r conv
