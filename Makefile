CC = mpicc
pagerank: pagerank.c
	$(CC) -std=c99 -march=native -O3 -Wall -Wno-unused-result pagerank.c -o pagerank -lm

clean:
	rm -f pagerank
