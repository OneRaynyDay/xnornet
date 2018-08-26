CC=g++
CFLAGS=-std=c++17 -Wall
SO_FLAGS=-shared -fPIC
LD_FLAGS=-I/Users/ray_zhang/anaconda3/envs/cling/include

.PHONY: all main

libxnor.so: logreg.cpp
	$(CC) $(SO_FLAGS) $(CFLAGS) $(LD_FLAGS) -o $@ $<

all: libxnor.so main

main: main.cpp
	$(CC) $(CFLAGS) $(LD_FLAGS) -o $@ $<

clean:
	rm -rf libxnor.so main make.log
