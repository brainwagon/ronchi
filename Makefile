CFLAGS=-g -Wall
LIBS=-lm

all: 	ronchi

ronchi:	ronchi.o
	$(CC) -o ronchi $(CFLAGS) ronchi.o $(LIBS)
