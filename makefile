CC = gcc
CXX = g++
CFLAGS = -O3 -ffast-math -Iinclude -I.
CXXFLAGS = -O3 -ffast-math -Iinclude -I.
LDFLAGS = -lstdc++

SRC = toto.c nnue_eval.c nnue/nnue.cpp nnue/misc.cpp
OBJ = $(SRC:.c=.o)
OBJ := $(OBJ:.cpp=.o)

TARGET = tce

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET)
