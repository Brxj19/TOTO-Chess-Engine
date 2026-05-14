CC = gcc
CXX = g++
CFLAGS = -O3 -ffast-math -Iinclude -I.
CXXFLAGS = -O3 -ffast-math -Iinclude -I.
LDFLAGS = -lstdc++ -lm

SRC = toto.c nnue_eval.c nnue/nnue.cpp nnue/misc.cpp tce_nnue/tce_nnue.c tce_nnue/tce_nnue_loader.c tce_nnue/tce_nnue_network.c tce_nnue/tce_nnue_accumulator.c tce_nnue/tce_nnue_features.c
OBJ = $(SRC:.c=.o)
OBJ := $(OBJ:.cpp=.o)

TARGET = tce
TCENNUE_CHECKER = tools/nnue_train/check_tcennue_loader
TCENNUE_INFER_CHECKER = tools/nnue_train/check_tcennue_inference
FILE ?= data/nnue_runs/baseline/tce_baseline.tcennue
VECTORS ?= tools/nnue_train/test_vectors/inference_sample.json

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(TARGET) $(TCENNUE_CHECKER) $(TCENNUE_INFER_CHECKER)

check-tcennue: $(TCENNUE_CHECKER)
	./$(TCENNUE_CHECKER) $(FILE)

$(TCENNUE_CHECKER): tools/nnue_train/check_tcennue_loader.c tce_nnue/tce_nnue.c tce_nnue/tce_nnue_loader.c
	$(CC) $(CFLAGS) -o $@ tools/nnue_train/check_tcennue_loader.c tce_nnue/tce_nnue.c tce_nnue/tce_nnue_loader.c

check-tcennue-infer: $(TCENNUE_INFER_CHECKER)
	./$(TCENNUE_INFER_CHECKER) $(FILE) $(VECTORS)

$(TCENNUE_INFER_CHECKER): tools/nnue_train/check_tcennue_inference.c tce_nnue/tce_nnue.c tce_nnue/tce_nnue_loader.c tce_nnue/tce_nnue_network.c tce_nnue/tce_nnue_accumulator.c tce_nnue/tce_nnue_features.c
	$(CC) $(CFLAGS) -o $@ tools/nnue_train/check_tcennue_inference.c tce_nnue/tce_nnue.c tce_nnue/tce_nnue_loader.c tce_nnue/tce_nnue_network.c tce_nnue/tce_nnue_accumulator.c tce_nnue/tce_nnue_features.c -lm
