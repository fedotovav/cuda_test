TARGET = cuda_test

BIN_DIR         = bin
OBJ_DIR         = obj

CU_TEST_SRCS = cuda.cu
CPP_SRCS     = utils.cpp main.cpp

OBJS = $(OBJ_DIR)/utils.o $(OBJ_DIR)/cuda.o $(OBJ_DIR)/main.o

DBG_FLAGS = #-g
RLS_FLAGS = -O2 -Ofast

CU_FLAGS = -gencode arch=compute_20,code=compute_20 $(DBG_FLAGS) 
CPP_FLAGS = -fopenmp $(DBG_FLAGS) $(RLS_FLAGS)
LINK_FLAGS = -fopenmp -lstdc++ -lgfortran -L/usr/local/cuda/lib64/ -lcuda -lcudart -lm
CPP_INCLUDE = -I/usr/local/cuda/include -I/usr/local/include

all: dir $(TARGET)

dir: 
	if !(test -d $(BIN_DIR)); then mkdir $(BIN_DIR); fi
	if !(test -d $(OBJ_DIR)); then mkdir $(OBJ_DIR); fi

$(TARGET): $(OBJS)
	gcc -Wall -o $(BIN_DIR)/$(TARGET) $(OBJS) $(LINK_FLAGS)

$(OBJ_DIR)/%.o: %.cpp
	g++ -c $< -o $@ $(CPP_INCLUDE) $(CPP_FLAGS)

$(OBJ_DIR)/%.o: %.cu
	/usr/local/cuda/bin/nvcc -w -c $< -o $@ $(CU_FLAGS)

clean:
	rm -rf $(BIN_DIR)/$(TARGET) $(OBJ_DIR)/*.o
