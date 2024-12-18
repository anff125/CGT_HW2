# Compiler
CXX := g++
NVCC := nvcc

# Compiler flags
CXXFLAGS := -std=c++11 -O2 -Wall
NVCCFLAGS := -O2 -arch=sm_60

# Directories
SRC_DIR := .
BOARD_DIR := board
MATH_LIB_DIR := math_lib

# Source files
SRCS := $(SRC_DIR)/main.cpp \
        $(BOARD_DIR)/board.cpp \
        $(MATH_LIB_DIR)/maths.cpp \
        $(SRC_DIR)/decide_parallel.cu

# Object files
OBJS := $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

# Executable name
TARGET := cgt_hw2

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile C++ source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile CUDA source files into object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Clean up generated files
clean:
	rm -f $(OBJS) $(TARGET)
