# Ignored warnings
IGNORED_WARNINGS = -Wno-unused-result -Wno-uninitialized -Wno-char-subscripts
OPENMP = -fopenmp

# Compiler and flags
CXX = g++
NVCC := nvcc

CXXFLAGS = -O2 -Wall $(IGNORED_WARNINGS) $(OPENMP)
NVCCFLAGS := -O2 -arch=sm_86 -rdc=true

# Directories
SRC_DIR := .
BOARD_DIR := board
MATH_LIB_DIR := math_lib

# Source files
SRCS := $(SRC_DIR)/main.cpp \
        $(BOARD_DIR)/board.cpp \
        $(MATH_LIB_DIR)/maths.cpp \
        $(SRC_DIR)/decide_cuda.cu

# Object files
OBJS := $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cu=.o)

# Executable name
TARGET := cuda_agent

# Default target: build without AMAF
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)	
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile C++ source files into object files
%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c -o $@ $<

# Compile CUDA source files into object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

seq:
	$(NVCC) *.cpp */*.cpp -o agent $(NVCCFLAGS) -DUSE_SEQ

# Target with AMAF
rave:
	$(CXX) *.cpp */*.cpp -o agent $(CXXFLAGS) -DUSE_AMAF

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

# ./agent
# 0 0 0
# 1 10 6 0 5 2
# 23 14 18 24 19 22
# 1
