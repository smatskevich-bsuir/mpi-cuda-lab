##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
MPICC=mpic++
MPICC_FLAGS=-std=c++17


##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-arch=sm_50 -std=c++17

# CUDA library directory:
CUDA_LIB_DIR= -L/usr/local/cuda/lib64
# CUDA include directory:
CUDA_INC_DIR= -I/usr/local/cuda/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = /cluster/computer

# Image name:
IMAGE = mpi-cuda

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/file_queue.o $(OBJ_DIR)/image_processing.o

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	@mkdir -p $(@D)
	$(MPICC) $(MPICC_FLAGS) $(CUDA_INC_DIR) $(OBJS) -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	@mkdir -p $(@D)
	$(MPICC) $(MPICC_FLAGS) $(CUDA_INC_DIR) -c $< -o $@ 

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(INC_DIR)/%.hpp
	@mkdir -p $(@D)
	$(MPICC) $(MPICC_FLAGS) -c $< -o $@ 

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	@mkdir -p $(@D)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 

# Clean objects in object directory.
clean:
	$(RM) -r bin

#build image
image:
	docker build -t $(IMAGE) .