SHELL := /bin/bash

## BUILD dir

DIR_BUILD := build
DIR_BIN := $(DIR_BUILD)/bin
DIR_OBJ := $(DIR_BUILD)/obj

DIRS := $(DIR_BIN) $(DIR_OBJ) $(DIR_BUILD)

## PROGRAMS

#SERIAL := serial
OPENMP := omp
CUDA := cuda

TARGETS := $(OPENMP) $(CUDA) # $(SERIAL)

EXE_INPUT := inputgen
TARGET_INPUT := $(addprefix $(DIR_BIN)/, $(EXE_INPUT))
# EXE_SERIAL := skyline
# TARGET_SERIAL := $(addprefix $(DIR_BIN)/, $(EXE_SERIAL))
EXE_OMP := $(basename $(wildcard omp-*.c))
TARGET_OMP := $(addprefix $(DIR_BIN)/, $(EXE_OMP))
EXE_CUDA := $(basename $(wildcard cuda-*.cu))
TARGET_CUDA := $(addprefix $(DIR_BIN)/, $(EXE_CUDA))

EXE := $(EXE_INPUT) $(EXE_OMP) $(EXE_CUDA) # $(EXE_SERIAL)

# C
CC = gcc
CFLAGS += -std=c99 -Wall -Wpedantic -Werror -O2 -D_XOPEN_SOURCE=600
LDLIBS += -lm

# CUDA
NVCC := nvcc
NVCFLAGS += -Wno-deprecated-gpu-targets
NVLDLIBS += -lm

all: $(TARGETS) $(EXE_INPUT)

$(INPUT): $(EXE_INPUT)
$(EXE_INPUT): $(TARGET_INPUT)
$(TARGET_INPUT): CFLAGS += -fopenmp

# $(SERIAL): $(EXE_SERIAL)
# $(EXE_SERIAL): $(TARGET_SERIAL)

$(OPENMP): $(EXE_OMP)
$(EXE_OMP): $(TARGET_OMP)
$(TARGET_OMP): CFLAGS += -fopenmp

$(CUDA): $(EXE_CUDA)
$(EXE_CUDA): $(TARGET_CUDA)

help:
	@echo
	@echo "Available targets:"
	@echo
	@echo " - Data:"
	@echo
	@echo "     inputs    generate all input files, you can change the dimensions,"
	@echo "               by default the variable WORK_SIZE is set to 2352"
	@echo
	@echo " - Code:"
	@echo
	@echo "       help    prints this message"
	@echo "        all    compile everything (default)"
	@echo "   inputgen    compile input generator program"
#	@echo "     serial    compile serial program"
	@echo "        omp    compile parallel OpenMP program"
	@echo "       cuda    compile CUDA program"
	@echo "      clean    cleanup generated executables"
	@echo "      purge    brutally removes build/* with all contents"
	@echo

.PRECIOUS: $(DIR_OBJ)/%.o

# C and OpenMP

$(DIR_BIN)/%: $(DIR_OBJ)/%.o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(DIR_OBJ)/%.o: %.c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@ $(LDLIBS)

# CUDA

$(DIR_BIN)/%: %.cu
	@mkdir -p $(DIR_BIN)
	$(NVCC) $(NVCFLAGS) $^ -o $@ $(NVLDLIBS)

# INPUT

$(DIR_BIN)/$(EXE_INPUT): $(DIR_OBJ)/$(EXE_INPUT).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(DIR_OBJ)/$(EXE_INPUT).o: $(EXE_INPUT).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	-\rm -f $(addprefix $(DIR_OBJ)/, $(addsuffix .o, $(EXE))) $(addprefix $(DIR_BIN)/, $(EXE))
	-\rm -df $(DIRS)

purge:
	-\rm -rf $(DIR_BUILD)

## INPUT

DIR_INPUT := input

CUBE_WITH_SPHERICAL_HOLE := cube_with_spherical_hole
HALF_SPHERE := half_sphere
CUBE := cube

SHAPES := $(CUBE_WITH_SPHERICAL_HOLE) $(HALF_SPHERE) $(CUBE)

inputs: $(addprefix $(DIR_INPUT)/, $(addsuffix .dat, $(SHAPES)))

$(DIR_INPUT)/$(CUBE_WITH_SPHERICAL_HOLE).dat: $(DIR_BIN)/$(EXE_INPUT)
	@mkdir -p $(DIR_INPUT)
	$< $@ $(CUBE_WITH_SPHERICAL_HOLE) $(WORK_SIZE)

$(DIR_INPUT)/$(HALF_SPHERE).dat: $(DIR_BIN)/$(EXE_INPUT)
	@mkdir -p $(DIR_INPUT)
	$< $@ $(HALF_SPHERE) $(WORK_SIZE)

$(DIR_INPUT)/$(CUBE).dat: $(DIR_BIN)/$(EXE_INPUT)
	@mkdir -p $(DIR_INPUT)
	$< $@ $(CUBE) $(WORK_SIZE)

## Special targets

.PHONY: all help clean purge $(TARGETS) $(EXE) $(SHAPES)
