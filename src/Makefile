SHELL := /bin/bash

## BUILD dir

DIR_BUILD := build
DIR_BIN := $(DIR_BUILD)/bin
DIR_OBJ := $(DIR_BUILD)/obj

DIRS := $(DIR_BIN) $(DIR_OBJ) $(DIR_BUILD)

### PROGRAMS ###

# SERIAL := serial
OPENMP := omp
CUDA := cuda

TARGETS := $(OPENMP) $(CUDA) # $(SERIAL)

EXE_INPUT := inputgen
TARGET_INPUT := $(addprefix $(DIR_BIN)/, $(EXE_INPUT))
EXE_SERIAL := projection
# TARGET_SERIAL := $(addprefix $(DIR_BIN)/, $(EXE_SERIAL))
EXE_OMP_PROJECTION := omp-$(EXE_SERIAL)
TARGET_OMP_PROJECTION := $(addprefix $(DIR_BIN)/, $(EXE_OMP_PROJECTION))
EXE_OMP :=  $(filter-out $(EXE_OMP_PROJECTION), $(basename $(wildcard omp-*.c)))
TARGET_OMP := $(addprefix $(DIR_BIN)/, $(EXE_OMP))
EXE_CUDA_PROJECTION := cuda-$(EXE_SERIAL)
TARGET_CUDA_PROJECTION := $(addprefix $(DIR_BIN)/, $(EXE_CUDA_PROJECTION))
EXE_CUDA := $(filter-out $(EXE_CUDA_PROJECTION), $(basename $(wildcard cuda-*.c)))
TARGET_CUDA := $(addprefix $(DIR_BIN)/, $(EXE_CUDA))

EXE := $(EXE_INPUT) $(EXE_OMP) $(EXE_CUDA) $(EXE_OMP_PROJECTION) $(EXE_CUDA_PROJECTION) # $(EXE_SERIAL)

# C
CC = gcc
LFLAGS += -O2
CFLAGS += -std=c99 -Wall -Wpedantic -Werror -O2 -D_XOPEN_SOURCE=600
LDLIBS += -lm

# CUDA
NVCC := nvcc
NVCFLAGS += -Xcompiler "-Wall -Werror" -Wno-deprecated-gpu-targets

ifeq ($(DEBUG),all)
	CFLAGS += -DPRINT -DDEBUG -DPRINT_VARIABLES
	NVCFLAGS += -DPRINT -DDEBUG -DPRINT_VARIABLES -lineinfo
else ifeq ($(DEBUG),yes)
	CFLAGS += -DPRINT -DDEBUG
	NVCFLAGS += -DPRINT -DDEBUG -lineinfo
else
	ifeq ($(DEBUG),simple)
		CFLAGS += -DPRINT
		NVCFLAGS += -DPRINT
	endif
# The following macro optimize the final code:
# -DNDEBUG - avoids for example assertions checks
# -DNO_CUDA_CHECK_ERROR - avoids hpc.h checks on cuda kernel errors during execution
	CFLAGS += -DNDEBUG
	NVCFLAGS += -DNDEBUG -DNO_CUDA_CHECK_ERROR
endif

all: $(EXE_INPUT) $(TARGETS)

$(EXE_INPUT): $(TARGET_INPUT)
$(TARGET_INPUT): CFLAGS += -fopenmp
$(TARGET_INPUT): LFLAGS += -fopenmp

# $(SERIAL): $(EXE_SERIAL)
# $(EXE_SERIAL): $(TARGET_SERIAL)

$(OPENMP): $(EXE_OMP_PROJECTION) $(EXE_OMP)
$(EXE_OMP_PROJECTION): $(TARGET_OMP_PROJECTION)
$(TARGET_OMP_PROJECTION): CFLAGS += -fopenmp
$(TARGET_OMP_PROJECTION): LFLAGS += -fopenmp
$(EXE_OMP): $(TARGET_OMP)
$(TARGET_OMP): CFLAGS += -fopenmp
$(TARGET_OMP): LFLAGS += -fopenmp

$(CUDA): $(EXE_CUDA_PROJECTION) $(EXE_CUDA)
$(EXE_CUDA_PROJECTION): $(TARGET_CUDA_PROJECTION)
$(EXE_CUDA): $(TARGET_CUDA)

.PRECIOUS: $(DIR_OBJ)/%.o

# C and OpenMP

$(DIR_BIN)/%: $(DIR_OBJ)/%.o
	@mkdir -p $(DIR_BIN)
	$(CC) $(LFLAGS) $^ -o $@ $(LDLIBS)

$(DIR_OBJ)/%.o: %.c common.h hpc.h
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

# CUDA

$(DIR_BIN)/%: %.cu common.h hpc.h
	@mkdir -p $(DIR_BIN)
	$(NVCC) $(NVCFLAGS) $< -o $@

# INPUT

$(DIR_BIN)/$(EXE_INPUT): $(DIR_OBJ)/$(EXE_INPUT).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(LFLAGS) $^ -o $@ $(LDLIBS)

$(DIR_OBJ)/$(EXE_INPUT).o: $(EXE_INPUT).c common.h
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-\rm -f $(addprefix $(DIR_OBJ)/, $(addsuffix .o, $(EXE))) $(addprefix $(DIR_BIN)/, $(EXE))
	-\rm -df $(DIRS)

### DATA ###

ifdef WORK_SIZE
    SUFFIX_SIZE=$(WORK_SIZE)
endif

## INPUT

DIR_INPUT := inputs

CUBE := Cube
CUBE_WITH_SPHERICAL_HOLE := CubeWithSphericalHole
HALF_SPHERE := HalfSphere

CUBE_WITH_SUFFIX := $(CUBE)$(SUFFIX_SIZE)
CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX := $(CUBE_WITH_SPHERICAL_HOLE)$(SUFFIX_SIZE)
HALF_SPHERE_WITH_SUFFIX := $(HALF_SPHERE)$(SUFFIX_SIZE)

SHAPES := $(CUBE_WITH_SUFFIX) $(CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX) $(HALF_SPHERE_WITH_SUFFIX)

inputs: $(addprefix $(DIR_INPUT)/, $(addsuffix .dat, $(SHAPES)))

$(DIR_INPUT)/$(CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX).dat: $(DIR_BIN)/$(EXE_INPUT)
	@mkdir -p $(DIR_INPUT)
	$< $@ $(CUBE_WITH_SPHERICAL_HOLE) $(WORK_SIZE)

$(DIR_INPUT)/$(HALF_SPHERE_WITH_SUFFIX).dat: $(DIR_BIN)/$(EXE_INPUT)
	@mkdir -p $(DIR_INPUT)
	$< $@ $(HALF_SPHERE) $(WORK_SIZE)

$(DIR_INPUT)/$(CUBE_WITH_SUFFIX).dat: $(DIR_BIN)/$(EXE_INPUT)
	@mkdir -p $(DIR_INPUT)
	$< $@ $(CUBE) $(WORK_SIZE)

## OUTPUT

DIR_OUTPUT := outputs

# General

outputs: omp_outputs cuda_outputs
images: omp_images cuda_images

# OpenMP

DIR_OUTPUT_OMP := $(DIR_OUTPUT)/omp

omp_outputs: $(addprefix $(DIR_OUTPUT_OMP)/, $(addsuffix .pgm, $(SHAPES)))

$(DIR_OUTPUT_OMP)/$(CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX).pgm: $(DIR_BIN)/$(EXE_OMP_PROJECTION) $(DIR_INPUT)/$(CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX).dat
	@mkdir -p $(DIR_OUTPUT_OMP)
	$< $(word 2,$^) $@

$(DIR_OUTPUT_OMP)/$(HALF_SPHERE_WITH_SUFFIX).pgm: $(DIR_BIN)/$(EXE_OMP_PROJECTION) $(DIR_INPUT)/$(HALF_SPHERE_WITH_SUFFIX).dat
	@mkdir -p $(DIR_OUTPUT_OMP)
	$< $(word 2,$^) $@

$(DIR_OUTPUT_OMP)/$(CUBE_WITH_SUFFIX).pgm: $(DIR_BIN)/$(EXE_OMP_PROJECTION) $(DIR_INPUT)/$(CUBE_WITH_SUFFIX).dat
	@mkdir -p $(DIR_OUTPUT_OMP)
	$< $(word 2,$^) $@

omp_images: $(addprefix $(DIR_OUTPUT_OMP)/, $(addsuffix .png, $(SHAPES)))

$(DIR_OUTPUT_OMP)/$(CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX).png: $(DIR_OUTPUT_OMP)/$(CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX).pgm
	convert $^ $@

$(DIR_OUTPUT_OMP)/$(HALF_SPHERE_WITH_SUFFIX).png: $(DIR_OUTPUT_OMP)/$(HALF_SPHERE_WITH_SUFFIX).pgm
	convert $^ $@

$(DIR_OUTPUT_OMP)/$(CUBE_WITH_SUFFIX).png: $(DIR_OUTPUT_OMP)/$(CUBE_WITH_SUFFIX).pgm
	convert $^ $@

# OpenMP

DIR_OUTPUT_CUDA := $(DIR_OUTPUT)/cuda

cuda_outputs: $(addprefix $(DIR_OUTPUT_CUDA)/, $(addsuffix .pgm, $(SHAPES)))

$(DIR_OUTPUT_CUDA)/$(CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX).pgm: $(DIR_BIN)/$(EXE_CUDA_PROJECTION) $(DIR_INPUT)/$(CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX).dat
	@mkdir -p $(DIR_OUTPUT_CUDA)
	$< $(word 2,$^) $@

$(DIR_OUTPUT_CUDA)/$(HALF_SPHERE_WITH_SUFFIX).pgm: $(DIR_BIN)/$(EXE_CUDA_PROJECTION) $(DIR_INPUT)/$(HALF_SPHERE_WITH_SUFFIX).dat
	@mkdir -p $(DIR_OUTPUT_CUDA)
	$< $(word 2,$^) $@

$(DIR_OUTPUT_CUDA)/$(CUBE_WITH_SUFFIX).pgm: $(DIR_BIN)/$(EXE_CUDA_PROJECTION) $(DIR_INPUT)/$(CUBE_WITH_SUFFIX).dat
	@mkdir -p $(DIR_OUTPUT_CUDA)
	$< $(word 2,$^) $@

cuda_images: $(addprefix $(DIR_OUTPUT_CUDA)/, $(addsuffix .png, $(SHAPES)))

$(DIR_OUTPUT_CUDA)/$(CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX).png: $(DIR_OUTPUT_CUDA)/$(CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX).pgm
	convert $^ $@

$(DIR_OUTPUT_CUDA)/$(HALF_SPHERE_WITH_SUFFIX).png: $(DIR_OUTPUT_CUDA)/$(HALF_SPHERE_WITH_SUFFIX).pgm
	convert $^ $@

$(DIR_OUTPUT_CUDA)/$(CUBE_WITH_SUFFIX).png: $(DIR_OUTPUT_CUDA)/$(CUBE_WITH_SUFFIX).pgm
	convert $^ $@

### GENERAL ###

help:
	@echo
	@echo "Available targets:"
	@echo
	@echo " General:"
	@echo "          help  Prints this message"
	@echo "         purge  Brutally removes $(DIR_BUILD)/*, $(DIR_INPUT)/* and $(DIR_OUTPUT)/*, with all contents"
	@echo
	@echo " Code:"
	@echo
	@echo "           all  Compile everything (default)"
	@echo "      inputgen  Compile input generator program"
#	@echo "        serial  Compile serial program"
	@echo "           omp  Compile OpenMP program"
	@echo "          cuda  Compile CUDA program"
	@echo "         clean  Cleanup generated executables"
	@echo
	@echo " Data:"
	@echo
	@echo "        inputs  Generate all input files inside $(DIR_INPUT)/*.dat,"
	@echo "                you can change the dimensions with the variable WORK_SIZE"
	@echo "       outputs  Generate all output files:"
	@echo "                puts OpenMP outputs in $(DIR_OUTPUT_OMP)/*.pgm,"
	@echo "                places CUDA outputs in $(DIR_OUTPUT_CUDA)/*.pgm"
	@echo "        images  Generate all output files and converts them in .png format:"
	@echo "                places outputs in OpenMP and CUDA outputs folders"
	@echo "   omp_outputs  Generate OpenMP output files inside $(DIR_OUTPUT_OMP)/*.pgm"
	@echo "    omp_images  Generate OpenMP output images in $(DIR_OUTPUT_OMP)/*.png"
	@echo "  cuda_outputs  Generate CUDA output files inside $(DIR_OUTPUT_CUDA)/*.pgm"
	@echo "   cuda_images  Generate CUDA output images in $(DIR_OUTPUT_CUDA)/*.png"
	@echo
	@echo " Variables:"
	@echo
	@echo "     WORK_SIZE  Specifies the inputs dimensions, by default it is set to 2352,"
	@echo "                it can be specified in each Data target"
	@echo "         DEBUG  Can be set to:"
	@echo "                 - \"all\"    allows all debug messages, enables CUDA kernels error checking,"
	@echo "                            and line info to test also with CUDA compute-sanitizer tool"
	@echo "                 - \"yes\"    allows essential debug messages, enables CUDA kernels error checking,"
	@echo "                            and line info to test also with CUDA compute-sanitizer tool"
	@echo "                 - \"simple\" allows essential debug messages"
	@echo

purge:
	-\rm -rf $(DIR_BUILD)
	-\rm -rf $(DIR_INPUT)
	-\rm -rf $(DIR_OUTPUT)

## Special targets

.PHONY: all help clean purge $(TARGETS) $(EXE)
