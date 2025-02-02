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
	CFLAGS += -DDEBUG -DPRINT_VARIABLES
	NVCFLAGS += -lineinfo $(MACROS)
else ifeq ($(DEBUG),yes)
	CFLAGS += -DDEBUG
	NVCFLAGS += -lineinfo $(MACROS)
else
# The following macro optimize the final code:
# -DNDEBUG - avoids for example assertions checks
# -DNO_CUDA_CHECK_ERROR - avoids hpc.h checks on cuda kernel errors during execution
	MACROS += -DNDEBUG
	NVCFLAGS += $(MACROS) -DNO_CUDA_CHECK_ERROR
endif

all: $(TARGETS) $(EXE_INPUT)

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

$(DIR_OBJ)/%.o: %.c hpc.h
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

# CUDA

$(DIR_BIN)/%: %.cu
	@mkdir -p $(DIR_BIN)
	$(NVCC) $(NVCFLAGS) $^ -o $@

# INPUT

$(DIR_BIN)/$(EXE_INPUT): $(DIR_OBJ)/$(EXE_INPUT).o
	@mkdir -p $(DIR_BIN)
	$(CC) $(LFLAGS) $^ -o $@ $(LDLIBS)

$(DIR_OBJ)/$(EXE_INPUT).o: $(EXE_INPUT).c
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	-\rm -f $(addprefix $(DIR_OBJ)/, $(addsuffix .o, $(EXE))) $(addprefix $(DIR_BIN)/, $(EXE))
	-\rm -df $(DIRS)

### DATA ###

ifndef WORK_SIZE
    SUFFIX_SIZE=default
else
    SUFFIX_SIZE=$(WORK_SIZE)
endif

## INPUT

DIR_INPUT := inputs

CUBE_WITH_SPHERICAL_HOLE := CubeWithSphericalHole
HALF_SPHERE := HalfSphere
CUBE := Cube

CUBE_WITH_SPHERICAL_HOLE_WITH_SUFFIX := $(CUBE_WITH_SPHERICAL_HOLE)$(SUFFIX_SIZE)
HALF_SPHERE_WITH_SUFFIX := $(HALF_SPHERE)$(SUFFIX_SIZE)
CUBE_WITH_SUFFIX := $(CUBE)$(SUFFIX_SIZE)

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
	@echo "        help    prints this message"
	@echo "       purge    brutally removes build/*, input/* and output/*"
	@echo "                with all contents"
	@echo
	@echo " Code:"
	@echo
	@echo "         all    compile everything (default)"
	@echo "    inputgen    compile input generator program"
#	@echo "      serial    compile serial program"
	@echo "         omp    compile parallel OpenMP program"
	@echo "        cuda    compile CUDA program,"
	@echo "                add DEBUG=yes if youu want to debug"
	@echo "       clean    cleanup generated executables"
	@echo
	@echo " Data:"
	@echo
	@echo "      inputs    generate all input files, you can change the dimensions,"
	@echo "                by default the variable WORK_SIZE is set to 2352"
	@echo "     outputs    generate all output files"
	@echo "                use OpenMP executable anmd then CUDA executable"
	@echo "                for inputs WORK_SIZE can be specified"
	@echo "      images    generate all output files in .png format"
	@echo "                use OpenMP executable anmd then CUDA executable"
	@echo "                for inputs WORK_SIZE can be specified"
	@echo " omp_outputs    generate all output files"
	@echo "                use OpenMP executable"
	@echo "                for inputs WORK_SIZE can be specified"
	@echo "  omp_images    generate all output files in .png format"
	@echo "                use OpenMP executable"
	@echo "                for inputs WORK_SIZE can be specified"
	@echo "cuda_outputs    generate all output files"
	@echo "                use CUDA executable"
	@echo "                for inputs WORK_SIZE can be specified"
	@echo " cuda_images    generate all output files in .png format"
	@echo "                use CUDA executable"
	@echo "                for inputs WORK_SIZE can be specified"
	@echo

purge:
	-\rm -rf $(DIR_BUILD)
	-\rm -rf $(DIR_INPUT)
	-\rm -rf $(DIR_OUTPUT)

## Special targets

.PHONY: all help clean purge $(TARGETS) $(EXE)
