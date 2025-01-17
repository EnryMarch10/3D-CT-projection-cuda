# Executables

This folder contains the executables developed in the project.

## Structure

### Makefile

The `make` command is very useful to compile the programs in this directory.

To understand what can be done with that command run:

```shell
make help
```

Then using one of the specified targets you can run:

```shell
make [TARGET] [VARIABLE=VALUE]
```

The **C** files are compiled inside `build/obj/` and their executables (linked version) can be found in `build/bin/`.

### Input Generation

The source file [inputgen.c](inputgen.c) contains the code necessary to generate the input files used as 3D inputs for the
projection algorithm that will create the 2D image projections.

This is a similar version of the one proposed by [Lorenzo Colletta](https://github.com/mmarzolla/3D-CT-projection-openmp.git).
It uses the same functions to generate the input, and parallelizes the CPU computation in the same way.

> **NOTE.** To generate all the input files locally with the Makefile run: `make inputs`.
