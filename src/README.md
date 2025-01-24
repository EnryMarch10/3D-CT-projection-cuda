# Executables

This folder contains the source code of the project.

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

The source file `inputgen.c` contains the code necessary to generate the input files used as 3D inputs for the projection
algorithm that will create the 2D image projections.

Usage:

```shell
./build/bin/inputgen DEST [OBJECT] [PIXELS]
```

Where:
- `DEST`: is the output file name, it is suggested to use `input/cube.dat` for example.
- `[OBJECT]`: can be: `cube` (default), `cube_with_spherical_hole` or `half_sphere`.
- `[PIXELS]`: is the number of pixels per side of the detector, every other parameter is set based to its value, if no value is
  given, default is `2352`.

This is a similar version of the one proposed by [Lorenzo Colletta](https://github.com/mmarzolla/3D-CT-projection-openmp.git).
It uses the same functions to generate the input, and parallelizes the CPU computation in the same way.

> [!TIP]
> To generate all the input files locally with the Makefile run: `make inputs`.
> In this case the number of pixels of the detector can be set with `make inputs WORK_SIZE=N`, where `N` is the chosen number of
> pixels.

### CPU Siddon's Projection

The source file `omp-projection.c` contains an OpenMP parallel implementation of the Siddon's projection algorithm.

Usage:

```shell
./build/bin/omp-projection INPUT OUTPUT
```

Where:
- `INPUT`: is the input file name of the 3D object, for example `input/cube.dat`.
- `OUTPUT`: is the output file name of the 2D projections, for example if compiled in normal mode it could be `output/cube.pgm`,
  if compiled in binary mode it could be `output/cube.dat`.

> [!TIP]
> Compile with: `make omp`.
