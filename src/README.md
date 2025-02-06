# Executables

This folder contains the source code of the project.

## Structure

### Documentation

The doxygen documentation of the code can be found inside the `docs` folder.

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

Execute this command to generate some simple projection images:
```shell
make images WORK_SIZE=400
```

> [!NOTE]
> This command is designed for quick and simple use, providing an overview of the output file system tree.

The generated file system tree will be:
- `build` contains miscellaneous files obtained from the source code.
  - `obj` contains the compiled source code.
  - `bin` contains the executables, `obj/*` linked version.
- `inputs` contains generated input `.dat` files.
- `outputs` contains output `.pgm` files and possibly their `.png` format.

Execute `make purge` to clean everything.

### Input Generation

The source file `inputgen.c` contains the code necessary to generate the input files used as 3D inputs for the projection
algorithm that will create the 2D image projections.

Usage:

```shell
./build/bin/inputgen DEST [OBJECT] [PIXELS]
```

Where:
- `DEST`: is the output file name, it is suggested to use `input/Cube.dat` for example.
- `[OBJECT]`: can be: `Cube` (default), `CubeWithSphericalHole` or `HalfSphere`.
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
./build/bin/omp-projection INPUT [OUTPUT] [Y_PLANES]
```

Where:
- `INPUT`: is the input file name of the 3D object, for example `input/cube.dat`.
- `[OUTPUT]`: is the output file name of the 2D projections, for example if compiled in normal mode it could be `output/Cube.pgm`,
  `/dev/null` can be used if you want to specify the y planes without creating an output file, for testing purposes.
- `[Y_PLANES]`: is the y axis slice size considered in the computation, that can be specified a priori, the default value is 100.

> [!TIP]
> Compile with: `make omp`.

### NVIDIA GPU Siddon's Projection

The source file `cuda-projection.cu` contains an CUDA parallel implementation of the Siddon's projection algorithm.

Usage:

```shell
./build/bin/cuda-projection INPUT [OUTPUT] [Y_PLANES]
```

Where:
- `INPUT`: is the input file name of the 3D object, for example `input/Cube.dat`.
- `[OUTPUT]`: is the output file name of the 2D projections, for example if compiled in normal mode it could be `output/Cube.pgm`,
  `/dev/null` can be used if you want to specify the y planes without creating an output file, for testing purposes.
- `[Y_PLANES]`: is the y axis slice size considered in the computation, that can be specified a priori, the default value is
  dynamically calculated depending on the GPU's available RAM.

> [!TIP]
> Compile with: `make cuda`.
> To compile CUDA sources in DEBUG mode run: `make cuda DEBUG=yes`.

### Throughput Scripts for Testing

The file `start-tests.sh` runs a script in background that executes CPU and then GPU throughput tests.

This script is designed to run independently of the terminal, making it suitable for execution on a remote server.

The folder used by the script are:
- `build` contains miscellaneous files obtained from the source code.
  - `obj` contains the compiled source code.
  - `bin` contains the executables, `obj/*` linked version.
- `inputs` contains generated input `.dat` files.
- `outputs` contains output `.pgm` files and possibly their `.png` format.
- `results` contains `.txt` files with the resulting wall clock times of the executed programs.
- `logs` contains `output.txt` and `error.txt` log files reporting standard output and error of the scripts.
