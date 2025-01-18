/****************************************************************************
 *
 * inputgen.c
 *
 * Copyright (C) 2025 Enrico Marchionni
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

/***
This program generates a three-dimensional voxel grid and stores it into the specified binary file.

COMPILE:

    gcc -Wall -Wpedantic -std=c99 -fopenmp inputgen.c ./source/voxel.c -I./source/ -lm -o inputgen

This program can also be compiled so that the output file has no header (the 16 integer values described below). 
This output file structure should not be used as input to the projection.c program. Use the -DRAW argument when compiling:

    gcc -Wall -Wpedantic -std=c99 -fopenmp -DRAW inputgen.c ./source/voxel.c -I./source/ -lm -o inputgen

RUN:

    inputgen output.dat [object Type] [integer]

- First parameter is the name of the file to store the output in;
- Second parameter is optional and can be: 1 (solid cube with spherical cavity), 2 (solid sphere) or 3 (solid cube),
  if not passed 3 (solid cube) is default;
- Third parameter is the number of pixel per side of the detector, every other parameter is set based to its value,
  if no value is given, default values are used;

OUTPUT FILE STRUCTURE:

The voxel (three-dimensional) grid is represented as a stack of two-dimensional grids.
Considering a three-dimensional Cartesian system where the x-axis is directed from left to right, the y-axis is
directed upwards, and the z-axis is orthogonal to them,
a two-dimensional grid can be viewed as a horizontal slice, orthogonal to the y-axis, of the object.

First a sequence of 16 integer values is given, representing on order:
 - gl_pixelDim
 - gl_angularTrajectory
 - gl_positionsAngularDistance
 - gl_objectSideLength
 - gl_detectorSideLength
 - gl_distanceObjectDetector
 - gl_distanceObjectSource
 - gl_voxelXDim
 - gl_voxelYDim
 - gl_voxelZDim
 - gl_nVoxel[0]
 - gl_nVoxel[1]
 - gl_nVoxel[2]
 - gl_nPlanes[0]
 - gl_nPlanes[1]
 - gl_nPlanes[2]

Then, the values composing the voxel grid are given for a total of (gl_nVoxel[0] * gl_nVoxel[1] * gl_nVoxel[2]) (double)
values. Each sequence of length v 1 ∗ v 3 represents a horizontal slice of the object stored as a one-dimensional array
of elements ordered first by the x coordinate and then by the z coordinate. The first slice memorized is the bottom one,
followed by the other slices in ascending order of the y coordinate.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "common.h"

#define OBJ_BUFFER 100 // limits the number of voxel along the y axis computed per time
#define N_PIXEL_ALONG_SIDE (DETECTOR_SIDE_LENGTH / PIXEL_DIM)
// #define DEFAULT_WORK_SIZE 2352 // default work size, equals to floor(DETECTOR_SIDE_LENGTH / PIXEL_DIM)

#define DEFAULT_FILE_NAME "input.dat"

#define CUBE "cube"
#define SPHERE "half_sphere"
#define CUBE_WITH_SPHERICAL_HOLE "cube_with_spherical_hole"

/**
 * The following global variables are defined as according to common.h header file.
 * In order to use them with the value given below, the third parameter must not be passed at launch of 'inputgen' program.
 * In case the third value is given at launch, this will be used to compute the value of gl_objectSideLength, gl_detectorSideLength,
 * gl_distanceObjectDetector and gl_distanceObjectSource; the remaining variables will keep the value given below.
 */
int gl_pixelDim = PIXEL_DIM;
int gl_angularTrajectory = ANGULAR_TRAJECTORY;
int gl_positionsAngularDistance = POSITIONS_ANGULAR_DISTANCE;
int gl_objectSideLength = OBJECT_SIDE_LENGTH;
int gl_detectorSideLength = DETECTOR_SIDE_LENGTH;
int gl_distanceObjectDetector = DISTANCE_OBJECT_DETECTOR;
int gl_distanceObjectSource = DISTANCE_OBJECT_SOURCE;
int gl_voxelXDim = VOXEL_X_DIM;
int gl_voxelYDim = VOXEL_Y_DIM;
int gl_voxelZDim = VOXEL_Z_DIM;

/**
 * The following arrays' value must be computed as follows:
 *     gl_nVoxel[3] = {gl_objectSideLength / gl_voxelXDim, gl_objectSideLength / gl_voxelYDim, gl_objectSideLength / gl_voxelZDim};
 *     gl_nPlanes[3] = {(gl_objectSideLength / gl_voxelXDim) + 1, (gl_objectSideLength / gl_voxelYDim) + 1, (gl_objectSideLength / gl_voxelZDim) + 1};
 */
int gl_nVoxel[3] = {N_VOXEL_X, N_VOXEL_Y, N_VOXEL_Z};
int gl_nPlanes[3] = {N_PLANES_X, N_PLANES_Y, N_PLANES_Z};

/**
 * Stores the environment values used to compute the voxel grid into the specified binary file.
 * 'filePointer' the file pointer to store the values in.
 * @returns the number of bytes which the header is made up of, 0 in case an error occurs on writing
 */
unsigned long writeSetUp(FILE *filePointer)
{
    int setUp[] = { gl_pixelDim,
                    gl_angularTrajectory,
                    gl_positionsAngularDistance,
                    gl_objectSideLength,
                    gl_detectorSideLength,
                    gl_distanceObjectDetector,
                    gl_distanceObjectSource,
                    gl_voxelXDim,
                    gl_voxelYDim,
                    gl_voxelZDim,
                    gl_nVoxel[0],
                    gl_nVoxel[1],
                    gl_nVoxel[2],
                    gl_nPlanes[0],
                    gl_nPlanes[1],
                    gl_nPlanes[2],
                    };

    if (!fwrite(setUp, sizeof(int), sizeof(setUp) / sizeof(int), filePointer)) {
        return 0;
    }
    return sizeof(setUp);
}

/**
 * Implements generateCubeSlice as according to voxel.h header file.
*/
void generateCubeSlice(double *f, int nOfSlices, int offset, int sideLength)
{
    const int innerToOuterDiff = gl_nVoxel[X] / 2 - sideLength / 2;
    const int rightSide = innerToOuterDiff + sideLength;

    // Iterates over each voxel of the grid
#pragma omp parallel for collapse(3) default(none) shared(f, nOfSlices, gl_nVoxel, offset, sideLength, innerToOuterDiff, rightSide)
    for (int n = 0 ; n < nOfSlices; n++) {
        for (int i = 0; i < gl_nVoxel[Z]; i++) {
            for (int j = 0; j < gl_nVoxel[X]; j++) {
                if ((i >= innerToOuterDiff) && (i <= rightSide) && (j >= innerToOuterDiff) && (j <= rightSide)
                    && (n + offset >= innerToOuterDiff) && (n + offset <=  gl_nVoxel[Y] - innerToOuterDiff)) {
                    // Voxel position is inside the cubic object
                    f[gl_nVoxel[Z] * i + j + n * gl_nVoxel[X] * gl_nVoxel[Z]] = 1.0;
                } else {
                    f[gl_nVoxel[Z] * i + j + n * gl_nVoxel[X] * gl_nVoxel[Z]] = 0.0;
                }
            }
        }
    }
}

/**
 * Implements generateSphereSlice as according to voxel.h header file.
*/
void generateSphereSlice(double *f, int nOfSlices, int offset, int diameter)
{
    // Iterates over each voxel of the grid
#pragma omp parallel for collapse(3) default(none) shared(f, nOfSlices, gl_nVoxel, offset, diameter, gl_objectSideLength, gl_voxelYDim, gl_voxelXDim, gl_voxelZDim)
    for (int n = 0; n < nOfSlices; n++) {
        for (int r = 0; r < gl_nVoxel[Z]; r++) {
            for (int c = 0; c < gl_nVoxel[X]; c++) {
                Point temp;
                temp.y = -(gl_objectSideLength / 2) + (gl_voxelYDim / 2) + (n + offset) * gl_voxelYDim;
                temp.x = -(gl_objectSideLength / 2) + (gl_voxelXDim / 2) + (c) * gl_voxelXDim;
                temp.z = -(gl_objectSideLength / 2) + (gl_voxelZDim / 2) + (r) * gl_voxelZDim;
                const double distance = sqrt(pow(temp.x, 2) + pow(temp.y, 2) + pow(temp.z, 2));
                if (distance <= diameter && c < gl_nVoxel[Z] / 2) {
                    // Voxel position is inside the sphere object
                    f[gl_nVoxel[Z] * r + c + n * gl_nVoxel[X] * gl_nVoxel[Z]] = 1.0;
                } else {
                    f[gl_nVoxel[Z] * r + c + n * gl_nVoxel[X] * gl_nVoxel[Z]] = 0.0;
                }
            }
        }
    }
}

/**
 * Implements generateCubeWithSphereSlice as according to voxel.h header file.
*/
void generateCubeWithSphereSlice(double *f, int nOfSlices, int offset, const int sideLength)
{
    const int innerToOuterDiff = gl_nVoxel[X] / 2 - sideLength / 2;
    const int rightSide = innerToOuterDiff + sideLength;
    const Point sphereCenter = {-sideLength * gl_voxelXDim / 4, -sideLength * gl_voxelYDim / 4, -sideLength * gl_voxelZDim / 4};

    // Iterates over each voxel of the grid
#pragma omp parallel for collapse(3) default(none) shared(f, nOfSlices, offset, sideLength, gl_nVoxel, innerToOuterDiff, rightSide, sphereCenter, gl_objectSideLength, gl_voxelYDim, gl_voxelXDim, gl_voxelZDim)
    for (int n = 0 ; n < nOfSlices; n++) {
        for (int i = 0; i < gl_nVoxel[Z]; i++) {
            for (int j = 0; j < gl_nVoxel[X]; j++) {
                f[(gl_nVoxel[Z]) * i + j + n * gl_nVoxel[X] * gl_nVoxel[Z]] = 0.0;
                if ((i >= innerToOuterDiff) && (i <= rightSide) && (j >= innerToOuterDiff) && (j <= rightSide)
                    && (n + offset >= innerToOuterDiff) && (n + offset <=  gl_nVoxel[Y] - innerToOuterDiff)) {
                    // Voxel position is inside the cubic object
                    Point temp;
                    temp.y = -(gl_objectSideLength / 2) + (gl_voxelYDim / 2) + (n + offset) * gl_voxelYDim;
                    temp.x = -(gl_objectSideLength / 2) + (gl_voxelXDim / 2) + (j) * gl_voxelXDim;
                    temp.z = -(gl_objectSideLength / 2) + (gl_voxelZDim / 2) + (i) * gl_voxelZDim;
                    const double distance = sqrt(pow(temp.x - sphereCenter.x, 2) + pow(temp.y - sphereCenter.y, 2) + pow(temp.z - sphereCenter.z, 2));

                    if (distance > sideLength * gl_voxelXDim / 6) {
                        // Voxel position is outside the spherical cavity
                        f[(gl_nVoxel[Z]) * i + j + n * gl_nVoxel[X] * gl_nVoxel[Z]] = 1.0;
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    char *fileName = DEFAULT_FILE_NAME;
    int objectType = 0; // represents the object type chosen
    // int n = DEFAULT_WORK_SIZE; // number of voxels along the detector's

    if (argc > 4) {
        fprintf(stderr,
            "Usage:\n%s [DEST] [OBJECT] [PIXELS]\nWhere:\n"
            " - [DEST]: is the output file name, default is '%s'.\n"
            " - [OBJECT]: can be: '%s' (default), '%s' or '%s'.\n"
            " - [PIXELS]: is the number of pixels per side of the detector, every other parameter is set based to its"
            " value, if no value is given, default values are used.\n",
            DEFAULT_FILE_NAME,
            CUBE,
            SPHERE,
            CUBE_WITH_SPHERICAL_HOLE,
            argv[0]);
        return EXIT_FAILURE;
    }
    if (argc > 1) {
        fileName = argv[1];
        if (argc > 2) {
            if (strcmp(argv[2], CUBE_WITH_SPHERICAL_HOLE) == 0) {
                objectType = 1;
            } else if (strcmp(argv[2], SPHERE) == 0) {
                objectType = 2;
            } else if (strcmp(argv[2], CUBE) != 0) {
                fprintf(stderr,
                    "Usage:\n%s [DEST] [OBJECT] [PIXELS]\nWhere:\n"
                    " - [OBJECT]: can be: '%s' (default), '%s' or '%s', nothing else is accepted.\n",
                    CUBE,
                    SPHERE,
                    CUBE_WITH_SPHERICAL_HOLE,
                    argv[0]);
                return EXIT_FAILURE;
            }
            if (argc > 3) {
                const int n = atoi(argv[3]);
                gl_objectSideLength = n * gl_voxelXDim * ((double) OBJECT_SIDE_LENGTH / (VOXEL_X_DIM * N_PIXEL_ALONG_SIDE));
                gl_detectorSideLength = n * gl_pixelDim;
                gl_distanceObjectDetector = 1.5 * gl_objectSideLength;
                gl_distanceObjectSource = 6 * gl_objectSideLength;
            }
        }
    }

    gl_nVoxel[X] = gl_objectSideLength / gl_voxelXDim;
    gl_nVoxel[Y] = gl_objectSideLength / gl_voxelYDim;
    gl_nVoxel[Z] = gl_objectSideLength / gl_voxelZDim;

    gl_nPlanes[X] = (gl_objectSideLength / gl_voxelXDim) + 1;
    gl_nPlanes[Y] = (gl_objectSideLength / gl_voxelYDim) + 1;
    gl_nPlanes[Z] = (gl_objectSideLength / gl_voxelZDim) + 1;

    // Array containing the coefficients of each voxel
    double *grid = (double *) malloc(sizeof(double) * gl_nVoxel[X] * gl_nVoxel[Z] * OBJ_BUFFER);

    FILE *filePointer = fopen(fileName, "wb");
    if (!filePointer) {
        fprintf(stderr, "Unable to open file '%s' in write binary mode!\n", fileName);
        return EXIT_FAILURE;
    }

#ifndef RAW
    // Write the voxel grid dimensions on file
    const unsigned long headerLength = writeSetUp(filePointer);
    if (!headerLength) {
        fprintf(stderr, "Unable to write on file '%s'!\n", fileName);
        return EXIT_FAILURE;
    }
#endif

    // Iterates over each object subsection which size is limited along the y coordinate by OBJ_BUFFER
    for (int slice = 0; slice < gl_nVoxel[Y]; slice += OBJ_BUFFER) {
        int nOfSlices;

        if (gl_nVoxel[Y] - slice < OBJ_BUFFER) {
            nOfSlices = gl_nVoxel[Y] - slice;
        } else {
            nOfSlices = OBJ_BUFFER;
        }

        // Generates object subsection
        switch (objectType) {
            case 1:
                generateCubeWithSphereSlice(grid, nOfSlices, slice, gl_nVoxel[X]);
                break;
            case 2:
                generateSphereSlice(grid, nOfSlices, slice, gl_objectSideLength / 2);
                break;
            default:
                generateCubeSlice(grid, nOfSlices, slice, gl_nVoxel[X]);
                break;
        }

        if (!fwrite(grid, sizeof(double), gl_nVoxel[X] * gl_nVoxel[Z] * nOfSlices, filePointer)) {
            fprintf(stderr, "Unable to write on file '%s'!\n", fileName);
            return EXIT_FAILURE;
        }
    }

    printf("Output file details:\n");
    printf("\tVoxel model size: %lu byte\n", sizeof(double) * gl_nVoxel[X] * gl_nVoxel[Z] * gl_nVoxel[Y]);
    printf("\tImage type: %lu bit real\n", sizeof(double) * 8);
    printf("\tImage width: %d pixels\n", gl_nVoxel[X]);
    printf("\tImage height: %d pixels\n", gl_nVoxel[Z]);
    printf("\tOffset to first image: %lu bytes\n", headerLength);
    printf("\tNumber of images: %d\n", gl_nVoxel[Y]);
    printf("\tGap between images: 0 bytes\n");

    fclose(filePointer);
    free(grid);
}
