/**
 * @file omp-projection.c
 * @author Enrico Marchionni (enrico.marchionni@studio.unibo.it)
 * @brief An OpenMP implementation of the Siddon's projection algorithm.
 * @date 2025-01
 * @details
 * This file contains an implementation of the projection algorithm
 * for generating 2D projections of a 3D object.
 * The algorithm is based on Siddon's algorithm and is parallelized
 * using OpenMP.
 * The algorithm reads the 3D object from a file and writes the
 * reconstructed projection images to another file.
 *
 * The algorithm is divided into several functions, each of which
 * implements some steps from Siddon's algorithm.
 * The main function reads the 3D image from the file and computes
 * the 2D projections.
 * @copyright
 * ```text
 * This file is part of 3D-CT-projection-cuda
 * (https://github.com/EnryMarch10/3D-CT-projection-cuda).
 * Copyright (C) 2025 Enrico Marchionni
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 * ```
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "hpc.h"
#include "common.h"

#define OBJ_BUFFER 100
#define TABLES_DIM 1024

int gl_pixelDim;
int gl_angularTrajectory;
int gl_positionsAngularDistance;
int gl_objectSideLength;
int gl_detectorSideLength;
int gl_distanceObjectDetector;
int gl_distanceObjectSource;
int gl_voxelXDim;
int gl_voxelYDim;
int gl_voxelZDim;
int gl_nVoxel[3];
int gl_nPlanes[3];

double *sineTable;
double *cosineTable;

/**
 * @brief Initializes `sin` and `cos` tables, with default values for a certain length.
 *
 * @param sinTable An array containing a certain number of precalculated sin values.
 * @param cosTable An array containing a certain number of precalculated cos values.
 * @param length The length of the arrays.
 */
void initTables(double *sinTable, double *cosTable, int length)
{
    const int nTheta = gl_angularTrajectory / gl_positionsAngularDistance;  // Number of angular positions

    for (int positionIndex = 0; positionIndex <= nTheta; positionIndex++) {
        sinTable[positionIndex] = sin((-gl_angularTrajectory / 2 + positionIndex * gl_positionsAngularDistance) * M_PI / 180);
        cosTable[positionIndex] = cos((-gl_angularTrajectory / 2 + positionIndex * gl_positionsAngularDistance) * M_PI / 180);
    }
}

/**
 * @brief Computes the minimum value between `a` and `b`.
 *
 * @param a The first value.
 * @param b The second value.
 * @return The minimum between `a` and `b`.
 */
int min(int a, int b)
{
    return a < b ? a : b;
}

/**
 * @brief Computes the minimum value between `a`, `b` and `c`.
 *
 * @param a The first value.
 * @param b The second value.
 * @param c The third value.
 * @return The minimum between `a`, `b` and `c`.
 */
int min3(int a, int b, int c)
{
    return min(a, min(b, c));
}

/**
 * @brief Computes the coordinate of a plane parallel relative to the YZ plane.
 *
 * @param index It is the index of the plane to be returned where 0 is the index of the smallest-valued coordinate plane.
 * @return The coordinate of a plane parallel relative to the YZ plane.
 */
double getXPlane(int index)
{
    return -(gl_objectSideLength / 2) + index * gl_voxelXDim;
}

/**
 * @brief Computes the coordinate of a plane parallel relative to the XZ plane.
 *
 * @param index It is the index of the plane to be returned where 0 is the index of the smallest-valued coordinate plane.
 * @return The coordinate of a plane parallel relative to the XZ plane.
 */
double getYPlane(int index)
{
    return -(gl_objectSideLength / 2) + index * gl_voxelYDim;
}

/**
 * @brief Computes the coordinate of a plane parallel relative to the XY plane.
 *
 * @param index It is the index of the plane to be returned where 0 is the index of the smallest-valued coordinate plane.
 * @return The coordinate of a plane parallel relative to the XY plane.
 */
double getZPlane(int index)
{
    return -(gl_objectSideLength / 2) + index * gl_voxelZDim;
}

/**
 * @brief Computes the maximum parametric value a, representing the last intersection between ray and object.
 *
 * @param a It is the array containing the parametric value of the intersection between the ray and the object's side along each axis.
 * @param isParallel It is a value corresponding to the axis to which the array is orthogonal, -1 otherwise.
 * @return The maximum parametric value a, representing the last intersection between ray and object.
 */
double getAMax(double a[3][2], int isParallel)
{
    double tempMax[3];
    double aMax = 1;
    for (int i = 0; i < 3; i++) {
        if (i != isParallel) {
            tempMax[i] = a[i][0] > a[i][1] ? a[i][0] : a[i][1];
        }
    }
    for (int i = 0; i < 3; i++) {
        if (i != isParallel) {
            aMax = aMax < tempMax[i] ? aMax : tempMax[i];
        }
    }
    return aMax;
}

/**
 * @brief Computes the minimum parametric value a, representing the fist intersection between ray and object.
 *
 * @param a It is the array containing the parametric value of the intersection between the ray and the object's side along each axis.
 * @param isParallel It is a value corresponding to the axis to which the array is orthogonal, -1 otherwise.
 * @return The minimum parametric value a, representing the last intersection between ray and object.
 */
double getAMin(double a[3][2], int isParallel)
{
    double tempMin[3];
    double aMin = 0;
    for (int i = 0; i < 3; i++) {
        if (i != isParallel) {
            tempMin[i] = a[i][0] < a[i][1] ? a[i][0] : a[i][1];
        }
    }
    for (int i = 0; i < 3; i++) {
        if (i != isParallel) {
            aMin = aMin > tempMin[i] ? aMin : tempMin[i];
        }
    }
    return aMin;
}

/**
 * @brief Computes the the intersections between a ray and a set of planes.
 *
 * @param source Represents the coordinate of the source.
 * @param pixel Represents the coordinate of a unit of the detector, relative to the specified source.
 * @param plane It is an array that contains the coordinates of each plane.
 * @param nPlanes Specifies the number of planes.
 * @param a It is an array that will be filled with the parametric values that identify the intersection points between the
 * ray and each plane.
 * @return 0 if ray is parallel to the planes, 1 otherwise.
 */
int getIntersection(double source, double pixel, double *plane, int nPlanes, double *a)
{
    if (source - pixel != 0) {
        for (int i = 0; i < nPlanes; i++) {
            a[i] = (plane[i] - source) / (pixel - source);
        }
        return 1;
    }
    return 0;
}

/**
 * @brief Computes the coordinates of the planes necessary to compute the intersections with the ray.
 * Then it calls \ref getIntersection().
 *
 * @param source Represents the coordinate of the source.
 * @param pixel Represents the coordinate of a unit of the detector, relative to the specified source.
 * @param planeIndexes It is a structure containing the index ranges for planes.
 * @param a It is an array that will be filled with the parametric values that identify the intersection points between the
 * ray and each plane.
 * @param axis It is the axis orthogonal to the set of planes to which compute the intersection.
 */
void getAllIntersections(const double source, const double pixel, const Ranges planeIndexesRanges, double *a, Axis axis)
{
    int start = 0, end = 0;
    double d;

    start = planeIndexesRanges.minIndx;
    end = planeIndexesRanges.maxIndx;
    double plane[end - start];
    if (axis == X) {
        plane[0] = getXPlane(start);
        d = gl_voxelXDim;
        if (pixel - source < 0) {
            plane[0] = getXPlane(end);
            d = -gl_voxelXDim;
        }
    } else if (axis == Y) {
        plane[0] = getYPlane(start);
        d = gl_voxelYDim;
        if (pixel - source < 0) {
            plane[0] = getYPlane(end);
            d = -gl_voxelYDim;
        }
    } else /* if (axis == Z) */ {
        plane[0] = getZPlane(start);
        d = gl_voxelZDim;
        if (pixel - source < 0) {
            plane[0] = getZPlane(end);
            d = -gl_voxelZDim;
        }
    }

    for (int i = 1; i < end - start; i++) {
        plane[i] = plane[i - 1] + d;
    }
    getIntersection(source, pixel, plane, end - start, a);
}

/**
 * @brief Retrieves the range of parametric values of the planes.
 *
 * @param source Represents the coordinate of the source.
 * @param pixel Represents the coordinate of a unit of the detector, relative to the specified source.
 * @param isParallel It has a value corresponding to the axis to which the array is orthogonal, -1 otherwise.
 * @param aMin It is the minimum parametric value of the intersection between the ray and the object.
 * @param aMax It is the maximum parametric value of the intersection between the ray and the object.
 * @param axis It is the axis orthogonal to the plane.
 * @return The range of parametric values of the planes.
 */
Ranges getRangeOfIndex(const double source, const double pixel, int isParallel, double aMin, double aMax, Axis axis)
{
    Ranges idxs;
    double firstPlane, lastPlane;
    int voxelDim;

    if (axis == X) {
        voxelDim = gl_voxelXDim;
        firstPlane = getXPlane(0);
        lastPlane = getXPlane(gl_nPlanes[X] - 1);
    } else if (axis == Y) {
        voxelDim = gl_voxelYDim;
        firstPlane = getYPlane(0);
        lastPlane = getYPlane(gl_nPlanes[Y] - 1);
    } else /* if (ax == Z) */ {
        voxelDim = gl_voxelZDim;
        firstPlane = getZPlane(0);
        lastPlane = getZPlane(gl_nPlanes[Z] - 1);
    }

    // Gets range of indexes of XZ parallel planes
    if (isParallel != Y) {
        if (pixel - source >= 0) {
            idxs.minIndx = gl_nPlanes[axis] - ceil((lastPlane - aMin * (pixel - source) - source) / voxelDim);
            idxs.maxIndx = 1 + floor((aMax * (pixel - source) + source - firstPlane) / voxelDim);
        } else {
            idxs.minIndx = gl_nPlanes[axis] - ceil((lastPlane - aMax * (pixel - source) - source) / voxelDim);
            idxs.maxIndx = floor((aMin * (pixel - source) + source - firstPlane) / voxelDim);
        }
    } else {
        idxs.minIndx = 0;
        idxs.maxIndx = 0;
    }
    return idxs;
}

/**
 * @brief Merges two sorted arrays into one single sorted array.
 *
 * @param a It is a pointer to a sorted array.
 * @param b It is a pointer to a sorted array.
 * @param lenA It is the length of the array `a`.
 * @param lenB It is the length of the array `b`.
 * @param c It is the computed merged array.
 * @return The length of the merged array.
 */
int merge(double *a, double *b, int lenA, int lenB, double *c)
{
    int i = 0, j = 0, k = 0;
    while (j < lenA && k < lenB) {
        if (a[j] < b[k]) {
            c[i] = a[j];
            j++;
        } else {
            c[i] = b[k];
            k++;
        }
        i++;
    }
    while (j < lenA) {
        c[i] = a[j];
        i++;
        j++;
    }
    while (k < lenB) {
        c[i] = b[k];
        i++;
        k++;
    }
    return lenA + lenB;
}

/**
 * @brief Merges three sorted arrays into one single sorted array.
 *
 * @param a It is a pointer to a sorted array.
 * @param b It is a pointer to a sorted array.
 * @param c It is a pointer to a sorted array.
 * @param lenA It is the length of the array `a`.
 * @param lenB It is the length of the array `b`.
 * @param lenC It is the length of the array `c`.
 * @param merged It is the computed merged array.
 * @return The length of the merged array.
 */
int merge3(double *a, double *b, double *c, int lenA, int lenB, int lenC, double *merged)
{
    double ab[lenA + lenB];
    merge(a, b, lenA, lenB, ab);
    return merge(ab, c, lenA + lenB, lenC, merged);
}

/**
 * @brief Retrieves the cartesian coordinates of the source.
 *
 * @param index A value that defines the angle being considered.
 * @return The coordinates of the source.
 */
Point getSource(int index)
{
    Point source;

    source.z = 0;
    source.x = sineTable[index] * gl_distanceObjectSource;
    source.y = cosineTable[index] * gl_distanceObjectSource;

    return source;
}

/**
 * @brief Retrieves the cartesian coordinates of a unit of the detector.
 *
 * @param r The row of the detector matrix.
 * @param c The column of the detector matrix.
 * @param index A value that defines the angle being considered, and consequently, the source.
 * @return The coordinates of a unit of the detector, relative to the specified source.
 */
Point getPixel(int r, int c, int index)
{
    Point pixel;
    const double sinAngle = sineTable[index];
    const double cosAngle = cosineTable[index];
    const double elementOffset =  gl_detectorSideLength / 2 - gl_pixelDim / 2;

    pixel.x = -gl_distanceObjectDetector * sinAngle + cosAngle * (-elementOffset + gl_pixelDim * c);
    pixel.y = -gl_distanceObjectDetector * cosAngle - sinAngle * (-elementOffset + gl_pixelDim * c);
    pixel.z = -elementOffset + gl_pixelDim * r;

    return pixel;
}

/**
 * @brief Computes a coordinate of the two planes of the object's sides orthogonal to the x axis.
 *
 * @param planes It is a pointer to an array of two elements,
 * each one of them is the coordinate of a plane parallel relative to the YZ plane.
 */
void getSidesXPlanes(double *planes)
{
    planes[0] = getXPlane(0);
    planes[1] = getXPlane(gl_nPlanes[X] - 1);
}

/**
 * @brief Computes a coordinate of the two planes of the object's sides orthogonal to the y axis.
 *
 * @param planes It is a pointer to an array of two elements,
 * each one of them is the coordinate of a plane parallel relative to the XZ plane.
 * @param slice It is a number that indicates the first voxel in the y axis from which the projection is being computed.
 * In this case this limits the planes considered.
 */
void getSidesYPlanes(double *planes, int slice)
{
    planes[0] = getYPlane(slice);
    planes[1] = getYPlane(min(gl_nPlanes[Y] - 1, OBJ_BUFFER + slice));
}

/**
 * @brief Computes a coordinate of the two planes of the object's sides orthogonal to the z axis.
 *
 * @param planes It is a pointer to an array of two elements,
 * each one of them is the coordinate of a plane parallel relative to the XY plane.
 */
void getSidesZPlanes(double *planes)
{
    planes[0] = getZPlane(0);
    planes[1] = getZPlane(gl_nPlanes[Z] - 1);
}

/**
 * @brief Computes the projection attenuation of the radiological path of a ray.
 *
 * @param source Represents the coordinate of the source.
 * @param pixel Represents the coordinate of the unit of the detector.
 * @param a It is an array that contains all intersection points merged, expressed parametrically.
 * @param lenA It is the length of the corresponding array.
 * @param slice It is a number that indicates the first voxel in the y axis from which the projection is being computed.
 * @param f It is an array of the coefficients of attenuation for each voxel.
 * @return The computed projection attenuation of the radiological path of a ray.
 */
double computeAbsorption(Point source, Point pixel, double *a, int lenA, int slice, double *f)
{
    const double d12 = sqrt(pow(pixel.x - source.x, 2) + pow(pixel.y - source.y, 2) + pow(pixel.z - source.z, 2));

    double attenuation = 0.0;

    for (int i = 0; i < lenA - 1; i++) {
        const double segments = d12 * (a[i + 1] - a[i]);
        const double aMid = (a[i + 1] + a[i]) / 2;
        const int xRow = min((source.x + aMid * (pixel.x - source.x) - getXPlane(0)) / gl_voxelXDim, gl_nVoxel[X] - 1);
        const int yRow = min3((source.y + aMid * (pixel.y - source.y) - getYPlane(slice)) / gl_voxelYDim, gl_nVoxel[Y] - 1, OBJ_BUFFER - 1);
        const int zRow = min((source.z + aMid * (pixel.z - source.z) - getZPlane(0)) / gl_voxelZDim, gl_nVoxel[Z] - 1);

        attenuation += f[yRow * gl_nVoxel[X] * gl_nVoxel[Z] + zRow * gl_nVoxel[Z] + xRow] * segments;
    }
    return attenuation;
}

/**
 * @brief Computes the projection of a sub-section of the object into the detector for each source position.
 *
 * @param slice It is a number that indicates the first voxel in the y axis from which the projection is being computed.
 * @param f It is an array that contains the coefficients of attenuation of the voxels contained in the sub-section.
 * @param attenuation It is the resulting array that contains the value of the computed projection attenuation for each pixel.
 * @param absMax It is the maximum projection attenuation computed.
 * @param absMin It is the minimum projection attenuation computed.
 */
void computeProjections(int slice, double *f, double *attenuation, double *absMax, double *absMin)
{
    const int nTheta = gl_angularTrajectory / gl_positionsAngularDistance; // Number of angular positions
    const int nSidePixels = gl_detectorSideLength / gl_pixelDim;

    double amax = -INFINITY;
    double amin = INFINITY;
    double temp[3][2];
    double aMerged[gl_nPlanes[X] + gl_nPlanes[X] + gl_nPlanes[X]];
    double aX[gl_nPlanes[X]];
    double aY[gl_nPlanes[Y]];
    double aZ[gl_nPlanes[Z]];

    // Iterates over each source
    for (int positionIndex = 0; positionIndex <= nTheta; positionIndex++) {
        const Point source = getSource(positionIndex);

        // Iterates over each pixel of the detector
#pragma omp parallel for collapse(2) schedule(dynamic) default(none) shared(nSidePixels, positionIndex, source, slice, f, attenuation, nTheta, gl_nVoxel) private(temp, aX, aY, aZ, aMerged) reduction(min:amin) reduction(max:amax)
        for (int r = 0; r < nSidePixels; r++) {
            for (int c = 0; c < nSidePixels; c++) {
                Point pixel;

                // Gets the pixel's center cartesian coordinates
                pixel = getPixel(r, c, positionIndex);

                // Computes Min-Max parametric values
                double aMin, aMax;
                double sidesPlanes[2];
                int isParallel = -1;
                getSidesXPlanes(sidesPlanes);
                if (!getIntersection(source.x, pixel.x, sidesPlanes, 2, &temp[X][0])) {
                    isParallel = X;
                }
                getSidesYPlanes(sidesPlanes, slice);
                if (!getIntersection(source.y, pixel.y, sidesPlanes, 2, &temp[Y][0])) {
                    isParallel = Y;
                }
                getSidesZPlanes(sidesPlanes);
                if (!getIntersection(source.z, pixel.z, sidesPlanes, 2, &temp[Z][0])) {
                    isParallel = Z;
                }

                aMin = getAMin(temp, isParallel);
                aMax = getAMax(temp, isParallel);

                if (aMin < aMax) {
                    // Computes Min-Max plane indexes
                    Ranges indices[3];
                    indices[X] = getRangeOfIndex(source.x, pixel.x, isParallel, aMin, aMax, X);
                    indices[Y] = getRangeOfIndex(source.y, pixel.y, isParallel, aMin, aMax, Y);
                    indices[Z] = getRangeOfIndex(source.z, pixel.z, isParallel, aMin, aMax, Z);

                    // Computes lengths of the arrays containing parametric value of the intersection with each set of parallel planes
                    int lenX = indices[X].maxIndx - indices[X].minIndx;
                    int lenY = indices[Y].maxIndx - indices[Y].minIndx;
                    int lenZ = indices[Z].maxIndx - indices[Z].minIndx;
                    if (lenX < 0) {
                        lenX = 0;
                    }
                    if (lenY < 0) {
                        lenY = 0;
                    }
                    if (lenZ < 0) {
                        lenZ = 0;
                    }
                    const int lenA = lenX + lenY + lenZ;

                    // Computes ray-planes intersection Nx + Ny + Nz
                    getAllIntersections(source.x, pixel.x, indices[X], aX, X);
                    getAllIntersections(source.y, pixel.y, indices[Y], aY, Y);
                    getAllIntersections(source.z, pixel.z, indices[Z], aZ, Z);

                    // Computes segments Nx + Ny + Nz
                    merge3(aX, aY, aZ, lenX, lenY, lenZ, aMerged);

                    // Associates each segment to the respective voxel Nx + Ny + Nz
                    const int pixelIndex = positionIndex * nSidePixels * nSidePixels + r * nSidePixels + c;
                    attenuation[pixelIndex] += computeAbsorption(source, pixel, aMerged, lenA, slice, f);
                    amax = fmax(amax, attenuation[pixelIndex]);
                    amin = fmin(amin, attenuation[pixelIndex]);
                }
            }
        }
    }
    *absMax = amax;
    *absMin = amin;
}

/**
 * @brief Reads the environment values used to compute the voxel grid from the specified binary file.
 *
 * @param filePointer It is the file pointer to read the values from.
 * @return 0 in case of writing failure, 1 otherwise.
 */
int readSetUP(FILE *filePointer)
{
    int buffer[16];
    if (!fread(buffer, sizeof(int), 16, filePointer)) {
        return 0;
    }

    gl_pixelDim = buffer[0];
    gl_angularTrajectory = buffer[1];
    gl_positionsAngularDistance = buffer[2];
    gl_objectSideLength = buffer[3];
    gl_detectorSideLength = buffer[4];
    gl_distanceObjectDetector = buffer[5];
    gl_distanceObjectSource = buffer[6];
    gl_voxelXDim = buffer[7];
    gl_voxelYDim = buffer[8];
    gl_voxelZDim = buffer[9];
    gl_nVoxel[0] = buffer[10];
    gl_nVoxel[1] = buffer[11];
    gl_nVoxel[2] = buffer[12];
    gl_nPlanes[0] = buffer[13];
    gl_nPlanes[1] = buffer[14];
    gl_nPlanes[2] = buffer[15];

    return 1;
}

int main(int argc, char *argv[])
{
    FILE* inputFilePointer;
    FILE* outputFilePointer;

    if (argc != 3) {
// #ifdef BINARY
//         fprintf(stderr, "Usage: %s input.dat output.dat\n"
//                         "- The first parameter is the name of the input file.\n"
//                         "- The second parameter is the name of a binary file to store the output at.\n",
//                         argv[0]);
// #else
        fprintf(stderr, "Usage: %s input.dat output.pgm\n"
                        "- The first parameter is the name of the input file.\n"
                        "- The second parameter is the name of a binary file to store the output at.\n",
                        argv[0]);
// #endif
        return EXIT_FAILURE;
    }
    const char* inputFileName = argv[1];
    const char* outputFileName = argv[2];

    inputFilePointer = fopen(inputFileName, "rb");
    if (!inputFilePointer) {
        fprintf(stderr, "Unable to open file '%s'!\n", inputFileName);
        return EXIT_FAILURE;
    }

    if (!readSetUP(inputFilePointer)) {
        fprintf(stderr, "Unable to read from file '%s'!\n", inputFileName);
        return EXIT_FAILURE;
    }

    int nSidePixels = gl_detectorSideLength / gl_pixelDim;

    // Number of angular positions
    const int nTheta = gl_angularTrajectory / gl_positionsAngularDistance;
    // Array containing the coefficients of each voxel
    double *grid = (double *) malloc(sizeof(double) * gl_nVoxel[X] * gl_nVoxel[Z] * OBJ_BUFFER);
    // Array containing the computed attenuation detected in each pixel of the detector
    double *attenuation = (double *) calloc(nSidePixels * nSidePixels * (nTheta + 1), sizeof(double));
    // Each thread will have its own variable to store its minimum and maximum attenuation computed
    double absMaxValue, absMinValue;

    sineTable = (double *) malloc(sizeof(double) * TABLES_DIM);
    cosineTable = (double *) malloc(sizeof(double) * TABLES_DIM);

    initTables(sineTable, cosineTable, TABLES_DIM);

    double totalTime = 0.0;

// #ifdef BINARY
//     outputFilePointer = fopen(outputFileName, "wb");
//     if (!outputFileName) {
//         fprintf(stderr, "Unable to open file '%s'!\n", outputFileName);
//         return EXIT_FAILURE;
//     }
// #else
    outputFilePointer = fopen(outputFileName, "w");
    if (!outputFileName) {
        fprintf(stderr, "Unable to open file '%s'!\n", outputFileName);
        return EXIT_FAILURE;
    }
// #endif

    // Iterates over object subsection
    for (int slice = 0; slice < gl_nVoxel[Y]; slice += OBJ_BUFFER) {
        int nOfSlices;

        if (gl_nVoxel[Y] - slice < OBJ_BUFFER) {
            nOfSlices = gl_nVoxel[Y] - slice;
        } else {
            nOfSlices = OBJ_BUFFER;
        }

        // Read voxels coefficients
        if (!fread(grid, sizeof(double), gl_nVoxel[X] * gl_nVoxel[Z] * nOfSlices, inputFilePointer)) {
            fprintf(stderr, "Unable to read from file '%s'!\n", inputFileName);
            return EXIT_FAILURE;
        }

        // Computes subsection projection
        double partialTime = hpc_gettime();
        computeProjections(slice, grid, attenuation, &absMaxValue, &absMinValue);
        totalTime += hpc_gettime() - partialTime;
    }
    fprintf(stderr, "Execution time (s) %.2f\n", totalTime);
    fflush(stderr);

    // Write on file
// #ifdef BINARY
//     int matrixDetails[] = {nTheta + 1, nSidePixels};
//     if (!fwrite(matrixDetails, sizeof(int), 2, outputFilePointer)) {
//         fprintf(stderr, "Unable to write on file '%s'!\n", outputFileName);
//         return EXIT_FAILURE;
//     }
//     double minMaxValues[] = {absMaxValue, absMinValue};
//     if (!fwrite(minMaxValues, sizeof(double), 2, outputFilePointer)) {
//         fprintf(stderr, "Unable to write on file '%s'!\n", outputFileName);
//         return EXIT_FAILURE;
//     }
//     for (int i = 0; i <= nTheta; i++) {
//         double angle = -gl_angularTrajectory / 2 + i * gl_positionsAngularDistance;
//         if (!fwrite(&angle, sizeof(double), 1, outputFilePointer)) {
//             fprintf(stderr, "Unable to write on file '%s'!\n", outputFileName);
//             return EXIT_FAILURE;
//         }
//         if (!fwrite(attenuation + i * nSidePixels * nSidePixels, sizeof(double), nSidePixels * nSidePixels, outputFilePointer)) {
//             fprintf(stderr, "Unable to write on file '%s'!\n", outputFileName);
//             return EXIT_FAILURE;
//         }
//     }
// #else
    // Iterates over each attenuation value computed, prints a value between [0-255]
    fprintf(outputFilePointer, "P2\n%d %d\n255", nSidePixels, nSidePixels * (nTheta + 1));
    for (double positionIndex = 0; positionIndex <= nTheta; positionIndex++) {
        double angle = -gl_angularTrajectory / 2 + positionIndex * gl_positionsAngularDistance;
        fprintf(outputFilePointer, "\n#%lf", angle);
        for (int i = 0; i < nSidePixels; i++) {
            fprintf(outputFilePointer, "\n");
            for (int j = 0; j < nSidePixels; j++) {
                int pixelIndex = positionIndex * nSidePixels * nSidePixels + i * nSidePixels + j;
                int color = (attenuation[pixelIndex] - absMinValue) * 255 / (absMaxValue - absMinValue);
                fprintf(outputFilePointer, "%d ", color);
            }
        }
    }
// #endif

    free(grid);
    free(attenuation);

    fclose(inputFilePointer);
    fclose(outputFilePointer);
}
