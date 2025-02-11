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

unsigned short yVoxels = 100;

unsigned short gl_pixelDim;
unsigned short gl_angularTrajectory;
unsigned short gl_positionsAngularDistance;
unsigned short gl_voxelXDim;
unsigned short gl_voxelYDim;
unsigned short gl_voxelZDim;
unsigned short gl_nVoxel[3];
unsigned short gl_nPlanes[3];

unsigned gl_objectSideLength;
unsigned gl_detectorSideLength;
unsigned gl_distanceObjectDetector;
unsigned gl_distanceObjectSource;

double *gl_sinTable, *gl_cosTable;

/**
 * @brief Initializes `sin` and `cos` tables, with default values for a certain length.
 *
 * @param sinTable An array containing a certain number of sin values to precalculate.
 * @param cosTable An array containing a certain number of cos values to precalculate.
 * @param length The length of the arrays.
 */
void initTables(double *const sinTable, double *const cosTable, const unsigned short length)
{
    for (unsigned short positionIndex = 0; positionIndex < length; positionIndex++) {
        sinTable[positionIndex] = sin((-(double) gl_angularTrajectory / 2 + (double) positionIndex * gl_positionsAngularDistance) * M_PI / 180);
        cosTable[positionIndex] = cos((-(double) gl_angularTrajectory / 2 + (double) positionIndex * gl_positionsAngularDistance) * M_PI / 180);
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
 * @brief Computes the maximum value between `a` and `b`.
 *
 * @param a The first value.
 * @param b The second value.
 * @return The maximum between `a` and `b`.
 */
int max(int a, int b)
{
    return a > b ? a : b;
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
double getXPlane(const unsigned short index)
{
    return -(double) gl_objectSideLength / 2 + (double) index * gl_voxelXDim;
}

/**
 * @brief Computes the coordinate of a plane parallel relative to the XZ plane.
 *
 * @param index It is the index of the plane to be returned where 0 is the index of the smallest-valued coordinate plane.
 * @return The coordinate of a plane parallel relative to the XZ plane.
 */
double getYPlane(const unsigned short index)
{
    return -(double) gl_objectSideLength / 2 + (double) index * gl_voxelYDim;
}

/**
 * @brief Computes the coordinate of a plane parallel relative to the XY plane.
 *
 * @param index It is the index of the plane to be returned where 0 is the index of the smallest-valued coordinate plane.
 * @return The coordinate of a plane parallel relative to the XY plane.
 */
double getZPlane(const unsigned short index)
{
    return -(double) gl_objectSideLength / 2 + (double) index * gl_voxelZDim;
}

/**
 * @brief Computes the maximum parametric value a, representing the last intersection between ray and object.
 *
 * @param a It is the array containing the parametric value of the intersection between the ray and the object's side along each axis.
 * @param isParallel It is a value corresponding to the axis to which the array is orthogonal, -1 otherwise.
 * @return The maximum parametric value a, representing the last intersection between ray and object.
 */
double getAMax(double a[3][2], const char isParallel)
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
 * @return The minimum parametric value a, representing the first intersection between ray and object.
 */
double getAMin(double a[3][2], const char isParallel)
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
 * @param planes It is an array that contains the coordinates of each plane.
 * @param nPlanes Specifies the number of planes.
 * @param a It is an array that will be filled with the parametric values that identify the intersection points between the
 * ray and each plane.
 * @return 0 if ray is parallel to the planes, 1 otherwise.
 */
int getIntersection(const double source, const double pixel, const double *const planes, const unsigned short nPlanes, double *const a)
{
    if (source - pixel != 0) {
        for (int i = 0; i < nPlanes; i++) {
            a[i] = (planes[i] - source) / (pixel - source);
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
 * @param planeIndexesRanges It is a structure containing the index ranges for planes.
 * @param a It is an array that will be filled with the parametric values that identify the intersection points between the
 * ray and each plane.
 * @param axis It is the axis orthogonal to the set of planes to which compute the intersection.
 */
void getAllIntersections(const double source, const double pixel, const Ranges planeIndexesRanges, double *const a, const Axis axis)
{
    int start = 0, end = 0;
    double d;

    start = planeIndexesRanges.minIndx;
    end = planeIndexesRanges.maxIndx;
    if (end > start) { // Avoids management of invalid array
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

        for (unsigned short i = 1; i < end - start; i++) {
            plane[i] = plane[i - 1] + d;
        }
        getIntersection(source, pixel, plane, end - start, a);
    }
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
Ranges getRangeOfIndex(const double source, const double pixel, const char isParallel, const double aMin, const double aMax, const Axis axis)
{
    Ranges idxs;
    double firstPlane, lastPlane;
    unsigned short voxelDim;

    if (axis == X) {
        voxelDim = gl_voxelXDim;
        firstPlane = getXPlane(0);
        lastPlane = getXPlane(gl_nPlanes[X] - 1);
    } else if (axis == Y) {
        voxelDim = gl_voxelYDim;
        firstPlane = getYPlane(0);
        lastPlane = getYPlane(gl_nPlanes[Y] - 1);
    } else /* if (axis == Z) */ {
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
int merge(const double *const a, const double *const b, const unsigned short lenA, const unsigned short lenB, double *const c)
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
unsigned short merge3(const double *const a, const double *const b, const double *const c, const unsigned short lenA, const unsigned short lenB, const unsigned short lenC, double *const merged)
{
    double ab[lenA + lenB];
    const unsigned short length = merge(a, b, lenA, lenB, ab);
    return merge(ab, c, length, lenC, merged);
}

/**
 * @brief Retrieves the cartesian coordinates of the source.
 *
 * @param sinTable An array containing a certain number of precalculated sin values.
 * @param cosTable An array containing a certain number of precalculated cos values.
 * @param index A value that defines the angle being considered.
 * @return The coordinates of the source.
 */
Point getSource(const double *const sinTable, const double *const cosTable, const unsigned short index)
{
    Point source;

    source.z = 0;
    source.x = sinTable[index] * gl_distanceObjectSource;
    source.y = cosTable[index] * gl_distanceObjectSource;

    return source;
}

/**
 * @brief Retrieves the cartesian coordinates of a unit of the detector.
 *
 * @param sinTable An array containing a certain number of precalculated sin values.
 * @param cosTable An array containing a certain number of precalculated cos values.
 * @param r The row of the detector matrix.
 * @param c The column of the detector matrix.
 * @param index A value that defines the angle being considered, and consequently, the source.
 * @return The coordinates of a unit of the detector, relative to the specified source.
 */
Point getPixel(const double *const sinTable, const double *const cosTable, const unsigned r, const unsigned c, const unsigned short index)
{
    Point pixel;
    const double sinAngle = sinTable[index];
    const double cosAngle = cosTable[index];
    const double elementOffset =  gl_detectorSideLength / 2 - gl_pixelDim / 2;

    pixel.x = -(double) gl_distanceObjectDetector * sinAngle + cosAngle * (-elementOffset + (double) gl_pixelDim * c);
    pixel.y = -(double) gl_distanceObjectDetector * cosAngle - sinAngle * (-elementOffset + (double) gl_pixelDim * c);
    pixel.z = -elementOffset + (double) gl_pixelDim * r;

    return pixel;
}

/**
 * @brief Computes a coordinate of the two planes of the object's sides orthogonal to the x axis.
 *
 * @param planes It is a pointer to an array of two elements,
 * each one of them is the coordinate of a plane parallel relative to the YZ plane.
 */
void getSidesXPlanes(double *const planes)
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
void getSidesYPlanes(double *const planes, const unsigned short slice)
{
    planes[0] = getYPlane(slice);
    planes[1] = getYPlane(min(gl_nPlanes[Y] - 1, yVoxels + slice));
}

/**
 * @brief Computes a coordinate of the two planes of the object's sides orthogonal to the z axis.
 *
 * @param planes It is a pointer to an array of two elements,
 * each one of them is the coordinate of a plane parallel relative to the XY plane.
 */
void getSidesZPlanes(double *const planes)
{
    planes[0] = getZPlane(0);
    planes[1] = getZPlane(gl_nPlanes[Z] - 1);
}

/**
 * @brief Computes the projection attenuation of the radiological path of a ray.
 *
 * @param slice It is a number that indicates the first voxel in the y axis from which the projection is being computed.
 * @param source Represents the coordinate of the source.
 * @param pixel Represents the coordinate of the unit of the detector.
 * @param a It is an array that contains all intersection points merged, expressed parametrically.
 * @param lenA It is the length of the corresponding array.
 * @param f It is an array of the coefficients of attenuation for each voxel.
 * @return The computed projection attenuation of the radiological path of a ray.
 */
double computeAbsorption(const unsigned short slice, const Point source, const Point pixel, const double *const a, const unsigned short lenA, const double *const f)
{
    double g = 0.0;

    if (lenA > 0) { // Avoids overflow on unsigned value
        const double deltaX = pixel.x - source.x;
        const double deltaY = pixel.y - source.y;
        const double deltaZ = pixel.z - source.z;
        const double d12 = sqrt(pow(deltaX, 2) + pow(deltaY, 2) + pow(deltaZ, 2));
        for (unsigned short i = 0; i < lenA - 1; i++) {
            const double aMid = (a[i + 1] + a[i]) / 2;
            const unsigned short x = min((source.x + aMid * deltaX - getXPlane(0)) / gl_voxelXDim, gl_nVoxel[X] - 1);
            const unsigned short y = min3((source.y + aMid * deltaY - getYPlane(slice)) / gl_voxelYDim, gl_nVoxel[Y] - 1, yVoxels - 1);
            const unsigned short z = min((source.z + aMid * deltaZ - getZPlane(0)) / gl_voxelZDim, gl_nVoxel[Z] - 1);

            // In a 3D matrix it would be: f[x][z][y]
            // d12 * (a[i + 1] - a[i] = segment length
            g += f[x + (unsigned) z*gl_nVoxel[Z] + (unsigned) y*gl_nVoxel[X]*gl_nVoxel[Z]] * d12 * (a[i + 1] - a[i]);
        }
    }
    return g;
}

/**
 * @brief Computes the projection of a sub-section of the object into the detector for each source position.
 *
 * @param slice It is a number that indicates the first voxel in the y axis from which the projection is being computed.
 * @param f It is an array that contains the coefficients of attenuation of the voxels contained in the sub-section.
 * @param g It is the resulting array that contains the value of the computed projection attenuation for each pixel.
 * @param gMin It is the minimum projection attenuation computed.
 * @param gMax It is the maximum projection attenuation computed.
 * @param nTheta It is the number of angular positions.
 * @param nSidePixels It is the number of pixels per size of the detector.
 */
void computeProjections(const unsigned short slice, double *f, double *g, double *gMin, double *gMax, const unsigned short nTheta, const unsigned nSidePixels)
{
    double l_gMin = INFINITY;
    double l_gMax = -INFINITY;

    // Iterates over each source
    for (unsigned short positionIndex = 0; positionIndex < nTheta; positionIndex++) {
        const Point source = getSource(gl_sinTable, gl_cosTable, positionIndex);

        // Iterates over each pixel of the detector
#pragma omp parallel for collapse(2) schedule(dynamic) default(none) shared(gl_sinTable, gl_cosTable, nSidePixels, positionIndex, source, slice, f, g, nTheta, gl_nVoxel, gl_nPlanes) reduction(min:l_gMin) reduction(max:l_gMax)
        for (unsigned r = 0; r < nSidePixels; r++) {
            for (unsigned c = 0; c < nSidePixels; c++) {
                double a[3][2];

                // Gets the pixel's center cartesian coordinates
                const Point pixel = getPixel(gl_sinTable, gl_cosTable, r, c, positionIndex);

                // Computes Min-Max parametric values
                double aMin, aMax;
                double sidesPlanes[2];
                char isParallel = -1;
                getSidesXPlanes(sidesPlanes);
                if (!getIntersection(source.x, pixel.x, sidesPlanes, 2, &a[X][0])) {
                    isParallel = X;
                }
                getSidesYPlanes(sidesPlanes, slice);
                if (!getIntersection(source.y, pixel.y, sidesPlanes, 2, &a[Y][0])) {
                    isParallel = Y;
                }
                getSidesZPlanes(sidesPlanes);
                if (!getIntersection(source.z, pixel.z, sidesPlanes, 2, &a[Z][0])) {
                    isParallel = Z;
                }

                aMin = getAMin(a, isParallel);
                aMax = getAMax(a, isParallel);

                if (aMin < aMax) {
                    // Computes Min-Max plane indexes
                    Ranges indices[3];
                    indices[X] = getRangeOfIndex(source.x, pixel.x, isParallel, aMin, aMax, X);
                    indices[Y] = getRangeOfIndex(source.y, pixel.y, isParallel, aMin, aMax, Y);
                    indices[Z] = getRangeOfIndex(source.z, pixel.z, isParallel, aMin, aMax, Z);

                    // Computes lengths of the arrays containing parametric value of the intersection with each set of parallel planes
                    const unsigned short lenX = max(0, indices[X].maxIndx - indices[X].minIndx);
                    const unsigned short lenY = max(0, indices[Y].maxIndx - indices[Y].minIndx);
                    const unsigned short lenZ = max(0, indices[Z].maxIndx - indices[Z].minIndx);

                    // Computes ray-planes intersection Nx + Ny + Nz
                    double aX[gl_nPlanes[X]];
                    double aY[gl_nPlanes[Y]];
                    double aZ[gl_nPlanes[Z]];
                    getAllIntersections(source.x, pixel.x, indices[X], aX, X);
                    getAllIntersections(source.y, pixel.y, indices[Y], aY, Y);
                    getAllIntersections(source.z, pixel.z, indices[Z], aZ, Z);

                    // Computes segments Nx + Ny + Nz
                    double aMerged[gl_nPlanes[X] + gl_nPlanes[Y] + gl_nPlanes[Z]];
                    const unsigned short lenA = merge3(aX, aY, aZ, lenX, lenY, lenZ, aMerged);

                    // Associates each segment to the respective voxel Nx + Ny + Nz
                    const unsigned pixelIndex = positionIndex * nSidePixels * nSidePixels + r * nSidePixels + c;
                    g[pixelIndex] += computeAbsorption(slice, source, pixel, aMerged, lenA, f);
                    l_gMax = fmax(l_gMax, g[pixelIndex]);
                    l_gMin = fmin(l_gMin, g[pixelIndex]);
                }
            }
        }
    }
    *gMax = l_gMax;
    *gMin = l_gMin;
}

/**
 * @brief Releases allocated resources of the OpenMP environment.
 */
void termEnvironment(void) {
    free(gl_sinTable);
    free(gl_cosTable);
}

/**
 * @brief Allocates resources necessary for the OpenMP environment.
 *
 * @param nTheta It is the number of angular positions.
 */
void initEnvironment(const unsigned short nTheta) {
    gl_sinTable = (double *) malloc(sizeof(double) * nTheta);
    gl_cosTable = (double *) malloc(sizeof(double) * nTheta);
    initTables(gl_sinTable, gl_cosTable, nTheta);
}

/**
 * @brief Reads the environment values used to compute the voxel grid from the specified binary file.
 *
 * @param filePointer It is the file pointer to read the values from.
 * @return EXIT_FAILURE in case of reading failure, EXIT_SUCCESS otherwise.
 */
int readSetUP(FILE *filePointer)
{
    unsigned short buffer0[12];
    if (!fread(buffer0, sizeof(unsigned short), sizeof(buffer0) / sizeof(unsigned short), filePointer)) {
        return EXIT_FAILURE;
    }

    unsigned char i = 0;
    gl_pixelDim = buffer0[i++];
    gl_angularTrajectory = buffer0[i++];
    gl_positionsAngularDistance = buffer0[i++];
    gl_voxelXDim = buffer0[i++];
    gl_voxelYDim = buffer0[i++];
    gl_voxelZDim = buffer0[i++];
    gl_nVoxel[X] = buffer0[i++];
    gl_nVoxel[Y] = buffer0[i++];
    gl_nVoxel[Z] = buffer0[i++];
    gl_nPlanes[X] = buffer0[i++];
    gl_nPlanes[Y] = buffer0[i++];
    gl_nPlanes[Z] = buffer0[i];

    unsigned buffer1[4];
    if (!fread(buffer1, sizeof(unsigned), sizeof(buffer1) / sizeof(unsigned), filePointer)) {
        return EXIT_FAILURE;
    }

    i = 0;
    gl_objectSideLength = buffer1[i++];
    gl_detectorSideLength = buffer1[i++];
    gl_distanceObjectDetector = buffer1[i++];
    gl_distanceObjectSource = buffer1[i];

#ifdef PRINT_VARIABLES
    printf("Variables READ:\n");
    printf("- unsigned short:\n");
    printf("    gl_pixelDim = %hu\n", gl_pixelDim);
    printf("    gl_angularTrajectory = %hu\n", gl_angularTrajectory);
    printf("    gl_positionsAngularDistance = %hu\n", gl_positionsAngularDistance);
    printf("    gl_voxelXDim = %hu\n", gl_voxelXDim);
    printf("    gl_voxelYDim = %hu\n", gl_voxelYDim);
    printf("    gl_voxelZDim = %hu\n", gl_voxelZDim);
    printf("    gl_nVoxel[X] = %hu\n", gl_nVoxel[X]);
    printf("    gl_nVoxel[Y] = %hu\n", gl_nVoxel[Y]);
    printf("    gl_nVoxel[Z] = %hu\n", gl_nVoxel[Z]);
    printf("    gl_nPlanes[X] = %hu\n", gl_nPlanes[X]);
    printf("    gl_nPlanes[Y] = %hu\n", gl_nPlanes[Y]);
    printf("    gl_nPlanes[Z] = %hu\n", gl_nPlanes[Z]);
    printf("- unsigned:\n");
    printf("    gl_objectSideLength = %u\n", gl_objectSideLength);
    printf("    gl_detectorSideLength = %u\n", gl_detectorSideLength);
    printf("    gl_distanceObjectDetector = %u\n", gl_distanceObjectDetector);
    printf("    gl_distanceObjectSource = %u\n", gl_distanceObjectSource);
#endif

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "Usage: %s INPUT [OUTPUT] [Y_MAX_VOXELS]\n"
                        "- INPUT: The first parameter is the name of the input file.\n"
                        "- [OUTPUT]: The second parameter is the name of a .pgm file to store the output at.\n"
                        "- [Y_MAX_VOXELS]: The third parameter is the maximum number of voxels considered in the Y axis for each iteration.\n",
                        argv[0]);
        return EXIT_FAILURE;
    }
    const char *const inputFileName = argv[1];
    const char *outputFileName = NULL;
    if (argc > 2) {
        outputFileName = argv[2];
    }

    FILE *const inputFilePointer = fopen(inputFileName, "rb");
    if (!inputFilePointer) {
        fprintf(stderr, "Unable to open file '%s'!\n", inputFileName);
        return EXIT_FAILURE;
    }

    if (readSetUP(inputFilePointer) == EXIT_FAILURE) {
        fprintf(stderr, "Unable to read from file '%s'!\n", inputFileName);
        return EXIT_FAILURE;
    }
    if (argc > 3) {
        const int yMaxVoxels = atoi(argv[3]);
        yVoxels = min(gl_nVoxel[Y], max(yMaxVoxels, 1));
    } else {
        yVoxels = min(gl_nVoxel[Y], yVoxels);
    }
    double partialTime = hpc_gettime();
    // Number of angular positions
    const unsigned short nTheta = gl_angularTrajectory / gl_positionsAngularDistance + 1;
    const unsigned nSidePixels = gl_detectorSideLength / gl_pixelDim;
    initEnvironment(nTheta);
    // Array containing the coefficients of each voxel
    double *const f = (double *) malloc(sizeof(double) * gl_nVoxel[X] * yVoxels * gl_nVoxel[Z]);
    // Array containing the computed attenuation detected in each pixel of the detector
    double *const g = (double *) calloc(nSidePixels * nSidePixels * nTheta, sizeof(double));
    // double *const g = (double *) malloc(sizeof(double) * nSidePixels * nSidePixels * nTheta);
    // Minimum and maximum attenuation computed
    double gMinValue = INFINITY, gMaxValue = -INFINITY;
    double totalTime = hpc_gettime() - partialTime;

    // Iterates over object subsection
    for (unsigned short slice = 0; slice < gl_nVoxel[Y]; slice += yVoxels) {
        unsigned short nOfSlices;

        if (gl_nVoxel[Y] - slice < yVoxels) {
            nOfSlices = gl_nVoxel[Y] - slice;
        } else {
            nOfSlices = yVoxels;
        }

        // Read voxels coefficients
        if (!fread(f, sizeof(double), (size_t) gl_nVoxel[X] * nOfSlices * gl_nVoxel[Z], inputFilePointer)) {
            fprintf(stderr, "Unable to read from file '%s'!\n", inputFileName);
            free(f);
            free(g);
            return EXIT_FAILURE;
        }

#ifdef PRINT
        static unsigned short it = 0;
        printf("IT %u of size %hu\n", ++it, nOfSlices);
#endif
        // Computes subsection projection
        partialTime = hpc_gettime();
        computeProjections(slice, f, g, &gMinValue, &gMaxValue, nTheta, nSidePixels);
        totalTime += hpc_gettime() - partialTime;
    }
    fclose(inputFilePointer);
    partialTime = hpc_gettime();
    free(f);
    termEnvironment();
    totalTime += hpc_gettime() - partialTime;
    printf("Execution time (s) %.2f\n", totalTime);

    if (outputFileName != NULL) {
        FILE *const outputFilePointer = fopen(outputFileName, "w");
        if (!outputFileName) {
            fprintf(stderr, "Unable to open file '%s'!\n", outputFileName);
            free(g);
            return EXIT_FAILURE;
        }
        // Iterates over each attenuation value computed, prints a value between [0-255]
        fprintf(outputFilePointer, "P2\n%d %d\n255", nSidePixels, nSidePixels * nTheta);
        for (unsigned short positionIndex = 0; positionIndex < nTheta; positionIndex++) {
            double angle = -(double) gl_angularTrajectory / 2 + (double) positionIndex * gl_positionsAngularDistance;
            fprintf(outputFilePointer, "\n#%lf", angle);
            for (unsigned i = 0; i < nSidePixels; i++) {
                fprintf(outputFilePointer, "\n");
                for (unsigned j = 0; j < nSidePixels; j++) {
                    const unsigned pixelIndex = positionIndex * nSidePixels * nSidePixels + i * nSidePixels + j;
                    int color = (g[pixelIndex] - gMinValue) * 255 / (gMaxValue - gMinValue);
                    fprintf(outputFilePointer, "%d ", color);
                }
            }
        }
        fclose(outputFilePointer);
    }
    free(g);

    return EXIT_SUCCESS;
}
