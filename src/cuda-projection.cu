/**
 * @file cuda-projection.cu
 * @author Enrico Marchionni (enrico.marchionni@studio.unibo.it)
 * @brief A CUDA implementation of the Siddon's projection algorithm.
 * @date 2025-02
 * @details
 * This file contains an implementation of the projection algorithm
 * for generating 2D projections of a 3D object.
 * The algorithm is based on Siddon's algorithm and is parallelized
 * using CUDA.
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

/************************************************* DEVICE *************************************************/

// Inputs configuration limits
// 1024^3 (voxels cube size) * 8 (double size in Bytes) = 8 GiB
// max file size if the voxels are the same for each axis, and if the doubles are 8 Bytes
#define MAX_PLANES 1300u
#define MAX_PLANES_x2 (MAX_PLANES * 2)
#define MAX_PLANES_x3 (MAX_PLANES * 3)
// CUDA devices typically have 64 KB of constant memory
// This solution allocates 16KB of memory considering the 2 tables + other variables of small size
#define MAX_CONSTANT_MEMORY 8192u // 8 KB in Bytes
#define MAX_TABLES_SIZE (MAX_CONSTANT_MEMORY / sizeof(double)) // 8 KB in doubles
// Cuda limits
#define BLKDIM_STEP 16u // Max is usually 32, but for some GPUs the amount of registers is not enough in that case
#define BLKDIM (BLKDIM_STEP * BLKDIM_STEP)

__constant__ unsigned short d_pixelDim;
__constant__ unsigned short d_angularTrajectory;
__constant__ unsigned short d_positionsAngularDistance;
__constant__ unsigned short d_voxelXDim;
__constant__ unsigned short d_voxelYDim;
__constant__ unsigned short d_voxelZDim;

__constant__ unsigned d_objectSideLength;
__constant__ unsigned d_detectorSideLength;
__constant__ unsigned d_distanceObjectDetector;
__constant__ unsigned d_distanceObjectSource;

__constant__ unsigned short d_nVoxel[3];
__constant__ unsigned short d_nPlanes[3];

__constant__ double d_gl_sinTable[MAX_TABLES_SIZE];
__constant__ double d_gl_cosTable[MAX_TABLES_SIZE];

__device__ unsigned short d_yVoxels;

__device__ double d_gMin;
__device__ double d_gMax;

/************************************************* HOST *************************************************/

unsigned short yVoxels = 0;

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

double *d_f, *d_g;

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
 * @brief Computes the coordinate of a plane parallel relative to the YZ plane.
 *
 * @param index It is the index of the plane to be returned where 0 is the index of the smallest-valued coordinate plane.
 * @return The coordinate of a plane parallel relative to the YZ plane.
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ double getXPlane(const unsigned short index)
{
    return -(double) d_objectSideLength / 2 + (double) index * d_voxelXDim;
}

/**
 * @brief Computes the coordinate of a plane parallel relative to the XZ plane.
 *
 * @param index It is the index of the plane to be returned where 0 is the index of the smallest-valued coordinate plane.
 * @return The coordinate of a plane parallel relative to the XZ plane.
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ double getYPlane(const unsigned short index)
{
    return -(double) d_objectSideLength / 2 + (double) index * d_voxelYDim;
}

/**
 * @brief Computes the coordinate of a plane parallel relative to the XY plane.
 *
 * @param index It is the index of the plane to be returned where 0 is the index of the smallest-valued coordinate plane.
 * @return The coordinate of a plane parallel relative to the XY plane.
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ double getZPlane(const unsigned short index)
{
    return -(double) d_objectSideLength / 2 + (double) index * d_voxelZDim;
}

/**
 * @brief Computes the maximum parametric value a, representing the last intersection between ray and object.
 *
 * @param a It is the array containing the parametric value of the intersection between the ray and the object's side along each axis.
 * @param isParallel It is a value corresponding to the axis to which the array is orthogonal, -1 otherwise.
 * @return The maximum parametric value a, representing the last intersection between ray and object.
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ double getAMax(double a[3][2], const char isParallel)
{
    double tempMax[3];
    double aMax = 1;
    for (char i = 0; i < 3; i++) {
        if (i != isParallel) {
            tempMax[i] = a[i][0] > a[i][1] ? a[i][0] : a[i][1];
        }
    }
    for (char i = 0; i < 3; i++) {
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
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ double getAMin(double a[3][2], const char isParallel)
{
    double tempMin[3];
    double aMin = 0;
    for (char i = 0; i < 3; i++) {
        if (i != isParallel) {
            tempMin[i] = a[i][0] < a[i][1] ? a[i][0] : a[i][1];
        }
    }
    for (char i = 0; i < 3; i++) {
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
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ char getIntersection(const double source, const double pixel, const double *const planes, const unsigned short nPlanes, double *const a)
{
    if (source - pixel != 0) {
        for (unsigned short i = 0; i < nPlanes; i++) {
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
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ void getAllIntersections(const double source, const double pixel, const Ranges planeIndexesRanges, double *const a, const Axis axis)
{
    int start = 0, end = 0;
    double d;

    start = planeIndexesRanges.minIndx;
    end = planeIndexesRanges.maxIndx;
    if (end > start) { // Avoids management of invalid array
        assert(end - start <= MAX_PLANES);
        double plane[MAX_PLANES];
        if (axis == X) {
            plane[0] = getXPlane(start);
            d = d_voxelXDim;
            if (pixel - source < 0) {
                plane[0] = getXPlane(end);
                d = -(double) d_voxelXDim;
            }
        } else if (axis == Y) {
            plane[0] = getYPlane(start);
            d = d_voxelYDim;
            if (pixel - source < 0) {
                plane[0] = getYPlane(end);
                d = -(double) d_voxelYDim;
            }
        } else /* if (axis == Z) */ {
            plane[0] = getZPlane(start);
            d = d_voxelZDim;
            if (pixel - source < 0) {
                plane[0] = getZPlane(end);
                d = -(double) d_voxelZDim;
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
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ Ranges getRangeOfIndex(const double source, const double pixel, const char isParallel, const double aMin, const double aMax, const Axis axis)
{
    Ranges idxs;
    double firstPlane, lastPlane;
    unsigned short voxelDim;

    if (axis == X) {
        voxelDim = d_voxelXDim;
        firstPlane = getXPlane(0);
        lastPlane = getXPlane(d_nPlanes[X] - 1);
    } else if (axis == Y) {
        voxelDim = d_voxelYDim;
        firstPlane = getYPlane(0);
        lastPlane = getYPlane(d_nPlanes[Y] - 1);
    } else /* if (axis == Z) */ {
        voxelDim = d_voxelZDim;
        firstPlane = getZPlane(0);
        lastPlane = getZPlane(d_nPlanes[Z] - 1);
    }

    // Gets range of indexes of XZ parallel planes
    if (isParallel != Y) {
        if (pixel - source >= 0) {
            idxs.minIndx = d_nPlanes[axis] - ceil((lastPlane - aMin * (pixel - source) - source) / voxelDim);
            idxs.maxIndx = 1 + floor((aMax * (pixel - source) + source - firstPlane) / voxelDim);
        } else {
            idxs.minIndx = d_nPlanes[axis] - ceil((lastPlane - aMax * (pixel - source) - source) / voxelDim);
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
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ unsigned short merge(const double *const a, const double *const b, const unsigned short lenA, const unsigned short lenB, double *const c)
{
    unsigned short i = 0, j = 0, k = 0;
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
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ unsigned short merge3(const double *const a, const double *const b, const double *const c, const unsigned short lenA, const unsigned short lenB, const unsigned short lenC, double *const merged)
{
    assert(lenA + lenB + lenC <= MAX_PLANES_x3);
    double ab[MAX_PLANES_x3];
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
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ Point getSource(const double *const sinTable, const double *const cosTable, const unsigned short index)
{
    Point source;

    source.z = 0.0;
    source.x = sinTable[index] * d_distanceObjectSource;
    source.y = cosTable[index] * d_distanceObjectSource;

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
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ Point getPixel(const double *const sinTable, const double *const cosTable, const unsigned r, const unsigned c, const unsigned short index)
{
    Point pixel;
    const double sinAngle = sinTable[index];
    const double cosAngle = cosTable[index];
    const double elementOffset = d_detectorSideLength / 2 - d_pixelDim / 2;

    pixel.x = -(double) d_distanceObjectDetector * sinAngle + cosAngle * (-elementOffset + (double) d_pixelDim * c);
    pixel.y = -(double) d_distanceObjectDetector * cosAngle - sinAngle * (-elementOffset + (double) d_pixelDim * c);
    pixel.z = -elementOffset + (double) d_pixelDim * r;

    return pixel;
}

/**
 * @brief Computes a coordinate of the two planes of the object's sides orthogonal to the x axis.
 *
 * @param planes It is a pointer to an array of two elements,
 * each one of them is the coordinate of a plane parallel relative to the YZ plane.
 */
__device__ void getSidesXPlanes(double *const planes)
{
    planes[0] = getXPlane(0);
    planes[1] = getXPlane(d_nPlanes[X] - 1);
}

/**
 * @brief Computes a coordinate of the two planes of the object's sides orthogonal to the y axis.
 *
 * @param planes It is a pointer to an array of two elements,
 * each one of them is the coordinate of a plane parallel relative to the XZ plane.
 * @param slice It is a number that indicates the first voxel in the y axis from which the projection is being computed.
 * In this case this limits the planes considered.
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ void getSidesYPlanes(double *const planes, const unsigned short slice)
{
    planes[0] = getYPlane(slice);
    planes[1] = getYPlane(min(d_nPlanes[Y] - 1, d_yVoxels + slice));
}

/**
 * @brief Computes a coordinate of the two planes of the object's sides orthogonal to the z axis.
 *
 * @param planes It is a pointer to an array of two elements,
 * each one of them is the coordinate of a plane parallel relative to the XY plane.
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ void getSidesZPlanes(double *const planes)
{
    planes[0] = getZPlane(0);
    planes[1] = getZPlane(d_nPlanes[Z] - 1);
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
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ double computeAbsorption(const unsigned short slice, const Point source, const Point pixel, const double *const a, const unsigned short lenA, const double *const f)
{
    double g = 0.0;

    if (lenA > 0) { // Avoids overflow on unsigned value
        const double deltaX = pixel.x - source.x;
        const double deltaY = pixel.y - source.y;
        const double deltaZ = pixel.z - source.z;
        const double d12 = sqrt(pow(deltaX, 2) + pow(deltaY, 2) + pow(deltaZ, 2));
        for (unsigned short i = 0; i < lenA - 1; i++) {
            const double aMid = (a[i + 1] + a[i]) / 2;
            const unsigned short x = min((int) ((source.x + aMid * deltaX - getXPlane(0)) / d_voxelXDim), d_nVoxel[X] - 1);
            const unsigned short y = min((int) ((source.y + aMid * deltaY - getYPlane(slice)) / d_voxelYDim), min(d_nVoxel[Y] - 1, d_yVoxels - 1));
            const unsigned short z = min((int) ((source.z + aMid * deltaZ - getZPlane(0)) / d_voxelZDim), d_nVoxel[Z] - 1);

            // In a 3D matrix it would be: f[x][z][y]
            // d12 * (a[i + 1] - a[i] = segment length
            g += f[x + (unsigned) z*d_nVoxel[Z] + (unsigned) y*d_nVoxel[X]*d_nVoxel[Z]] * d12 * (a[i + 1] - a[i]);
        }
    }
    return g;
}

/**
 * @brief Atomically sets `addr` value to `value` if it is lower.
 *
 * @param addr The address that contains the value that could be set atomically.
 * @param value It is the value that could be exchanged atomically with `addr` value.
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ __forceinline__ double atomicMinDouble(double *const addr, const double value) {
    unsigned long long *addr_as_ull = (unsigned long long *) addr;
    unsigned long long old = *addr_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(fmin(__longlong_as_double(assumed), value)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

/**
 * @brief Atomically sets `addr` value to `value` if it is greater.
 *
 * @param addr The address that contains the value that could be set atomically.
 * @param value It is the value that could be exchanged atomically with `addr` value.
 * @return __device__ Indicates that this is a CUDA function that can be called from a kernel.
 */
__device__ __forceinline__ double atomicMaxDouble(double *const addr, const double value) {
    unsigned long long *addr_as_ull = (unsigned long long *) addr;
    unsigned long long old = *addr_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(fmax(__longlong_as_double(assumed), value)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

/**
 * @brief Computes the projection of a sub-section of the object into the detector for each source position on an NVIDIA GPU.
 *
 * @param slice It is a number that indicates the first voxel in the y axis from which the projection is being computed.
 * @param nTheta It is the number of angular positions.
 * @param nSidePixels It is the number of pixels per size of the detector.
 * @param f It is an array that contains the coefficients of attenuation of the voxels contained in the sub-section.
 * @param g It is the resulting array that contains the value of the computed projection attenuation for each pixel.
 * @param isFirst Tells if `g` array is uninitialized or it is not, this function initializes it if necessary.
 * @return __global__ Indicates that this is a CUDA kernel function, so it is executed on the device (GPU) and not the host (CPU).
 */
__global__ void computeProjections(const unsigned short slice, const unsigned short nTheta, const unsigned nSidePixels, const double *const f, double *const g, const char isFirst) {
    __shared__ double l_gMins[BLKDIM];
    __shared__ double l_gMaxs[BLKDIM];
    const unsigned l_index = threadIdx.x + threadIdx.y * blockDim.x;
    const unsigned r = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned c = threadIdx.x + blockIdx.x * blockDim.x;
    l_gMins[l_index] = INFINITY;
    l_gMaxs[l_index] = -INFINITY;

    if (r < nSidePixels && c < nSidePixels) {
        double a[3][2];

        if (isFirst) {
            for (unsigned short positionIndex = 0; positionIndex < nTheta; positionIndex++) {
                g[positionIndex * nSidePixels * nSidePixels + r * nSidePixels + c] = 0.0;
            }
        }
        // Iterates over each source
        for (unsigned short positionIndex = 0; positionIndex < nTheta; positionIndex++) {
            const Point source = getSource(d_gl_sinTable, d_gl_cosTable, positionIndex);

            // Computes the attenuation over a single pixel of the detector
            const Point pixel = getPixel(d_gl_sinTable, d_gl_cosTable, r, c, positionIndex);

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
                double aX[MAX_PLANES];
                double aY[MAX_PLANES];
                double aZ[MAX_PLANES];
                getAllIntersections(source.x, pixel.x, indices[X], aX, X);
                getAllIntersections(source.y, pixel.y, indices[Y], aY, Y);
                getAllIntersections(source.z, pixel.z, indices[Z], aZ, Z);

                // Computes segments Nx + Ny + Nz
                double aMerged[MAX_PLANES_x3];
                const unsigned short lenA = merge3(aX, aY, aZ, lenX, lenY, lenZ, aMerged);

                // Associates each segment to the respective voxel Nx + Ny + Nz
                const unsigned pixelIndex = positionIndex * nSidePixels * nSidePixels + r * nSidePixels + c;
                g[pixelIndex] += computeAbsorption(slice, source, pixel, aMerged, lenA, f);
                l_gMins[l_index] = fmin(l_gMins[l_index], g[pixelIndex]);
                l_gMaxs[l_index] = fmax(l_gMaxs[l_index], g[pixelIndex]);
            }
        }
    }

    unsigned b_size = blockDim.x / 2;
    __syncthreads();
    while (b_size > 0) {
        if (l_index < b_size) {
            if (l_gMins[l_index] > l_gMins[l_index + b_size]) {
                l_gMins[l_index] = l_gMins[l_index + b_size];
            }
            if (l_gMaxs[l_index] < l_gMaxs[l_index + b_size]) {
                l_gMaxs[l_index] = l_gMaxs[l_index + b_size];
            }
        }
        b_size = b_size / 2;
        __syncthreads();
    }

    if (l_index == 0) {
        atomicMinDouble(&d_gMin, l_gMins[l_index]);
        atomicMaxDouble(&d_gMax, l_gMaxs[l_index]);
    }
}

#ifdef DEBUG
static void printSizeMaxGB(const char *name, size_t size, const char* type="") {
    if (size > 1024) {
        double approximation = size / 1024.0;
        if (approximation > 1024.0) {
            approximation = approximation / 1024.0;
            if (approximation > 1024.0) {
                approximation = approximation / 1024.0;
                printf("%s %s = %.3lf GB\n", type, name, approximation);
            } else {
                printf("%s %s = %.2lf MB\n", type, name, approximation);
            }
        } else {
            printf("%s %s = %.1lf KB\n", type, name, approximation);
        }
    } else {
        printf("%s %s = %lu B\n", type, name, size);
    }
}
#endif

/**
 * @brief Releases allocated resources of the CUDA environment.
 */
void termEnvironment(void) {
    cudaSafeCall(cudaFree(d_f));
    cudaSafeCall(cudaFree(d_g));
    free(gl_sinTable);
    free(gl_cosTable);
}

/**
 * @brief Collets GPU computed data on the CPU and releases allocated resources of the CUDA environment.
 *
 * @param g It is the resulting array that contains the value of the computed projection attenuation for each pixel.
 * @param sizeG It is the size of the output array that contains the value of the computed projection attenuation for each pixel.
 * @param gMin It is the minimum attenuation computed.
 * @param gMax It is the maximum attenuation computed.
 */
void termEnvironment(double *g, size_t sizeG, double *gMin, double *gMax) {
    cudaSafeCall(cudaMemcpy(g, d_g, sizeG, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpyFromSymbol(gMin, d_gMin, sizeof(d_gMin)));
    cudaSafeCall(cudaMemcpyFromSymbol(gMax, d_gMax, sizeof(d_gMax)));
    termEnvironment();
}

/**
 * @brief Allocates resources in the CUDA environment.
 *
 * @param sizeF It is the size of the input array that contains the coefficients of attenuation of the voxels contained in the
 * sub-section.
 * @param sizeG It is the size of the output array that contains the value of the computed projection attenuation for each pixel.
 * @param nTheta It is the number of angular positions.
 * @param nSidePixels It is the number of pixels per size of the detector.
 * @param gMin It is the minimum attenuation computed.
 * @param gMax It is the maximum attenuation computed.
 */
void initEnvironment(size_t *sizeF, size_t sizeG, const unsigned short nTheta, const unsigned nSidePixels, double *gMin, double *gMax) {
#ifdef DEBUG
    printf("CONFIG (threads and blocks):\n");
    const unsigned short tmp = (nSidePixels + BLKDIM_STEP - 1) / BLKDIM_STEP;
    printf("%s = %dx%d\n", "2D grid", tmp, tmp);
    printf("%s = %dx%d\n\n", "2D block", BLKDIM_STEP, BLKDIM_STEP);
    printf("%s = %d\n", "N blocks", tmp * tmp);
    printf("%s = %d\n\n", "N threads", BLKDIM_STEP * BLKDIM_STEP);
#endif
    cudaSafeCall(cudaMemcpyToSymbol(d_pixelDim, &gl_pixelDim, sizeof(d_pixelDim)));
    cudaSafeCall(cudaMemcpyToSymbol(d_angularTrajectory, &gl_angularTrajectory, sizeof(d_angularTrajectory)));
    cudaSafeCall(cudaMemcpyToSymbol(d_positionsAngularDistance, &gl_positionsAngularDistance, sizeof(d_positionsAngularDistance)));
    cudaSafeCall(cudaMemcpyToSymbol(d_objectSideLength, &gl_objectSideLength, sizeof(d_objectSideLength)));
    cudaSafeCall(cudaMemcpyToSymbol(d_detectorSideLength, &gl_detectorSideLength, sizeof(d_detectorSideLength)));
    cudaSafeCall(cudaMemcpyToSymbol(d_distanceObjectDetector, &gl_distanceObjectDetector, sizeof(d_distanceObjectDetector)));
    cudaSafeCall(cudaMemcpyToSymbol(d_distanceObjectSource, &gl_distanceObjectSource, sizeof(d_distanceObjectSource)));
    cudaSafeCall(cudaMemcpyToSymbol(d_voxelXDim, &gl_voxelXDim, sizeof(d_voxelXDim)));
    cudaSafeCall(cudaMemcpyToSymbol(d_voxelYDim, &gl_voxelYDim, sizeof(d_voxelYDim)));
    cudaSafeCall(cudaMemcpyToSymbol(d_voxelZDim, &gl_voxelZDim, sizeof(d_voxelZDim)));
    cudaSafeCall(cudaMemcpyToSymbol(d_nVoxel, gl_nVoxel, sizeof(d_nVoxel)));
    cudaSafeCall(cudaMemcpyToSymbol(d_nPlanes, gl_nPlanes, sizeof(d_nPlanes)));

    gl_sinTable = (double *) malloc(sizeof(double) * nTheta);
    gl_cosTable = (double *) malloc(sizeof(double) * nTheta);
    initTables(gl_sinTable, gl_cosTable, nTheta);
    cudaSafeCall(cudaMemcpyToSymbol(d_gl_sinTable, gl_sinTable, sizeof(d_gl_sinTable)));
    cudaSafeCall(cudaMemcpyToSymbol(d_gl_cosTable, gl_cosTable, sizeof(d_gl_cosTable)));

    cudaSafeCall(cudaMemcpyToSymbol(d_gMin, gMin, sizeof(d_gMin)));
    cudaSafeCall(cudaMemcpyToSymbol(d_gMax, gMax, sizeof(d_gMax)));

    cudaSafeCall(cudaMalloc((void **) &d_g, sizeG));

    if (!yVoxels) {
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        // At least 1 GB of estimated free global memory are necessary for the kernel execution in a 8 GB RAM GPU
        freeMem -= (totalMem * 2 / 8);
        unsigned voxelsY = gl_nVoxel[Y];
        size_t size = sizeof(double) * gl_nVoxel[X] * voxelsY * gl_nVoxel[Z];
        while (size > freeMem) {
            // 5 / 8 is around 5 GB if considering an 8 GB input size
            voxelsY = voxelsY * 5 / 8;
            if (voxelsY <= 0) {
                fprintf(stderr, "The voxels Y size is too small respect to the other sizes:\n"
                                "- N voxels X: %u.\n"
                                "- N voxels Y: %u.\n"
                                "- N voxels Z: %u.\n"
                                "Total size reduced to the minimum possible is %lu Bytes!\n"
                                "This is too much for this GPU with %lu Bytes of usable global memory (of %lu Bytes total)!\n",
                                gl_nVoxel[X], gl_nVoxel[Y], gl_nVoxel[Z], size, freeMem, totalMem);
                termEnvironment();
                exit(EXIT_FAILURE);
            }
            size = sizeof(double) * gl_nVoxel[X] * voxelsY * gl_nVoxel[Z];
        }
        yVoxels = voxelsY;
        *sizeF = size;
    } else {
        *sizeF = sizeof(double) * gl_nVoxel[X] * yVoxels * gl_nVoxel[Z];
    }

    cudaSafeCall(cudaMalloc((void **) &d_f, *sizeF));
    cudaSafeCall(cudaMemcpyToSymbol(d_yVoxels, &yVoxels, sizeof(d_yVoxels)));
#ifdef DEBUG
    printSizeMaxGB("f", *sizeF, "GLOBAL");
    printSizeMaxGB("g", sizeG, "GLOBAL");
    printf("\n");
#endif
}

/**
 * @brief Computes the projection of a sub-section of the object into the detector for each source position.
 *
 * @param slice It is a number that indicates the first voxel in the y axis from which the projection is being computed.
 * @param f It is an array that contains the coefficients of attenuation of the voxels contained in the sub-section.
 * @param sizeF It is the size of the `f`.
 * @param nTheta It is the number of angular positions.
 * @param nSidePixels It is the number of pixels per size of the detector.
 * @param isFirst Tells if `g` array is uninitialized or it is not, this function tells to initialize it if necessary.
 */
void getProjections(const unsigned short slice, double *f, const size_t sizeF, const unsigned short nTheta, const unsigned nSidePixels, const char isFirst)
{
#ifdef PRINT
    printf("%.4lf> Copying f...\n", hpc_gettime());
#endif
    cudaSafeCall(cudaMemcpy(d_f, f, sizeF, cudaMemcpyHostToDevice));
    static dim3 block(BLKDIM_STEP, BLKDIM_STEP);
    static dim3 grid((nSidePixels + BLKDIM_STEP - 1) / BLKDIM_STEP, (nSidePixels + BLKDIM_STEP - 1) / BLKDIM_STEP);
#ifdef PRINT
    printf("%.4lf> Executing kernel...\n", hpc_gettime());
#endif
    computeProjections<<<grid, block>>>(slice, nTheta, nSidePixels, d_f, d_g, isFirst);
    cudaCheckError();
}

/**
 * @brief Reads the environment values used to compute the voxel grid from the specified binary file.
 *
 * @param filePointer It is the file pointer to read the values from.
 * @return EXIT_FAILURE in case of reading failure, EXIT_SUCCESS otherwise.
 */
int readSetUP(FILE *const filePointer)
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
    if (gl_nPlanes[X] > MAX_PLANES || gl_nPlanes[Y] > MAX_PLANES || gl_nPlanes[Z] > MAX_PLANES) {
        fprintf(stderr, "The maximum number of planes per axis is %u planes!\n", MAX_PLANES);
        return EXIT_FAILURE;
    }
    if (argc > 3) {
        const int yMaxVoxels = atoi(argv[3]);
        yVoxels = min(gl_nVoxel[Y], max(yVoxels, 1));
    }
    // Number of angular positions
    double partialTime = hpc_gettime();
    const unsigned short nTheta = gl_angularTrajectory / gl_positionsAngularDistance + 1;
    if (nTheta > MAX_TABLES_SIZE) {
        fprintf(stderr, "Number of positions required %u is too large, max %lu!\n", nTheta, MAX_TABLES_SIZE);
        exit(EXIT_FAILURE);
    }
    const unsigned nSidePixels = gl_detectorSideLength / gl_pixelDim;
    // Size of the array containing the computed attenuation detected in each pixel of the detector
    const size_t sizeG = sizeof(double) * nSidePixels * nSidePixels * nTheta;
    // Minimum and maximum attenuation computed
    double gMinValue = INFINITY, gMaxValue = -INFINITY;
    size_t sizeF;
    initEnvironment(&sizeF, sizeG, nTheta, nSidePixels, &gMinValue, &gMaxValue);
    // Array containing the coefficients of each voxel
    double *const f = (double *) malloc(sizeF);
    double totalTime = hpc_gettime() - partialTime;

    // Iterates over object subsections
    for (unsigned short slice = 0; slice < gl_nVoxel[Y]; slice += yVoxels) {
        unsigned short nOfSlices;

        if (gl_nVoxel[Y] - slice < yVoxels) {
            nOfSlices = gl_nVoxel[Y] - slice;
        } else {
            nOfSlices = yVoxels;
        }

        // Read voxels coefficients
#ifdef PRINT
        printf("%.4lf> Reading f...\n", hpc_gettime());
#endif
        if (!fread(f, sizeof(double), (size_t) gl_nVoxel[X] * nOfSlices * gl_nVoxel[Z], inputFilePointer)) {
            fprintf(stderr, "Unable to read from file '%s'!\n", inputFileName);
            free(f);
            termEnvironment();
            return EXIT_FAILURE;
        }

#ifdef PRINT
        static unsigned short it = 0;
        printf("%.4lf> IT %u of size %hu\n", hpc_gettime(), ++it, nOfSlices);
#endif
        // Computes subsection projection
        partialTime = hpc_gettime();
        getProjections(slice, f, sizeF, nTheta, nSidePixels, !slice);
        totalTime += hpc_gettime() - partialTime;
    }
    fclose(inputFilePointer);
    partialTime = hpc_gettime();
    free(f);
    // Array containing the computed attenuation detected in each pixel of the detector
    double *const g = (double *) malloc(sizeG);
    termEnvironment(g, sizeG, &gMinValue, &gMaxValue);
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
            const double angle = -(double) gl_angularTrajectory / 2 + (double) positionIndex * gl_positionsAngularDistance;
            fprintf(outputFilePointer, "\n#%lf", angle);
            for (unsigned i = 0; i < nSidePixels; i++) {
                fprintf(outputFilePointer, "\n");
                for (unsigned j = 0; j < nSidePixels; j++) {
                    const unsigned pixelIndex = positionIndex * nSidePixels * nSidePixels + i * nSidePixels + j;
                    const int color = (g[pixelIndex] - gMinValue) * 255 / (gMaxValue - gMinValue);
                    fprintf(outputFilePointer, "%d ", color);
                }
            }
        }
        fclose(outputFilePointer);
    }
    free(g);

    return EXIT_SUCCESS;
}
