/**
 * @file common.h
 * @author Enrico Marchionni (enrico.marchionni@studio.unibo.it)
 * @brief Configures common data that can be used for input generation
 * and structures useful also for projection implementations.
 * @date 2025-01
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

#ifndef COMMON_H
#define COMMON_H

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

/*
 * The following constants are input parameters values to be considered as a reference.
 */
#define VOXEL_X_DIM 100 // voxel side length along x-axis
#define VOXEL_Y_DIM 100 // voxel side length along y-axis
#define VOXEL_Z_DIM 100 // voxel side length along z-axis

#define PIXEL_DIM 85 // side length of a pixel of the detector
#define ANGULAR_TRAJECTORY 90 // total angular distance traveled by the source
#define POSITIONS_ANGULAR_DISTANCE 15 // angular distance between each source position

#define OBJECT_SIDE_LENGTH 100000u // side length of the object
#define DETECTOR_SIDE_LENGTH 200000u // side length of the detector
#define DISTANCE_OBJECT_DETECTOR 150000u // distance between the object's center and the detector
#define DISTANCE_OBJECT_SOURCE 600000u // distance between the object's center and the source position

/*
 * The following constraints represent the reference value for gl_nVoxel and gl_nPlanes variables.
 */
#define N_VOXEL_X  (OBJECT_SIDE_LENGTH / VOXEL_X_DIM) // number of voxel the object is composed of along the X axis
#define N_VOXEL_Y  (OBJECT_SIDE_LENGTH / VOXEL_Y_DIM) // number of voxel the object is composed of along the Y axis
#define N_VOXEL_Z  (OBJECT_SIDE_LENGTH / VOXEL_Z_DIM) // number of voxel the object is composed of along the Z axis
#define N_PLANES_X (N_VOXEL_X + 1) // number of planes along the X axis
#define N_PLANES_Y (N_VOXEL_Y + 1) // number of planes along the Y axis
#define N_PLANES_Z (N_VOXEL_Z + 1) // number of planes along the Z axis

/**
 * @brief Enumerates the cartesian axis of a cartesian 3D coordinate system.
 */
typedef enum {
    X,
    Y,
    Z
} Axis;

/**
 * @brief Models a point of coordinates (x, y, z) in the cartesian 3D coordinate system.
 */
typedef struct {
    double x;
    double y;
    double z;
} Point;

/**
 * @brief Models the range of indices of the planes to consider to compute the intersections with the rays.
 */
typedef struct {
    int minIndx;
    int maxIndx;
} Ranges;

#endif
