#pragma once

#include <rclcpp/rclcpp.hpp>

class Isegmentation
{
public:
    /**
     * → input point buffer (as floats xyzxyz...)
     * num_points: number of points
     * out_points: caller-allocated buffer, at least as big as input
     * out_num_points: actual number of points in segmentation result
     */
    virtual void segment(const float *points,
                         int num_points,
                         float *out_points,
                         int &out_num_points) = 0;
};