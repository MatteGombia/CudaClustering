#pragma once

#include "isegmentation.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <cuda_runtime.h>
#include <cstddef>

typedef struct
{
    double distanceThreshold;
    int maxIterations;
    double probability;
    bool optimizeCoefficients;
} segParam_t;

class cudaSegmentation
{
public:
    // Now Just support: SAC_RANSAC + SACMODEL_PLANE
    cudaSegmentation(int ModelType, int MethodType, cudaStream_t stream = 0);

    ~cudaSegmentation(void);

    /*
    Input:
        cloud_in: data pointer for points cloud
        nCount: count of points in cloud_in
    Output:
        Index: data pointer which has the index of points in a plane from input
        modelCoefficients: data pointer which has the group of coefficients of the plane
    */
    int set(segParam_t param);
    void segment(float *cloud_in, int nCount,
                 int *index, float *modelCoefficients);

private:
    void *m_handle = NULL;
};

class CudaSegmentation : public Isegmentation
{
    
private:
    segParam_t segP;
    float *modelCoefficients = NULL;
    unsigned int memoryAllocated = 0;
    cudaStream_t stream = NULL;
    int *index = NULL;

public:
    CudaSegmentation();
    void segment(const float *points,
                 int num_points,
                 float *out_points,
                 int &out_num_points) override;
};