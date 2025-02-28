#pragma once 
#include <iostream>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <Eigen/Dense>
#include "cuda_runtime.h"

#include "cuda_clustering/clustering/iclustering.hpp"

/* GPU stuff */
/*#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/containers/device_array.hpp>
#include <pcl/gpu/segmentation/gpu_extract_clusters.h>
#include <pcl/gpu/segmentation/impl/gpu_extract_clusters.hpp>*/

typedef struct {
  unsigned int minClusterSize;
  unsigned int maxClusterSize;
  float voxelX;
  float voxelY;
  float voxelZ;
  int countThreshold;
} extractClusterParam_t;

class cudaExtractCluster
{
  public:
    cudaExtractCluster(cudaStream_t stream = 0);
    ~cudaExtractCluster(void);
    int set(extractClusterParam_t param);
    int extract(float *cloud_in, int nCount, float *output, unsigned int *index);
  private:
    void *m_handle = NULL;
};

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}


class CudaClustering : public IClustering
{
    public:
        CudaClustering();
        void getInfo();

        void extractClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
};