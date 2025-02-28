#include "cuda_clustering/filtering/ifiltering.hpp"

class CudaFilter : IFilter
{
    protected:
        pcl::PointCloud<pcl::PointXYZ>::Ptr filterPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc);
    public:
        CudaFilter();
};