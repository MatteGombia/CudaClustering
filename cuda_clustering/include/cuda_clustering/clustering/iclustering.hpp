#pragma once 
#include "cuda_clustering/clustering/cluster_filtering/icluster_filtering.hpp"

class IClustering 
{
    public:
        virtual void extractClusters(bool is_cuda_pointer, float* input, unsigned int inputSize, std::shared_ptr<visualization_msgs::msg::Marker> cones) = 0;
        virtual void getInfo() = 0;
        IClusterFiltering* filter;
};