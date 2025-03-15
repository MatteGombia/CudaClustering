#pragma once
#include "cuda_clustering/clustering/cluster_filtering/icluster_filtering.hpp"

class DimensionFilter : public IClusterFiltering
{
    private:
        float clusterMaxX, clusterMaxY, clusterMaxZ, maxHeight;
        bool isCone(float minX, float maxX, float minY, float maxY, float minZ, float maxZ);
    public:
        std::optional<geometry_msgs::msg::Point> analiseCluster(float* cluster, unsigned int points_num);
        DimensionFilter();
};