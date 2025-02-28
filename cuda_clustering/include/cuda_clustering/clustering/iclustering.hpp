#pragma once 
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

class IClustering 
{
    public:
        void exctractClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        virtual void getInfo() = 0;
        IClustering();
};