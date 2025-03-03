#pragma once 
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

class IClustering 
{
    public:
        virtual void extractClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::shared_ptr<visualization_msgs::msg::Marker> cones) = 0;
        virtual void getInfo() = 0;
};