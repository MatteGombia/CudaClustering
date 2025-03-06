#pragma once 
#include <pcl_conversions/pcl_conversions.h>

class IClustering 
{
    public:
        virtual void extractClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::shared_ptr<visualization_msgs::msg::Marker> cones) = 0;
        virtual void getInfo() = 0;
};