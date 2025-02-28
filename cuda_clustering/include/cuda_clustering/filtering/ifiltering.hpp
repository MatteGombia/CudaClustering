#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>


class IFilter
{
    protected:
        virtual pcl::PointCloud<pcl::PointXYZ>::Ptr filterPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc) = 0;
    public:
        IFilter();
};