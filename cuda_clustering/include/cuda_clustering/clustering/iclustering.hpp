#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

struct IClustering 
{
    protected:
        virtual void getInfo() = 0;

    public:
        IClustering();
        void exctractClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
};