#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <Eigen/Dense>
#include "cuda_runtime.h"

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

typedef enum {
    PASSTHROUGH = 0,
    VOXELGRID = 1,
} FilterType_t;

typedef struct {
    FilterType_t type;
    //0=x,1=y,2=z
    //type PASSTHROUGH
    int dim;
    float upFilterLimits;
    float downFilterLimits;
    bool limitsNegative;
    //type VOXELGRID
    float voxelX;
    float voxelY;
    float voxelZ;

} FilterParam_t;

class cudaFilter
{
public:
    cudaFilter(cudaStream_t stream = 0);
    ~cudaFilter(void);
    /*
    Input:
        source: data pointer for points cloud
        nCount: count of points in cloud_in
    Output:
        output: data pointer which has points filtered by CUDA
        countLeft: count of points in output
    */
    int set(FilterParam_t param);
    int filter(void *output, unsigned int *countLeft, void *source, unsigned int nCount);

    void *m_handle = NULL;
};



class CudaClusteringNode : public rclcpp::Node
{
    private:
        std::string input_topic, frame_id;
        float minClusterSize,maxClusterSize, voxelX,voxelY,voxelZ, countThreshold, clusterMaxX, clusterMaxY, clusterMaxZ, maxHeight;
        bool filterOnZ;

        /* Publisher */
        //rclcpp::Publisher<geometry_msgs::msg::PoseArray>::Ptr pose_array_pub_;

        /* Subscriber */
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub;

        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr cones_array_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cp_pub;

        void getInfo();

        void testCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

        void testPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

        void testPclGpu(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered);

        pcl::PointCloud<pcl::PointXYZ>::Ptr testCUDAFiltering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc);
    public:
        CudaClusteringNode();

        /* Load parameters function */
        void loadParameters();

        /* PointCloud Callback */
        void scanCallback(const sensor_msgs::msg::PointCloud2::Ptr sub_cloud);
};