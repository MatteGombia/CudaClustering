#include "cuda_clustering/cuda_clustering.hpp"

CudaClusteringNode::CudaClusteringNode() : Node("cuda_clustering_node"){
    this->loadParameters();

    this->getInfo();

    /* Define QoS for Best Effort messages transport */
	  auto qos = rclcpp::QoS(rclcpp::KeepLast(10), rmw_qos_profile_sensor_data);

    this->cones_array_pub = this->create_publisher<visualization_msgs::msg::Marker>("/perception/newclusters", 100);
    this->filtered_cp_pub  = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_cp", 100);

    /* Create subscriber */
    this->cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(this->input_topic, qos, 
        std::bind(&CudaClusteringNode::scanCallback, this, std::placeholders::_1));
}

void CudaClusteringNode::loadParameters()
{

    declare_parameter("input_topic", ""); 
    declare_parameter("frame_id", ""); 
    declare_parameter("minClusterSize", 0.0); 
    declare_parameter("maxClusterSize", 0.0); 
    declare_parameter("voxelX", 0.0); 
    declare_parameter("voxelY", 0.0); 
    declare_parameter("voxelZ", 0.0); 
    declare_parameter("countThreshold", 0.0); 
    declare_parameter("clusterMaxX", 0.0); 
    declare_parameter("clusterMaxY", 0.0); 
    declare_parameter("clusterMaxZ", 0.0); 
    declare_parameter("maxHeight", 0.0);
    declare_parameter("filterOnZ", false);


    get_parameter("input_topic", this->input_topic); 
    get_parameter("frame_id", this->frame_id); 
    get_parameter("minClusterSize", this->minClusterSize); 
    get_parameter("maxClusterSize", this->maxClusterSize); 
    get_parameter("voxelX", this->voxelX); 
    get_parameter("voxelY", this->voxelY); 
    get_parameter("voxelZ", this->voxelZ); 
    get_parameter("countThreshold", this->countThreshold); 
    get_parameter("clusterMaxX", this->clusterMaxX); 
    get_parameter("clusterMaxY", this->clusterMaxY); 
    get_parameter("clusterMaxZ", this->clusterMaxZ); 
    get_parameter("maxHeight", this->maxHeight); 
    get_parameter("filterOnZ", this->filterOnZ); 
}

void CudaClusteringNode::scanCallback(sensor_msgs::msg::PointCloud2::Ptr sub_cloud)
{
    // Create a PCL PointCloud object
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Convert from sensor_msgs::PointCloud2 to pcl::PointCloud
    pcl::fromROSMsg(*sub_cloud, *pcl_cloud);

    if(this->filterOnZ){
      pcl_cloud = this->testCUDAFiltering(pcl_cloud);
    }

    RCLCPP_INFO(this->get_logger(), "-------------- test CUDA lib -----------");
    testCUDA(pcl_cloud);
}