#pragma once 
#include <iostream>
#include <string.h>

#include "cuda_clustering/clustering/cuda_clustering.hpp"
#include "cuda_clustering/filtering/cuda_filtering.hpp"
#include "cuda_clustering/clustering/iclustering.hpp"
#include "cuda_clustering/filtering/ifiltering.hpp"

#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>


class ControllerNode : public rclcpp::Node
{
    private:

        std::string input_topic, frame_id;
        float minClusterSize, maxClusterSize, voxelX, voxelY, voxelZ, countThreshold, clusterMaxX, clusterMaxY, clusterMaxZ, maxHeight;
        bool filterOnZ;

        IFilter *filter;
        IClustering *clustering;

        /* Publisher */
        //rclcpp::Publisher<geometry_msgs::msg::PoseArray>::Ptr pose_array_pub_;

        /* Subscriber */
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub;

        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr cones_array_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cp_pub;

        /* Load parameters function */
        void loadParameters();

        /* PointCloud Callback */
        void scanCallback(const sensor_msgs::msg::PointCloud2::Ptr sub_cloud);
    public:
        ControllerNode();
};