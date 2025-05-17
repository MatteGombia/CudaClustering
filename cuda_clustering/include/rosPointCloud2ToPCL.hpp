#pragma once

// ROS2 PointCloud2 message
#include <sensor_msgs/msg/point_cloud2.hpp>

// PCL conversions & types
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>

#include <vector>

namespace pcl_df
{

    template <typename T>
    void fromROSMsg(const sensor_msgs::msg::PointCloud2 &cloud, pcl::PointCloud<T> &pcl_cloud)
    {
        // Copy info fields
        pcl_conversions::toPCL(cloud.header, pcl_cloud.header);
        pcl_cloud.width = cloud.width;
        pcl_cloud.height = cloud.height;
        pcl_cloud.is_dense = (cloud.is_dense == 1);

        pcl::MsgFieldMap field_map;
        std::vector<pcl::PCLPointField> msg_fields;
        pcl_conversions::toPCL(cloud.fields, msg_fields);
        pcl::createMapping<T>(msg_fields, field_map);

        // Copy point data
        std::uint32_t num_points = cloud.width * cloud.height;
        pcl_cloud.points.resize(num_points);
        std::uint8_t *cloud_data = reinterpret_cast<std::uint8_t *>(&pcl_cloud.points[0]);

        // Fast path when layout matches exactly
        if (field_map.size() == 1 &&
            field_map[0].serialized_offset == 0 &&
            field_map[0].struct_offset == 0 &&
            field_map[0].size == cloud.point_step &&
            field_map[0].size == sizeof(T))
        {
            std::uint32_t row_size = static_cast<std::uint32_t>(sizeof(T) * pcl_cloud.width);
            const std::uint8_t *msg_data = cloud.data.data();
            if (cloud.row_step == row_size)
            {
                memcpy(cloud_data, msg_data, cloud.data.size());
            }
            else
            {
                for (std::uint32_t r = 0; r < cloud.height; ++r)
                {
                    memcpy(cloud_data,
                           msg_data + r * cloud.row_step,
                           row_size);
                    cloud_data += row_size;
                }
            }
        }
        else
        {
            // Generic per-field copy
            for (std::uint32_t r = 0; r < cloud.height; ++r)
            {
                const std::uint8_t *row_data = cloud.data.data() + r * cloud.row_step;
                for (std::uint32_t c = 0; c < cloud.width; ++c)
                {
                    const std::uint8_t *pt_data = row_data + c * cloud.point_step;
                    for (auto &m : field_map)
                    {
                        memcpy(cloud_data + m.struct_offset,
                               pt_data + m.serialized_offset,
                               m.size);
                    }
                    cloud_data += sizeof(T);
                }
            }
        }
    }

} // namespace pcl_df
