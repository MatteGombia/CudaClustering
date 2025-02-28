#include "cuda_clustering/CudaClustering/cuda_clustering.hpp"

void CudaClustering::getInfo(void)
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    RCLCPP_INFO(this->get_logger(),"\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        RCLCPP_INFO(this->get_logger(),"----device id: %d info----\n", i);
        RCLCPP_INFO(this->get_logger(),"  GPU : %s \n", prop.name);
        RCLCPP_INFO(this->get_logger(),"  Capbility: %d.%d\n", prop.major, prop.minor);
        RCLCPP_INFO(this->get_logger(),"  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        RCLCPP_INFO(this->get_logger(),"  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        RCLCPP_INFO(this->get_logger(),"  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        RCLCPP_INFO(this->get_logger(),"  warp size: %d\n", prop.warpSize);
        RCLCPP_INFO(this->get_logger(),"  threads in a block: %d\n", prop.maxThreadsPerBlock);
        RCLCPP_INFO(this->get_logger(),"  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        RCLCPP_INFO(this->get_logger(),"  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    RCLCPP_INFO(this->get_logger(),"\n");
}

void CudaClustering::extractClusters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  cudaStream_t stream = NULL;
  cudaStreamCreate (&stream);

  RCLCPP_INFO(this->get_logger(), "-------------- cudaExtractCluster -----------");
  /*add cuda cluster*/
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNew(new pcl::PointCloud<pcl::PointXYZ>);
  cloudNew = cloud;
  float *inputEC = NULL;
  unsigned int sizeEC = cloudNew->size();
  cudaMallocManaged(&inputEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, inputEC);
  cudaMemcpyAsync(inputEC, cloudNew->points.data(), sizeof(float) * 4 * sizeEC, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  float *outputEC = NULL;
  cudaMallocManaged(&outputEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, outputEC);
  cudaMemcpyAsync(outputEC, cloudNew->points.data(), sizeof(float) * 4 * sizeEC, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  unsigned int *indexEC = NULL;
  cudaMallocManaged(&indexEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, indexEC);
  cudaMemsetAsync(indexEC, 0, sizeof(float) * 4 * sizeEC, stream);
  cudaStreamSynchronize(stream);

  extractClusterParam_t ecp;
  ecp.minClusterSize = this->minClusterSize;           // Minimum cluster size to filter out noise
  ecp.maxClusterSize = this->maxClusterSize;        // Maximum size for large objects
  ecp.voxelX = this->voxelX;                  // Down-sampling resolution in X (meters)
  ecp.voxelY = this->voxelY;                  // Down-sampling resolution in Y (meters)
  ecp.voxelZ = this->voxelZ;                 // Down-sampling resolution in Z (meters)
  ecp.countThreshold = this->countThreshold;           // Minimum points per voxel

  cudaExtractCluster cudaec(stream);
  cudaec.set(ecp);

  t1 = std::chrono::steady_clock::now();

  cudaec.extract(inputEC, sizeEC, outputEC, indexEC);
  cudaStreamSynchronize(stream);
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  RCLCPP_INFO(this->get_logger(), "CUDA extract by Time: %f ms.", time_span.count());

  for (int i = 1; i <= indexEC[0]; i++)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);

    cloud_cluster->width  = indexEC[i];
    cloud_cluster->height = 1;
    cloud_cluster->points.resize (cloud_cluster->width * cloud_cluster->height);
    cloud_cluster->is_dense = true;

    unsigned int outoff = 0;
    for (int w = 1; w < i; w++)
    {
      if (i>1) {
        outoff += indexEC[w];
      }
    }

    double maxX=-1000, maxY=-1000, maxZ=-1000, minX=1000, minY=1000, minZ=1000;
    for (std::size_t k = 0; k < indexEC[i]; ++k)
    {
      cloud_cluster->points[k].x = outputEC[(outoff+k)*4+0];
      cloud_cluster->points[k].y = outputEC[(outoff+k)*4+1];
      cloud_cluster->points[k].z = outputEC[(outoff+k)*4+2];

      if(cloud_cluster->points[k].x > maxX)
        maxX = cloud_cluster->points[k].x;
      if(cloud_cluster->points[k].y > maxY)
        maxY = cloud_cluster->points[k].y;
      if(cloud_cluster->points[k].z > maxZ)
        maxZ = cloud_cluster->points[k].z;
      if(cloud_cluster->points[k].x < minX)
        minX = cloud_cluster->points[k].x;
      if(cloud_cluster->points[k].y < minY)
        minY = cloud_cluster->points[k].y;
      if(cloud_cluster->points[k].z < minZ)
        minZ = cloud_cluster->points[k].z;
    }
    
    if(minZ < this->maxHeight &&
        (maxX - minX) < this->clusterMaxX &&
        (maxY - minY) < this->clusterMaxY &&
        (maxZ - minZ) < this->clusterMaxZ){
      geometry_msgs::msg::Point pnt;
      pnt.x = (maxX + minX) / 2;
      pnt.y = (maxY + minY) / 2;
      pnt.z = (maxZ + minZ) / 2;
      cones.points.push_back(pnt);
      RCLCPP_INFO(this->get_logger(), "PointCloud representing the Cluster: %d data points.", cloud_cluster->size() );
    }
    else{
      RCLCPP_INFO(this->get_logger(), "DISCARDED the Cluster: %d data points.", cloud_cluster->size() );
    }
  }
  cones_array_pub->publish(cones);

  cudaFree(inputEC);
  cudaFree(outputEC);
  cudaFree(indexEC);
  /*end*/
}
