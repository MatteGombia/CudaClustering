#include "cuda_clustering/cuda_clustering.hpp"

CudaClusteringNode::CudaClusteringNode() : Node("cuda_clustering_node"){
    this->loadParameters();

    this->getInfo();

    /* Define QoS for Best Effort messages transport */
	  auto qos = rclcpp::QoS(rclcpp::KeepLast(10), rmw_qos_profile_sensor_data);

    this->cones_array_pub = this->create_publisher<visualization_msgs::msg::Marker>("/perception/newclusters", 100);

    /* Create subscriber */
    this->cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(this->input_topic, qos, 
        std::bind(&CudaClusteringNode::scanCallback, this, std::placeholders::_1));
    
    
}

void CudaClusteringNode::loadParameters()
{

    declare_parameter("input_topic", ""); 
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
      pcl::PointCloud<pcl::PointXYZ>::Ptr = this->testCUDAFiltering(pcl_cloud);
    }

    RCLCPP_INFO(this->get_logger(), "-------------- test CUDA lib -----------");
    testCUDA(pcl_cloud);

    // RCLCPP_INFO(this->get_logger(), "\n-------------- test PCL lib -----------");
    // testPCL(pcl_cloud);

    //RCLCPP_INFO(this->get_logger(), "\n-------------- test PCL GPU lib -----------");
    //testPclGpu(pcl_cloud);
}

void CudaClusteringNode::getInfo(void)
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

pcl::PointCloud<pcl::PointXYZ>::Ptr CudaClusteringNode::testCUDAFiltering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  cudaStream_t stream = NULL;
  cudaStreamCreate ( &stream );

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst(new pcl::PointCloud<pcl::PointXYZ>);

  unsigned int nCount = cloudSrc->width * cloudSrc->height;
  float *inputData = (float *)cloudSrc->points.data();

  cloudDst->width  = nCount;
  cloudDst->height = 1;
  cloudDst->resize (cloudDst->width * cloudDst->height);

  float *outputData = (float *)cloudDst->points.data();

  memset(outputData,0,sizeof(float)*4*nCount);

  std::cout << "\n------------checking CUDA ---------------- "<< std::endl;
  std::cout << "CUDA Loaded "
      << cloudSrc->width*cloudSrc->height
      << " data points from PCD file with the following fields: "
      << pcl::getFieldsList (*cloudSrc)
      << std::endl;

  float *input = NULL;
  cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, input );
  cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  float *output = NULL;
  cudaMallocManaged(&output, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, output );
  cudaStreamSynchronize(stream);

  cudaFilter filterTest(stream);
  FilterParam_t setP;

  unsigned int countLeft = 0;
  std::cout << "\n------------checking CUDA PassThrough ---------------- "<< std::endl;

  memset(outputData,0,sizeof(float)*4*nCount);

  FilterType_t type = PASSTHROUGH;

  setP.type = type;
  setP.dim = 0;
  setP.upFilterLimits = 0.5;
  setP.downFilterLimits = -0.5;
  setP.limitsNegative = false;
  filterTest.set(setP);

  cudaDeviceSynchronize();
  t1 = std::chrono::steady_clock::now();
  filterTest.filter(output, &countLeft, input, nCount);
  checkCudaErrors(cudaMemcpyAsync(outputData, output, sizeof(float) * 4 * countLeft, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaDeviceSynchronize());
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "CUDA PassThrough by Time: " << time_span.count() << " ms." << std::endl;
  std::cout << "CUDA PassThrough before filtering: " << nCount << std::endl;
  std::cout << "CUDA PassThrough after filtering: " << countLeft << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNew(new pcl::PointCloud<pcl::PointXYZ>);
  cloudNew->width = countLeft;
  cloudNew->height = 1;
  cloudNew->points.resize (cloudNew->width * cloudNew->height);

  int check = 0;
  for (std::size_t i = 0; i < cloudNew->size(); ++i)
  {
      cloudNew->points[i].x = output[i*4+0];
      cloudNew->points[i].y = output[i*4+1];
      cloudNew->points[i].z = output[i*4+2];
  }
  return cloudNew;
}

void CudaClusteringNode::testCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  visualization_msgs::msg::Marker cones;
  cones.id = 0;
  cones.header.frame_id = "os_sensor";
  cones.header.stamp = this->now();
  cones.ns = "ListaConiRilevati";
  cones.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  cones.action = visualization_msgs::msg::Marker::ADD;
  cones.scale.x = 0.3; //0.5
  cones.scale.y = 0.2;
  cones.scale.z = 0.2;
  cones.color.a = 1.0; //1.0
  cones.color.r = 1.0;
  cones.color.g = 0.0;
  cones.color.b = 1.0;
  //Initialize with identity quaternion = no rotation
  cones.pose.orientation.x = 0.0;
  cones.pose.orientation.y = 0.0;
  cones.pose.orientation.z = 0.0;
  cones.pose.orientation.w = 1.0;

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

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

void CudaClusteringNode::testPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  cloud_f = cloud;
  // cluster
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

  t1 = std::chrono::steady_clock::now();
  tree->setInputCloud (cloud);
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  RCLCPP_INFO(this->get_logger(), "PCL(CPU) cluster kd-tree by Time: " ,time_span.count() ," ms.");

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (1); // 2cm
  ec.setMinClusterSize (5);
  ec.setMaxClusterSize (50);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_f);

  t1 = std::chrono::steady_clock::now();
  ec.extract (cluster_indices);
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  RCLCPP_INFO(this->get_logger(), "PCL(CPU) cluster extracted by Time: %f ms.", time_span.count());

  RCLCPP_INFO(this->get_logger(), "PointCloud cluster_indices: %d.", cluster_indices.size() );

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->push_back ((*cloud_f)[*pit]); //*
    cloud_cluster->width = cloud_cluster->size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    RCLCPP_INFO(this->get_logger(), "PointCloud representing the Cluster: %d data points.",cloud_cluster->size() );
  }
}

void CudaClusteringNode::testPclGpu(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered)
{
  // pcl::gpu::Octree::PointCloud cloud_device;
  // cloud_device.upload(cloud_filtered->points);
  
  // pcl::gpu::Octree::Ptr octree_device (new pcl::gpu::Octree);
  // octree_device->setCloud(cloud_device);
  // octree_device->build();

  // std::vector<pcl::PointIndices> cluster_indices_gpu;
  // pcl::gpu::EuclideanClusterExtraction<pcl::PointXYZ> gec;
  // gec.setClusterTolerance (0.02); // 2cm
  // gec.setMinClusterSize (5);
  // gec.setMaxClusterSize (25);
  // gec.setSearchMethod (octree_device);
  // gec.setHostCloud( cloud_filtered);
  // gec.extract (cluster_indices_gpu);

  // for (const pcl::PointIndices& cluster : cluster_indices_gpu)
  // {
  //   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_gpu (new pcl::PointCloud<pcl::PointXYZ>);
  //   for (const auto& index : (cluster.indices))
  //     cloud_cluster_gpu->push_back ((*cloud_filtered)[index]); //*
  //   cloud_cluster_gpu->width = cloud_cluster_gpu->size ();
  //   cloud_cluster_gpu->height = 1;
  //   cloud_cluster_gpu->is_dense = true;

  //   std::cout << "PointCloud representing the Cluster: " << cloud_cluster_gpu->size () << " data points." << std::endl;
  // }
}