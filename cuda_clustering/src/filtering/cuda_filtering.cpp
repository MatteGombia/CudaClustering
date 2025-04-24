#include "cuda_clustering/filtering/cuda_filtering.hpp"
CudaFilter::CudaFilter()
{
  FilterType_t type = PASSTHROUGH;

  this->setP.type = type;
  this->setP.dim = 2;
  this->setP.upFilterLimits = 1.0;
  this->setP.downFilterLimits = 0.0;
  this->setP.limitsNegative = false;

  cudaStreamCreate ( &stream );
}

void CudaFilter::reallocateMemory(unsigned int size)
{
  //stream = NULL;
  //cudaStreamCreate (&stream);

  cudaFree(input);
  cudaMallocManaged(&inputEC, sizeof(float) * 4 * size, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, inputEC);

  cudaFree(output);
  cudaMallocManaged(&outputEC, sizeof(float) * 4 * size, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, outputEC);
}
void CudaFilter::filterPoints(float* input, unsigned int inputSize, float* output, unsigned int* outputSize)
{
  if(memoryAllocated < inputSize){
    reallocateMemory(inputSize);
    memoryAllocated = inputSize;
  }

  cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * inputSize, cudaMemcpyHostToDevice, stream);

  cudaFilter filterTest(stream);
  std::cout << "\n------------checking CUDA PassThrough ---------------- "<< std::endl;

  filterTest.set(this->setP);
  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  filterTest.filter(output, outputSize, input, inputSize);
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA PassThrough Time: %f ms.", time_span.count());
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA PassThrough before filtering: %d", inputSize);
  RCLCPP_INFO(rclcpp::get_logger("clustering_node"), "CUDA PassThrough after filtering: %d", outputSize);
}