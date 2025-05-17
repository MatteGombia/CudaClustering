#include "cuda_clustering/segmentation/cuda_segmentation.hpp"
#include <vector>

CudaSegmentation::CudaSegmentation() {}

void CudaSegmentation::reallocateMemory(unsigned int size)
{
  cudaFree(input);
  cudaMallocManaged(&input, sizeof(float) * 4 * size, cudaMemAttachHost);
  cudaStreamAttachMemAsync(stream, input);
}

void CudaSegmentation::freeResources()
{
  cudaFree(input);
  cudaFree(index);
  cudaFree(modelCoefficients);
  cudaStreamDestroy(stream);
}

// Funzione principale per segmentare i punti di input
// inputData: array nel host di float (x, y, z, intensità) × nCount
// nCount: numero di punti in input
// out_points: buffer preallocato per restituire gli inlier
// out_num_points: numero effettivo di inlier trovati
void CudaSegmentation::segment(
    const float *inputData,
    int nCount,
    float **out_points,
    unsigned int *out_num_points)
{
  // Inizio misurazione del tempo
  auto t1 = std::chrono::steady_clock::now();

  size_t inputBytes = sizeof(float) * 4 * nCount;
  if (inputBytes > memory_allocated)
  {
    CudaSegmentation::reallocateMemory(inputBytes);
    memory_allocated = inputBytes;
  }

  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Avvio segmentazione di %d punti", nCount);

  // 1) Creazione dello stream CUDA

  cudaError_t err = cudaStreamCreate(&stream);
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "cudaStreamCreate: %s", cudaGetErrorString(err));
  if (err != cudaSuccess)
  {
    throw std::runtime_error("cudaStreamCreate non riuscita: " + std::string(cudaGetErrorString(err)));
  }

  // 2) Associazione del buffer di input
  cudaMemcpyAsync(input, inputData, inputBytes, cudaMemcpyHostToDevice, stream);

  // 3) Allocazione e inizializzazione del buffer degli indici

  size_t indexBytes = sizeof(int) * nCount;
  err = cudaMallocManaged(&index, indexBytes, cudaMemAttachHost);
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "cudaMallocManaged(index): %s, bytes=%zu", cudaGetErrorString(err), indexBytes);
  if (err != cudaSuccess)
  {
    CudaSegmentation::freeResources();
    throw std::runtime_error("cudaMallocManaged(index) non riuscita");
  }
  cudaStreamAttachMemAsync(stream, index);

  // 4) Allocazione dei coefficienti del modello

  err = cudaMallocManaged(&modelCoefficients, sizeof(float) * 4, cudaMemAttachHost);
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "cudaMallocManaged(coeffs): %s", cudaGetErrorString(err));
  if (err != cudaSuccess)
  {
    CudaSegmentation::freeResources();
    throw std::runtime_error("cudaMallocManaged(modelCoefficients) non riuscita: " + std::string(cudaGetErrorString(err)));
  }
  cudaStreamAttachMemAsync(stream, modelCoefficients);

  // 5) Configurazione ed esecuzione del RANSAC su GPU

  cudaSegmentation impl(SACMODEL_PLANE, SAC_RANSAC, stream);

  segParam_t segP;
  segP.distanceThreshold = 0.01f;
  segP.maxIterations = 50;
  segP.probability = 0.99;
  segP.optimizeCoefficients = true;
  impl.set(segP);

  impl.segment(input, nCount, index, modelCoefficients);
  cudaDeviceSynchronize();
  auto kernelErr = cudaGetLastError();

  if (kernelErr != cudaSuccess)
  {
    throw std::runtime_error("Kernel CUDA non riuscito: " + std::string(cudaGetErrorString(kernelErr)));
  }

  // 6) Raccolta degli inlier

  std::vector<int> inliers;
  inliers.reserve(nCount);
  for (int i = 0; i < nCount; ++i)
  {
    if (index[i] == 1)
      inliers.push_back(i);
  }
  *out_num_points = static_cast<unsigned int>(inliers.size());

  // 7) Allocazione e popolazione dei punti in output

  size_t outBytes = sizeof(float) * 4 * (*out_num_points);
  
  err = cudaMallocManaged(out_points, outBytes, cudaMemAttachGlobal);
  if (err != cudaSuccess)
  {
    *out_num_points = 0;
    *out_points = nullptr;
    CudaSegmentation::freeResources();
    throw std::runtime_error("cudaMallocManaged(out_points) non riuscita: " + std::string(cudaGetErrorString(err)));
  }

  for(size_t i = 0; i < inliers.size(); ++i)
  {
    int idx = inliers[i];
    if (idx < 0 || idx >= nCount) // controllo di validità dell'indice
    {
      RCLCPP_ERROR(rclcpp::get_logger("CudaSegmentation"), "Invalid inlier index: %d", idx);
      continue;
    }

    (*out_points)[4 * i + 0] = input[4 * idx + 0]; // x
    (*out_points)[4 * i + 1] = input[4 * idx + 1]; // y
    (*out_points)[4 * i + 2] = input[4 * idx + 2]; // z
    (*out_points)[4 * i + 3] = 1.0f;               // intensità (dummy value)
  }
  cudaDeviceSynchronize();

  // Fine misurazione del tempo
  auto t2 = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1);
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Segmentazione completata in %.3f ms", duration.count());

  // Log dei coefficienti del modello
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Coefficienti modello: [%.4f, %.4f, %.4f, %.4f]",
              modelCoefficients[0], modelCoefficients[1], modelCoefficients[2], modelCoefficients[3]);

  // Pulizia delle risorse
  CudaSegmentation::freeResources();
}
