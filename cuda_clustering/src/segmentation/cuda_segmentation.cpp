#include "cuda_clustering/segmentation/cuda_segmentation.hpp"
#include <vector> 

CudaSegmentation::CudaSegmentation() {}

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

  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Avvio segmentazione di %d punti", nCount);

  // 1) Creazione dello stream CUDA
  cudaStream_t stream = nullptr;
  cudaError_t err = cudaStreamCreate(&stream);
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "cudaStreamCreate: %s", cudaGetErrorString(err));
  if (err != cudaSuccess)
  {
    throw std::runtime_error("cudaStreamCreate non riuscita: " + std::string(cudaGetErrorString(err)));
  }

  // 2) Allocazione e associazione del buffer di input
  size_t inputBytes = sizeof(float) * 4 * nCount;
  float *input = nullptr;
  err = cudaMallocManaged(&input, inputBytes, cudaMemAttachHost);
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "cudaMallocManaged(input): %s, bytes=%zu", cudaGetErrorString(err), inputBytes);
  if (err != cudaSuccess)
  {
    cudaStreamDestroy(stream);
    throw std::runtime_error("cudaMallocManaged(input) non riuscita: " + std::string(cudaGetErrorString(err)));
  }
  cudaStreamAttachMemAsync(stream, input);
  cudaMemcpyAsync(input, inputData, inputBytes, cudaMemcpyHostToDevice, stream);

  // 3) Allocazione e inizializzazione del buffer degli indici
  size_t indexBytes = sizeof(int) * nCount;
  int *index = nullptr;
  err = cudaMallocManaged(&index, indexBytes, cudaMemAttachHost);
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "cudaMallocManaged(index): %s, bytes=%zu", cudaGetErrorString(err), indexBytes);
  if (err != cudaSuccess)
  {
    cudaFree(input);
    cudaStreamDestroy(stream);
    throw std::runtime_error("cudaMallocManaged(index) non riuscita");
  }
  cudaStreamAttachMemAsync(stream, index);

  // 4) Allocazione dei coefficienti del modello
  float *modelCoefficients = nullptr;
  err = cudaMallocManaged(&modelCoefficients, sizeof(float) * 4, cudaMemAttachHost);
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "cudaMallocManaged(coeffs): %s", cudaGetErrorString(err));
  if (err != cudaSuccess)
  {
    cudaFree(index);
    cudaFree(input);
    cudaStreamDestroy(stream);
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

  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Avvio del kernel di segmentazione...");
  impl.segment(input, nCount, index, modelCoefficients);
  cudaDeviceSynchronize();
  auto kernelErr = cudaGetLastError();
  if (kernelErr != cudaSuccess)
  {
    throw std::runtime_error("Kernel CUDA non riuscito: " + std::string(cudaGetErrorString(kernelErr)));
  }

  cudaStreamSynchronize(stream);

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
  float *d_out = nullptr;
  err = cudaMallocManaged(&d_out, outBytes, cudaMemAttachGlobal);
  if (err != cudaSuccess)
  {
    RCLCPP_ERROR(rclcpp::get_logger("CudaSegmentation"), "cudaMallocManaged(out) non riuscita: %s", cudaGetErrorString(err));
    *out_num_points = 0;
    *out_points = nullptr;
    cudaFree(modelCoefficients);
    cudaFree(index);
    cudaFree(input);
    cudaStreamDestroy(stream);
    return;
  }
  for (unsigned i = 0; i < *out_num_points; ++i)
  {
    int idx = inliers[i];
    d_out[4 * i + 0] = input[4 * idx + 0];
    d_out[4 * i + 1] = input[4 * idx + 1];
    d_out[4 * i + 2] = input[4 * idx + 2];
    d_out[4 * i + 3] = 1.0f;
  }
  cudaDeviceSynchronize();

  *out_points = d_out;

  // Fine misurazione del tempo
  auto t2 = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1);
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Segmentazione completata in %.3f ms", duration.count());

  // Log dei coefficienti del modello
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Coefficienti modello: [%.4f, %.4f, %.4f, %.4f]",
              modelCoefficients[0], modelCoefficients[1], modelCoefficients[2], modelCoefficients[3]);

  // Pulizia delle risorse
  cudaFree(input);
  cudaFree(index);
  cudaFree(modelCoefficients);
  cudaStreamDestroy(stream);
}
