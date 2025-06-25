#include "cuda_clustering/segmentation/cuda_segmentation.hpp"

CudaSegmentation::CudaSegmentation(segParam_t &params)
{
  segP.distanceThreshold = params.distanceThreshold;
  segP.maxIterations = params.maxIterations;
  segP.probability = params.probability;
  segP.optimizeCoefficients = params.optimizeCoefficients;
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
    float *inputData,
    int nCount,
    float **out_points,
    unsigned int *out_num_points)
{
  // Inizio misurazione del tempo
  auto t1 = std::chrono::steady_clock::now();

  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Avvio segmentazione di %d punti", nCount);

  // 1) Creazione dello stream CUDA

  cudaStreamCreate(&stream);

  // 2) Associazione del buffer di input
  // cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  // cudaError_t err = cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
  // if (err != cudaSuccess)
  // {
  //   throw std::runtime_error("cudaMallocManaged(input) non riuscita");
  // }

  // 3) Allocazione e inizializzazione del buffer degli indici

  cudaError_t err = cudaMallocManaged(&index, sizeof(int) * nCount, cudaMemAttachHost);
  RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "cudaMallocManaged(index): %s, bytes=%zu", cudaGetErrorString(err), sizeof(int) * nCount);
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
  cudaMemcpyAsync(modelCoefficients, 0, 4 * sizeof(float), cudaMemcpyHostToDevice, stream); // Inizializza i coefficienti a zero
  std::cout << "coefficienti prima seg: coefficiente[3] = " << modelCoefficients[3] << " coefficiente [2] = " << modelCoefficients[2] << " coefficiente[1] = " << modelCoefficients[1] << std::endl;


  // 5) Configurazione ed esecuzione del RANSAC su GPU

  cudaSegmentation impl(SACMODEL_PLANE, SAC_RANSAC, stream);

  impl.set(segP);
  std::cout << "prima seg" << std::endl;
  impl.segment(inputData, nCount, index, modelCoefficients);
  std::cout << "dopo seg" << std::endl;
  cudaDeviceSynchronize();

    std::cout << "coefficienti dopo seg: coefficiente[3] = " << modelCoefficients[3] << " coefficiente [2] = " << modelCoefficients[2] << " coefficiente[1] = " << modelCoefficients[1] << std::endl;


  if (std::isnan(modelCoefficients[0]) || std::abs(modelCoefficients[3]) > 20)
  {
    std::cout << "Segmentation non valida: coefficiente[3] = " << modelCoefficients[3] << " coefficiente [2] = " << modelCoefficients[2] << " coefficiente[1] = " << modelCoefficients[1] << std::endl;
    skip = true; // Segmentation non valida, salto parte finale
  }
  std::cout << "PASSATO CONTROLLO COEFF" << std::endl;
  if (!skip)
  {
    // 6) Raccolta degli inlier

    std::vector<int> inliers;
    inliers.reserve(nCount);
    for (int i = 0; i < nCount; ++i)
    {
      if (index[i] == -1)
        inliers.push_back(i);
    }
    *out_num_points = static_cast<unsigned int>(inliers.size());

    std::cout << "OUTPOINTS = " << *out_num_points << std::endl;

    // 7) Allocazione e popolazione dei punti in output

    size_t outBytes = sizeof(float) * 4 * (*out_num_points);

    // err = cudaMallocManaged(out_points, outBytes, cudaMemAttachGlobal);
    // if (err != cudaSuccess)
    // {
    //   *out_num_points = 0;
    //   *out_points = nullptr;
    //   CudaSegmentation::freeResources();
    //   throw std::runtime_error("cudaMallocManaged(out_points) non riuscita: " + std::string(cudaGetErrorString(err)));
    // }

    std::cout << "PASSATO OUT_POINTS" << std::endl;

    for (size_t i = 0; i < inliers.size(); ++i)
    {
      int idx = inliers[i];
      if (idx < 0 || idx >= nCount) // controllo di validità dell'indice
      {
        RCLCPP_ERROR(rclcpp::get_logger("CudaSegmentation"), "Invalid inlier index: %d", idx);
        continue;
      }
      (*out_points)[4 * i + 0] = inputData[4 * idx + 0]; // x
      (*out_points)[4 * i + 1] = inputData[4 * idx + 1]; // y
      (*out_points)[4 * i + 2] = inputData[4 * idx + 2]; // z
      // (*out_points)[4 * i + 3] = 1.0f;               // intensità (dummy value)
    }
    cudaDeviceSynchronize();

    std::cout << "FINE" << std::endl;

    // Fine misurazione del tempo
    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1);
    RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Segmentazione completata in %.3f ms", duration.count());

    // Log dei coefficienti del modello
    RCLCPP_INFO(rclcpp::get_logger("CudaSegmentation"), "Coefficienti modello: [%.4f, %.4f, %.4f, %.4f]",
                modelCoefficients[0], modelCoefficients[1], modelCoefficients[2], modelCoefficients[3]);
  }
  // Pulizia delle risorse
  CudaSegmentation::freeResources();
  skip = false; // Reset dello stato di skip per la prossima chiamata
}