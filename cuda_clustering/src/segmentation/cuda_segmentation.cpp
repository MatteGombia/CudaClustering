#include "cuda_segmentation.hpp"
#include <cuda_runtime.h>                         // API CUDA runtime
#include <vector>                                 // std::vector

// Costruttore: inizializza parametri di default e puntatori a nullptr
CudaSegmentation::CudaSegmentation()
    : stream(0), memoryAllocated(0), index(nullptr), modelCoefficients(nullptr)
{
    // Parametri RANSAC di default per estrazione piano
    segP.distanceThreshold = 0.01;    // soglia di distanza per considerare un inlier
    segP.maxIterations = 50;          // numero massimo di iterazioni RANSAC
    segP.probability = 0.99;          // probabilità di successo desiderata
    segP.optimizeCoefficients = true; // ottimizza i coefficienti del modello
}

// Funzione principale per segmentare il punto di input
// points: array host di float (x,y,z,intensity)xnum_points
// num_points: numero di punti
// out_points: buffer preallocato per restituire gli inlier
// out_num_points: numero effettivo di inlier trovati
void CudaSegmentation::segment(const float *points,
                               int num_points,
                               float *out_points,
                               int &out_num_points)
{
    // Calcola dimensione in byte dei punti (3 float per point x,y,z)
    const size_t hostPointsBytes = sizeof(float) * 3 * num_points;

    // (Ri)alloca buffer managed se memoria insufficiente
    if (memoryAllocated < hostPointsBytes)
    {
        // Se già presenti, libera le vecchie regioni
        if (index)
            cudaFree(index);
        if (modelCoefficients)
            cudaFree(modelCoefficients);

        // Alloca array di index (un int per punto) in Unified Memory
        cudaMallocManaged(&index, sizeof(int) * num_points, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream, index);

        // Alloca coefficiente del piano (4 float: a,b,c,d)
        cudaMallocManaged(&modelCoefficients, sizeof(float) * 4, cudaMemAttachHost);
        cudaStreamAttachMemAsync(stream, modelCoefficients);

        // Aggiorna quantità di memoria allocata di riferimento
        memoryAllocated = hostPointsBytes;
    }

    // Alloca array di index (un int per punto) e li inserisce in Unified Memory (in gpu locale)
    cudaMallocManaged(&index, sizeof(int) * num_points, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, index);

    // Alloca coefficiente del piano (4 float: a,b,c,d) e li inserisce in Unified Memory (in gpu locale)
    cudaMallocManaged(&modelCoefficients, sizeof(float) * 4, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, modelCoefficients);

    // Alloca e copia in Unified Memory i punti di input
    float *devPoints = nullptr;
    cudaMallocManaged(&devPoints, hostPointsBytes, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, devPoints);
    // Copia asincrona da host a device
    cudaMemcpyAsync(devPoints, points, hostPointsBytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream); // aspetta completamento copia

    // Crea istanza dell'implementazione CUDA RANSAC
    cudaSegmentation segImpl(
        SACMODEL_PLANE,     // tipo modello sacmodel_plane = 0
        SacMethod::SAC_RANSAC,     // metodo RANSAC = 0
        stream // usa lo stream specificato
    );
    // Applica i parametri configurati
    segImpl.set(segP);
    // Esegue la segmentazione sul device
    segImpl.segment(devPoints, num_points, index, modelCoefficients);

    // Costruisce lista indici inlier (index[i]==1)
    std::vector<int> inliers;
    inliers.reserve(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        if (index[i] == 1)
        {
            inliers.push_back(i);
        }
    }

    // Scrive nel buffer di output solo i punti inlier
    out_num_points = static_cast<int>(inliers.size());
    for (int i = 0; i < out_num_points; ++i)
    {
        int idx = inliers[i];
        // Copia x,y,z,intensity per ogni inlier
        out_points[3 * i + 0] = points[3 * idx + 0];
        out_points[3 * i + 1] = points[3 * idx + 1];
        out_points[3 * i + 2] = points[3 * idx + 2];
    }

    // libera memoria allocata
    cudaFree(devPoints);
    cudaFree(index);
    cudaFree(modelCoefficients);
}

/*
CudaSegmentation::CudaSegmentation()
{
    segP.distanceThreshold = 0.01;
    segP.maxIterations = 100;
    segP.probability = 0.99;
    segP.optimizeCoefficients = true;

    modelCoefficients = NULL;
    memoryAllocated = 0;
}
void CudaSegmentation::segment(const float *points, int num_points, float *out_points, int &out_num_points)
{
    cudaStreamCreate(&stream);

    // index
    //  index should >= nCount of maximum inputdata,
    // index can be used for multi-inputs, be allocated and freed just at beginning and end
    cudaMallocManaged(&index, sizeof(int) * num_points, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, index);
    cudaStreamSynchronize(stream);
    // modelCoefficients can be used for multi-inputs, be allocated and freed just at beginning and end
    cudaMallocManaged(&modelCoefficients, sizeof(float) * 4, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, modelCoefficients);
    cudaStreamSynchronize(stream);

    // segmentation
    cudaSegmentation cudaSeg(SACMODEL_PLANE, SAC_RANSAC, stream);
    cudaSeg.set(setP);
    cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * num_points, cudaMemcpyHostToDevice, stream);
    cudaSeg.segment(input, num_points, index, modelCoefficients);

    float *output = new float[num_points * 4];
    std::vector<int> indexV;

    for (int i = 0; i < num_points; i++)
    {
        if (index[i] == 1)
            indexV.push_back(i);
            output.push_back(input[i]);
    }

    std::cout << "CUDA modelCoefficients: " << modelCoefficients[0]
              << " " << modelCoefficients[1]
              << " " << modelCoefficients[2]
              << " " << modelCoefficients[3]
              << std::endl;


    cudaFree(input);
    cudaFree(index);
    cudaFree(modelCoefficients);
}
*/