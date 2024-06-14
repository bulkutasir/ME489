%%writefile kmeans_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curand.h>

// Constants
#define BUFSIZE 512
#define MAX_ITER 10000
#define THREADS_PER_BLOCK 256

// Global variables
int Nd, Nc, Np;
double TOL;

// Function to read input file
double readInputFile(char *fileName, char* tag) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("Error opening the input file\n");
        exit(EXIT_FAILURE);
    }
    double result = 0.0;
    char buffer[BUFSIZE], fileTag[BUFSIZE];
    while (fgets(buffer, BUFSIZE, fp) != NULL) {
        sscanf(buffer, "%s", fileTag);
        if (strstr(fileTag, tag)) {
            fgets(buffer, BUFSIZE, fp);
            sscanf(buffer, "%lf", &result);
            fclose(fp);
            return result;
        }
    }
    printf("ERROR Could not find the tag: [%s] in the file [%s]\n", tag, fileName);
    fclose(fp);
    exit(EXIT_FAILURE);
}

// Function to read data file
void readDataFile(char *fileName, double *data) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("Error opening the input file\n");
    }
    int sk = 0;
    char buffer[BUFSIZE];
    while (fgets(buffer, BUFSIZE, fp) != NULL) {
        if (Nd == 2)
            sscanf(buffer, "%lf %lf", &data[sk * Nd + 0], &data[sk * Nd + 1]);
        if (Nd == 3)
            sscanf(buffer, "%lf %lf %lf", &data[sk * Nd + 0], &data[sk * Nd + 1], &data[sk * Nd + 2]);
        if (Nd == 4)
            sscanf(buffer, "%lf %lf %lf %lf", &data[sk * Nd + 0], &data[sk * Nd + 1], &data[sk * Nd + 2], &data[sk * Nd + 3]);
        sk++;
    }
    fclose(fp);
}

// Function to write data to file
void writeDataToFile(char *fileName, double *data, int *Ci) {
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening the output file\n");
    }
    for (int p = 0; p < Np; p++) {
        fprintf(fp, "%d ", Ci[p]); // Cluster number
        for (int dim = 0; dim < Nd; dim++) {
            fprintf(fp, "%.4f ", data[p * Nd + dim]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}


// Function to write centroid to file
void writeCentroidToFile(char *fileName, double *Cm) {
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening the output file\n");
    }
    for (int n = 0; n < Nc; n++) {
        for (int dim = 0; dim < Nd; dim++) {
            fprintf(fp, "%.4f ", Cm[n * Nd + dim]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Kernel to calculate Euclidean distances from each point to each centroid
__global__ void distanceKernel(double *data, double *Cm, double *distances, int Nd, int Nc, int Np) {
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx < Np) {
        for (int clusterIdx = 0; clusterIdx < Nc; clusterIdx++) {
            double sum = 0.0;
            for (int dim = 0; dim < Nd; dim++) {
                double diff = data[pointIdx * Nd + dim] - Cm[clusterIdx * Nd + dim];
                sum += diff * diff;
            }
            distances[pointIdx * Nc + clusterIdx] = sqrt(sum);
        }
    }
}

// Kernel to assign each point to the nearest centroid
__global__ void assignPointsKernel(double *distances, int *Ci, int Np, int Nc) {
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx < Np) {
        double minDist = INFINITY;
        int minIdx = 0;
        for (int clusterIdx = 0; clusterIdx < Nc; clusterIdx++) {
            double dist = distances[pointIdx * Nc + clusterIdx];
            if (dist < minDist) {
                minDist = dist;
                minIdx = clusterIdx;
            }
        }
        Ci[pointIdx] = minIdx;
    }
}

// Update centroids
__global__ void updateCentroidsKernel(double *data, int *Ci, int *Ck, double *Cm, int Nd, int Np, int Nc) {
    extern __shared__ double sharedData[];
    int tid = threadIdx.x;
    int clusterIdx = blockIdx.x;
    // Initialize shared memory
    for (int dim = 0; dim < Nd; dim++) {
        sharedData[tid * Nd + dim] = 0;
    }
    __syncthreads();

    int start = blockIdx.y * blockDim.y + threadIdx.y;
    int stride = blockDim.y * gridDim.y;
    for (int i = start; i < Np; i += stride) {
        if (Ci[i] == clusterIdx) {
            for (int dim = 0; dim < Nd; dim++) {
                atomicAddDouble(&sharedData[tid * Nd + dim], data[i * Nd + dim]);
            }
            atomicAdd(&Ck[clusterIdx], 1);
        }
    }
    __syncthreads();
    // Reduce results within the block
    if (tid == 0) {
        for (int i = 1; i < blockDim.x; i++) {
            for (int dim = 0; dim < Nd; dim++) {
                sharedData[dim] += sharedData[i * Nd + dim];
            }
        }
        // Write block results to global memory
        for (int dim = 0; dim < Nd; dim++) {
            atomicAddDouble(&Cm[clusterIdx * Nd + dim], sharedData[dim]);
        }
    }
}

// Normalize centroids
__global__ void normalizeCentroidsKernel(double *Cm, int *Ck, int Nd, int Nc) {
    int clusterIdx = blockIdx.x;
    if (clusterIdx < Nc) {
        if (Ck[clusterIdx] > 0) {
            for (int dim = 0; dim < Nd; dim++) {
                Cm[clusterIdx * Nd + dim] /= Ck[clusterIdx];
            }
        }
    }
}

// Main kMeans function
void kMeans(double *data, int *Ci, int *Ck, double *Cm) {
    double *d_data, *d_Cm, *d_distances;
    int *d_Ci, *d_Ck;
    size_t dataSize = Np * Nd * sizeof(double);
    size_t centroidSize = Nc * Nd * sizeof(double);
    size_t distanceSize = Np * Nc * sizeof(double);
    size_t CiSize = Np * sizeof(int);
    size_t CkSize = Nc * sizeof(int);
    // Allocate memory on the device
    cudaMalloc(&d_data, dataSize);
    cudaMalloc(&d_Cm, centroidSize);
    cudaMalloc(&d_distances, distanceSize);
    cudaMalloc(&d_Ci, CiSize);
    cudaMalloc(&d_Ck, CkSize);
    // Copy data to device
    cudaMemcpy(d_data, data, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cm, Cm, centroidSize, cudaMemcpyHostToDevice);
    double err = INFINITY;
    double prev_err = INFINITY;
    float percent_change = INFINITY;
    int sk = 0;
    while (percent_change > TOL && sk < MAX_ITER) {
        cudaMemset(d_Ck, 0, CkSize);
        // Calculate distances
        int blocksPerGrid = (Np + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        distanceKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_data, d_Cm, d_distances, Nd, Nc, Np);
        cudaDeviceSynchronize();
        // Assign points to nearest centroids
        assignPointsKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_distances, d_Ci, Np, Nc);
        cudaDeviceSynchronize();
        // Update centroids
        dim3 blocks(Nc, 1);
        dim3 threads(THREADS_PER_BLOCK, 1);
        updateCentroidsKernel<<<blocks, threads, Nd * THREADS_PER_BLOCK * sizeof(double)>>>(d_data, d_Ci, d_Ck, d_Cm, Nd, Np, Nc);
        cudaDeviceSynchronize();
        // Normalize centroids
        normalizeCentroidsKernel<<<Nc, 1>>>(d_Cm, d_Ck, Nd, Nc);
        cudaDeviceSynchronize();
        // Copy results back to host for error computation and next iteration preparation
        cudaMemcpy(Cm, d_Cm, centroidSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(Ci, d_Ci, CiSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(Ck, d_Ck, CkSize, cudaMemcpyDeviceToHost);

        err = 0.0;
        for (int i = 0; i < Nc; i++) {
            for (int j = 0; j < Nd; j++) {
                err += pow(Cm[i * Nd + j], 2);
            }
        }
        err = sqrt(err);

        // Calculate percent change in error
        percent_change = fabs(((prev_err - err)));
        if(sk < 5) {
            percent_change = 100;
        }
        printf("Iteration %d, Error: %.2f\n", sk, percent_change);

        // Store current error as previous error for next iteration
        prev_err = err;
        sk++;
    }
    // Free device memory
    cudaFree(d_data);
    cudaFree(d_Cm);
    cudaFree(d_distances);
    cudaFree(d_Ci);
    cudaFree(d_Ck);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./kmns input.dat data.dat\n");
        return -1;
    }
    Np = (int)readInputFile(argv[1], (char*)"[NUMBER_OF_POINTS]");
    Nd = (int)readInputFile(argv[1], (char*)"[DATA_DIMENSION]");
    Nc = (int)readInputFile(argv[1], (char*)"[NUMBER_OF_CLUSTERS]");
    TOL = readInputFile(argv[1], (char*)"[TOLERANCE]");
    printf("Number of points: %d\n", Np);
    printf("Number of dimensions: %d\n", Nd);
    printf("Number of clusters: %d\n", Nc);
    printf("Tolerance: %lf\n", TOL);
    // Allocate memory for data and Cm
    double *data = (double*)malloc(Np * Nd * sizeof(double));
    int *Ci = (int*)calloc(Np, sizeof(int));
    int *Ck = (int*)calloc(Nc, sizeof(int));
    double *Cm = (double*)calloc(Nc * Nd, sizeof(double));
    // Read data from file
    readDataFile(argv[2], data);
    // Initialize Cm randomly
    printf("Initial Cm:\n");
    for (int n = 0; n < Nc; n++) {
        int idx = rand() % Np;
        for (int dim = 0; dim < Nd; dim++) {
            Cm[n * Nd + dim] = data[idx * Nd + dim];
        }
        printf("Centroid %d: ", n);
        for (int dim = 0; dim < Nd; dim++) {
            printf("%f ", Cm[n * Nd + dim]);
        }
        printf("\n");
    }
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    kMeans(data, Ci, Ck, Cm);
    cudaEventRecord(end);
    printf("Final Cm:\n");
    for (int n = 0; n < Nc; n++) {
        printf("Centroid %d: ", n);
        for (int dim = 0; dim < Nd; dim++) {
            printf("%f ", Cm[n * Nd + dim]);
        }
        printf("\n");
    }
    float elapsed;
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed, start, end);
    elapsed /= 1000.; // convert to seconds
    printf("elapsed time: %g\n", elapsed);
    writeDataToFile((char*)"output.dat", data, Ci);
    return 0;
}