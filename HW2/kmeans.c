#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#pragma comment(lib, "math.lib")

typedef struct {
    double *coords;
    int cluster;
} DataPoint;

typedef struct {
    double *coords; 
} Centroid;

void readmyinput(int *N, int *K, int *DIM, double *TOL);
void readmydatap(DataPoint *dataPoints, int N, int DIM);
void calcCardinalities(DataPoint *dataPoints, int N, int K);
void calcCentroids(DataPoint *dataPoints, int N, int K, int DIM, Centroid *centroids);
double calcDistance(const double *pointCoords, const double *centroidCoords, int DIM);
void printClusterInfo(DataPoint *dataPoints, int N, int K, int DIM, Centroid *centroids);
void printClusterMembers(DataPoint *dataPoints, int N, int K, int DIM);

int main() {
    int N, K, DIM;
    double TOL;
    double diff = DBL_MAX; // Initialize to maximum double value to ensure the loop starts
    // Read configurations from input.dat
    readmyinput(&N, &K, &DIM, &TOL);
    //printf("N: %d, K: %d, DIM: %d, Tolerance: %.5f\n", N, K, DIM, TOL);

    // Allocate memory 
    DataPoint *dataPoints = (DataPoint *)malloc(N * sizeof(DataPoint));
    for (int i = 0; i < N; i++) {
        dataPoints[i].coords = (double *)malloc(DIM * sizeof(double));
    }
    // Read data points from data.dat
    readmydatap(dataPoints, N, DIM);

    // Verify data points taken
    //printf("Data Points:\n");
    for (int i = 0; i < N && i < 5; i++) { // Print only the first 5 for brevity
        for (int j = 0; j < DIM; j++) {
            //printf("%.15f ", dataPoints[i].coords[j]);
        }
        //printf("\n");
    }

    // Initial cluster assignment 
    for (int i = 0; i < N; i++) {
        dataPoints[i].cluster = i % K;
    }

    calcCardinalities(dataPoints, N, K);

    // Initialize first centroids
    Centroid *centroids = (Centroid *)malloc(K * sizeof(Centroid));
    for (int k = 0; k < K; k++) {
        centroids[k].coords = (double *)malloc(DIM * sizeof(double));
    }

    calcCentroids(dataPoints, N, K, DIM, centroids);

    Centroid *prevCentroids = (Centroid *)malloc(K * sizeof(Centroid));
    for (int k = 0; k < K; k++) {
        prevCentroids[k].coords = (double *)malloc(DIM * sizeof(double));
    }

    // THE MAIN ITERATION LOOP
    while (diff > TOL) {
        // Step 1: Save current centroids to prevCentroids for comparison
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < DIM; j++) {
                prevCentroids[k].coords[j] = centroids[k].coords[j];
            }
        }

        // Step 2: Reassign clusters based on current centroids
        for (int i = 0; i < N; i++) {
            double nearestDistance = DBL_MAX;
            int nearestCentroid = 0;
            for (int k = 0; k < K; k++) {
                double distance = calcDistance(dataPoints[i].coords, centroids[k].coords, DIM);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestCentroid = k;
                }
            }
            dataPoints[i].cluster = nearestCentroid;
        }

        // Step 3: Recalculate centroids
        calcCentroids(dataPoints, N, K, DIM, centroids);


        // Step 4: Calculate the maximum change in centroids
        diff = 0.0;
        for (int k = 0; k < K; k++) {
            double centroidChange = calcDistance(centroids[k].coords, prevCentroids[k].coords, DIM);
            if (centroidChange > diff) {
                diff = centroidChange;
            }
        }
    }

    printClusterInfo(dataPoints, N, K, DIM, centroids);
    printClusterMembers(dataPoints, N, K, DIM);

    // Freeing Memory
    for (int k = 0; k < K; k++) {
        free(prevCentroids[k].coords);
    }
    free(prevCentroids);
    
    for (int i = 0; i < N; i++) {
        free(dataPoints[i].coords);
    }
    free(dataPoints);

    for (int k = 0; k < K; k++) {
        free(centroids[k].coords); // Free each centroid's coordinates
    }
    free(centroids);

    return 0;
}
void readmyinput(int *N, int *K, int *DIM, double *TOL) {
    FILE *file = fopen("input.dat", "r");
    if (file == NULL) {
        //printf("Failed to open file\n");
        return;
    }

    fscanf(file, "[NUMBER_OF_POINTS]\n%d\n", N);
    fscanf(file, "[NUMBER_OF_CLUSTERS]\n%d\n", K);
    fscanf(file, "[DATA_DIMENSION]\n%d\n", DIM);
    fscanf(file, "[TOLERANCE]\n%lf\n", TOL);

    fclose(file);
}
void readmydatap(DataPoint *dataPoints, int N, int DIM) {
    FILE *file = fopen("data.dat", "r");
    if (!file) {
        //printf("Failed to open data file.\n");
        return;
    }
    //printf("Verify that N: %d, DIM: %d\n", N, DIM);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DIM; j++) {
            if (fscanf(file, "%lf", &dataPoints[i].coords[j]) != 1) {
                //printf("Error reading data for point %d, dimension %d.\n", i, j);
            }
        }
    }

    fclose(file);
}
int krDelta(int a, int b) {
    return a == b ? 1 : 0;
}
void calcCardinalities(DataPoint *dataPoints, int N, int K) {
    int *cardinalities = (int *)malloc(K * sizeof(int));
    for (int k = 0; k < K; k++) {
        cardinalities[k] = 0;
        for (int i = 0; i < N; i++) {
            // Directly applying krDelta for conceptual clarity
            cardinalities[k] += krDelta(k, dataPoints[i].cluster);
        }
    }
    for (int k = 0; k < K; k++) {
        printf("Cluster %d has %d data points.\n", k, cardinalities[k]);
    }
}
void calcCentroids(DataPoint *dataPoints, int N, int K, int DIM, Centroid *centroids) {
    int *cardinalities = (int *)malloc(K * sizeof(int));
    // Initialize centroid coordinates and cardinalities
    for (int k = 0; k < K; k++) {
        centroids[k].coords = (double *)malloc(DIM * sizeof(double));
        cardinalities[k] = 0;
        for (int j = 0; j < DIM; j++) {
            centroids[k].coords[j] = 0.0; // Initialize sum to 0
        }
    }

    // Sum coordinates for each cluster and count cardinalities
    for (int i = 0; i < N; i++) {
        int clusterIndex = dataPoints[i].cluster;
        cardinalities[clusterIndex]++;
        for (int j = 0; j < DIM; j++) {
            centroids[clusterIndex].coords[j] += dataPoints[i].coords[j];
        }
    }

    // Divide by cardinalities to get the mean (centroid) for each cluster
    for (int k = 0; k < K; k++) {
        printf("Cluster %d cardinality: %d\n", k, cardinalities[k]);
        if (cardinalities[k] != 0) { // Avoid division by zero
            for (int j = 0; j < DIM; j++) {
                centroids[k].coords[j] /= cardinalities[k];
            }
        }
        else {
            int randomIndex = rand() % N;
            for (int j = 0; j < DIM; j++) {
            centroids[k].coords[j] = 0.1*dataPoints[randomIndex].coords[j];
            }
    }
}
    printf("Updated Centroid Coordinates:\n");
    for (int k = 0; k < K; k++) {
        printf("Centroid %d: (", k);
        for (int j = 0; j < DIM; j++) {
            printf("%.3f", centroids[k].coords[j]);
            if (j < DIM - 1) printf(", ");
        }
        printf(")\n");
    }
    free(cardinalities);
}
double calcDistance(const double *pointCoords, const double *centroidCoords, int DIM) {
    double sum = 0.0;
    for (int i = 0; i < DIM; i++) {
        double diff = pointCoords[i] - centroidCoords[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}
void printClusterInfo(DataPoint *dataPoints, int N, int K, int DIM, Centroid *centroids) {
    int *cardinalities = (int *)calloc(K, sizeof(int)); // Use calloc to initialize to 0
    
    // Count the number of points in each cluster
    for (int i = 0; i < N; i++) {
        cardinalities[dataPoints[i].cluster]++;
    }
    
    // Print the information
    for (int k = 0; k < K; k++) {
        printf("(%d of %d) points are in the cluster %d with centroid (", cardinalities[k], N, k);
        for (int j = 0; j < DIM; j++) {
            printf(" %.2f", centroids[k].coords[j]);
            if (j < DIM - 1) printf(",");
        }
        printf(" )\n");
    }
    
    free(cardinalities);
}
void printClusterMembers(DataPoint *dataPoints, int N, int K, int DIM) {
    FILE *file = fopen("output.dat", "w"); // w for write r for read!!!
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < N; i++) { 
            if (dataPoints[i].cluster == k) { 
                fprintf(file, "%d", k); 
                for (int j = 0; j < DIM; j++) { 
                    fprintf(file, " %.4f", dataPoints[i].coords[j]);
                }
                fprintf(file, "\n");
            }
        }

        fprintf(file, "\n");
    }
}
