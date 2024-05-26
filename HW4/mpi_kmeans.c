#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

int Nd, Nc, Np_all, rank, size, Np_process;
double TOL;

#define BUFSIZE 512
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX_ITER 10000

double readInputFile(char *fileName, char* tag) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("Error opening the input file\n");
        exit(EXIT_FAILURE);
    }

    double result;
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

    printf("ERROR! Could not find the tag: [%s] in the file [%s]\n", tag, fileName);
    fclose(fp);
    exit(EXIT_FAILURE);
    return -1;  // Unreachable, but suppresses compiler warning
}

void readDataFile(char *fileName, double *data) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("Error opening the input file\n");
        exit(EXIT_FAILURE);
    }

    int sk = 0;
    char buffer[BUFSIZE];
    int shift = Nd;

    while (fgets(buffer, BUFSIZE, fp) != NULL) {
        if (Nd == 2) sscanf(buffer, "%lf %lf", &data[sk * shift + 0], &data[sk * shift + 1]);
        if (Nd == 3) sscanf(buffer, "%lf %lf %lf", &data[sk * shift + 0], &data[sk * shift + 1], &data[sk * shift + 2]);
        if (Nd == 4) sscanf(buffer, "%lf %lf %lf %lf", &data[sk * shift + 0], &data[sk * shift + 1], &data[sk * shift + 2], &data[sk * shift + 3]);
        sk++;
    }
    fclose(fp);
}

void writeDataToFile(char *fileName, double *data, int *Ci) {
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening the output file\n");
        exit(EXIT_FAILURE);
    }

    for (int p = 0; p < Np_all; p++) {
        fprintf(fp, "%d %d ", p, Ci[p]);
        for (int dim = 0; dim < Nd; dim++) {
            fprintf(fp, "%.4f ", data[p * Nd + dim]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void writeCentroidToFile(char *fileName, double *Cm) {
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening the output file\n");
        exit(EXIT_FAILURE);
    }

    for (int n = 0; n < Nc; n++) {
        for (int dim = 0; dim < Nd; dim++) {
            fprintf(fp, "%.4f ", Cm[n * Nd + dim]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

double distance(double *a, double *b) {
    double sum = 0.0;
    for (int dim = 0; dim < Nd; dim++) {
        sum += pow((a[dim] - b[dim]), 2);
    }
    return sqrt(sum);
}

void assignPoints(double *data, int *Ci, int *Ck, double *Cm) {
    for (int n = 0; n < Nc; n++) {
        Ck[n] = 0;
    }
    for (int p = 0; p < Np_process; p++) {
    if(rank==size-1 && p+Np_process*rank==Np_all){
        break;
    }
    double min_distance = INFINITY;
    int cluster_index = 0;
    int pAll = rank * Np_process + p;
    
    for (int n = 0; n < Nc; n++) {
        double d = distance(&data[pAll * Nd], &Cm[n * Nd]);
        if (d < min_distance) {
            min_distance = d;
            cluster_index = n;
        }
    }

        Ck[cluster_index]++;
        Ci[p] = cluster_index;

    }
}

double updateCentroids(double *data, int *Ci, int *Ck, double *Cm) {
    double *CmCopy = (double *)malloc(Nc * Nd * sizeof(double));
    memcpy(CmCopy, Cm, Nc * Nd * sizeof(double));

    for (int n = 0; n < Nc; n++) {
        for (int dim = 0; dim < Nd; dim++) {
            Cm[n * Nd + dim] = 0.0;
        }
    }

    for (int p = 0; p < Np_all; p++) {
        int cluster_index = Ci[p];

        for (int dim = 0; dim < Nd; dim++) {
            Cm[cluster_index * Nd + dim] += data[p * Nd + dim];
        }
    }

    double err = 0.0;
    for (int n = 0; n < Nc; n++) {
        if (Ck[n] > 0) {
            for (int dim = 0; dim < Nd; dim++) {
                Cm[n * Nd + dim] /= Ck[n];
            }
        }

        for (int dim = 0; dim < Nd; dim++) {
            err = MAX(err, fabs(Cm[n * Nd + dim] - CmCopy[n * Nd + dim]));
        }
    }

    free(CmCopy);
    return err;
}

void kMeans(double *data, int *Ci, int *Ck, double *Cm, int *Ci_all, int *Ck_all) {
    if (rank == 0) {
        srand(time(NULL));
        for (int n = 0; n < Nc; n++) {
            int ids = rand() % Np_all;
            for (int dim = 0; dim < Nd; dim++) {
                Cm[n * Nd + dim] = data[ids * Nd + dim];
            }
            Ck_all[n] = 0;
            Ci_all[ids] = n;
        }
    }

    MPI_Scatter(Ci_all, Np_process, MPI_INT, Ci, Np_process, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(Cm, Nc * Nd, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double err = INFINITY;
    int sk = 0;
    while (err > TOL && sk < MAX_ITER) {
        assignPoints(data, Ci, Ck, Cm);

        MPI_Reduce(Ck, Ck_all, Nc, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Allgather(Ci, Np_process, MPI_INT, Ci_all, Np_process, MPI_INT,MPI_COMM_WORLD);
  
        if (rank == 0) {
            err = updateCentroids(data, Ci_all, Ck_all, Cm);
            printf("\rIteration %d, Error: %.12e \n", sk, err);
            
            fflush(stdout);
        }
        sk++;
        MPI_Barrier;
        MPI_Bcast(Cm, Nc * Nd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    double walltime = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) printf("Usage: ./kmeans input.dat data.dat\n");
        MPI_Finalize();
        return -1;
    }

    
    Np_all = (int) readInputFile(argv[1], "NUMBER_OF_POINTS");
    Nc = (int) readInputFile(argv[1], "NUMBER_OF_CLUSTERS");
    Nd = (int) readInputFile(argv[1], "DATA_DIMENSION");
    TOL = readInputFile(argv[1], "TOLERANCE");
   
    
    Np_process = (Np_all + size - 1) / size;  // Ensure every process gets at least one point

    double *data = (double*) malloc(Np_all * Nd * sizeof(double));
    readDataFile(argv[2], data);

    int *Ci = (int *) calloc(Np_process, sizeof(int));
    int *Ck = (int *) calloc(Nc, sizeof(int));
    double *Cm = (double*) calloc(Nc * Nd, sizeof(double));

    int *Ci_all = (int*) calloc(Np_all, sizeof(int));
    int *Ck_all = (int*) calloc(Nc, sizeof(int));

    MPI_Barrier;
    kMeans(data, Ci, Ck, Cm, Ci_all, Ck_all);
    walltime = MPI_Wtime() - walltime;

    double walltime_max;
    MPI_Reduce(&walltime, &walltime_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total Compute Time: %f using %d processors\n", walltime_max, size);
        for (int n = 0; n < Nc; n++) {
            int Npoints = Ck_all[n];
            printf("(%d of %d) points are in the cluster %d with centroid( ", Npoints, Np_all, n);
            for (int dim = 0; dim < Nd; dim++) {
                printf("%f ", Cm[n * Nd + dim]);
            }
            printf(")\n");
        }

        writeDataToFile("output.dat", data, Ci_all);
        writeCentroidToFile("centroids.dat", Cm);
    }

    free(Ci_all);free(data);free(Ci);free(Ck);free(Cm);
    MPI_Finalize();
    
    return 0;
}
