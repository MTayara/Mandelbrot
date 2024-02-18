#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));

    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE *pgmimg;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", image[i][j]);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char *argv[]) {
    int image[HEIGHT][WIDTH];
    double AVG = 0;
    int N = 10; 
    double total_time[N];
    struct complex c;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = HEIGHT / size;
    int start_row = rank * rows_per_process;
    int end_row = start_row + rows_per_process;

    if (rank == size - 1) {
        end_row = HEIGHT;
    }

    for (int k = 0; k < N; k++) {
        MPI_Barrier(MPI_COMM_WORLD);
        clock_t start_time = clock();
        #pragma omp parallel for schedule(static)
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[i][j] = cal_pixel(c);
            }


        }

        clock_t end_time = clock(); 
        total_time[k] = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Execution time of trial [%d] on process %d: %f seconds\n", k, rank, total_time[k]);
        AVG += total_time[k];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(image[start_row], rows_per_process * WIDTH, MPI_INT, image, rows_per_process * WIDTH, MPI_INT, 0, MPI_COMM_WORLD);
    double global_AVG = 0;
    MPI_Reduce(&AVG, &global_AVG, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);




    if (rank == 0) {
        printf("The average execution time of %d trials is: %f ms\n", N * size, global_AVG / (N * size) * 1000);
    }
    save_pgm("mandelbrot_static.pgm", image);
    MPI_Finalize();
    return 0;


    
}
