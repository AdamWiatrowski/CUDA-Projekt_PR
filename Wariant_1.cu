#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>

#define N 1026 // rozmiar macierzy;
#define R 33 // promien sumowania
#define K 1 // ilosc elementow obliczana przez kazdy watek

#define OUT_SIZE (N - 2 * R) // rozmiar tablicy wyjsciowej
#define BLOCK_SIZE 32 // rozmiar bloku
#define GRID_SIZE ((OUT_SIZE/BLOCK_SIZE)*(OUT_SIZE/(K*BLOCK_SIZE)))
#define SIZE_X (BLOCK_SIZE + 2*R)
#define SIZE_Y (K*BLOCK_SIZE + 2*R)

__global__ void calculate(int* out, int* tab, int N_k, int R_k, int K_k) {

    __shared__ int shared_tab[SIZE_X][SIZE_Y];

    int row = threadIdx.y;
    int col = K_k * threadIdx.x;

    // zmienne potrzebne do przesunięcia
    int in_row = (OUT_SIZE / (K_k * BLOCK_SIZE));
    int x = blockIdx.x;
    int shift_col = (x % (OUT_SIZE / (K_k * BLOCK_SIZE))) * (K_k * BLOCK_SIZE);
    int shift_row = floorf(x / in_row) * BLOCK_SIZE;

    // kopiowanie z pamieci globalnej do pamieci wspoldzielonej (jeden watek)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int row = 0; row < SIZE_X; row++) {
            for (int col = 0; col < SIZE_Y; col++) {
                // pobieramy odpowiednie dane w z przesunieciem
                shared_tab[row][col] = tab[(row + shift_row) * N_k + col + shift_col];
            }
        }
    }

    // synchronizacja watkow
    __syncthreads();


    // obliczenia
    for (int offset = 0; offset < K_k; offset++) {
        int i = R_k + row;
        int j = R_k + col + offset;
        int sum = 0;
        for (int k = i - R_k; k <= i + R_k; k++) {
            for (int l = j - R_k; l <= j + R_k; l++) {
                sum += shared_tab[k][l];
            }
        }
        out[(row + shift_row) * OUT_SIZE + (col + shift_col + offset)] = sum;

    }

}

// wyliczanie wynikow na cpu = sekwencyjnie;
void calculate_cpu(int* out, int* tab) {
    for (int row = 0; row < OUT_SIZE; row++) {
        for (int col = 0; col < OUT_SIZE; col++) {
            int i = R + row;
            int j = R + col;
            int sum = 0;
            for (int k = i - R; k <= i + R; k++) {
                for (int l = j - R; l <= j + R; l++) {
                    sum += tab[k * N + l];
                }
            }
            out[row * OUT_SIZE + col] = sum;
        }
    }
}

// porownanie dwoch macierzy
void compare(int* a, int* b, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (a[i * size + j] != b[i * size + j]) {
                printf("Equal = False\n");
                return;
            }
        }
    }
    printf("Equal = True\n");
}

// wypelnienie macierzy losowymi danymi
void randomize(int* array, int size) {
    srand(time(0));
    for (int i = 0; i < size * size; i++) {
        array[i] = rand() % 3 + 1;
    }
}

// wypisanie tablicy
void printArray(int* array, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", array[i * size + j]);
        }
        printf("\n");
    }
}

int main() {

    int N_kernel = N;
    int R_kernel = R;
    int K_kernel = K;


    static int tab[N][N];
    static int out[OUT_SIZE][OUT_SIZE];
    static int out_cpu[OUT_SIZE][OUT_SIZE];

    randomize((int*)tab, N);

    //printArray((int*)tab, N);

    int* dev_tab, * dev_out;


    cudaMalloc((void**)&dev_tab, N * N * sizeof(int));
    cudaMalloc((void**)&dev_out, OUT_SIZE * OUT_SIZE * sizeof(int));


    // synchroniczne kopiowanie z CPU do GPU - wariant 3.
    cudaMemcpy(dev_tab, tab, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    //dim3 dimGrid(GRID_SIZE);

    calculate << <GRID_SIZE, dimBlock >> > (dev_out, dev_tab, N_kernel, R_kernel, K_kernel);


    // synchroniczne kopiowanie z GPU do CPU - wariant 3.
    cudaMemcpy(out, dev_out, OUT_SIZE * OUT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);


    //printf("\nGPU\n");
    //printArray((int*)out, OUT_SIZE);

    // Obliczenia na CPU
    calculate_cpu((int*)out_cpu, (int*)tab);

    //printf("\nCPU\n");
    //printArray((int*)out_cpu, OUT_SIZE);

    // porownanie tablic
    compare((int*)out, (int*)out_cpu, OUT_SIZE);

    cudaFree(dev_tab);
    cudaFree(dev_out);

    return 0;
}