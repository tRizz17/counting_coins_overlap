#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
// CUDA runtime
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA

// Dimensions of a rectangular block of threads.  CUDA blocks will be launched with THREAD_GRID_XxTHREAD_GRID_Y threads
#define THREAD_GRID_X 16
#define THREAD_GRID_Y 16

// Dimensions of a rectangular block of pixels to calculate the SAD values.
#define SAD_SIZE_X (THREAD_GRID_X)
#define SAD_SIZE_Y (THREAD_GRID_Y)

// Enable the bounding box image feature for debugging purposes.
// Will make your program slower
// #define BOUNDS_IMAGE

// Output data in CSV format
// #define CSV

// Number of times to run to measure peak performance
#define NUMTRIES 20

// A coin is 'found' if the SAD calculation is < than this number
#define CORRELATION_THRESHOLD (1 * SAD_SIZE_X * SAD_SIZE_Y)

// Decoration to help indicate which Kernel parameter is in or out.
#define IN
#define OUT

// GPU implementation buffers
// Device buffers
unsigned char *device_image, *device_bounds_image, *device_coin;
unsigned int *device_count_per_block;

// Host buffers
unsigned int *count_per_block;

void CudaCheckError(const char *file, unsigned int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        fprintf(stderr, "Cuda failure %s:%d: '%s'\n", file, line, cudaGetErrorString(e));
    }
}

// Function to load a PGM image from a file
unsigned char *load_pgm_image(const char *filename, int *width, int *height)
{
    unsigned char *image;
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        perror("Error opening image file for reading");
        return NULL;
    }

    char header[100];
    fscanf(fp, "%s", header);
    if (strcmp(header, "P5") != 0)
    {
        fprintf(stderr, "Error: Unsupported image format (expected P5)\n");
        fclose(fp);
        return NULL;
    }

    fscanf(fp, "%d %d", width, height);
#ifndef CSV
    fprintf(stderr, "Source image: %s : %d x %d\n", filename, *width, *height);
#endif
    int max_value;
    fscanf(fp, "%d", &max_value);

    if (max_value != 255)
    {
        fprintf(stderr, "Error: Unsupported maximum pixel value (expected 255)\n");
        fclose(fp);
        return NULL;
    }

cudaError_t status = cudaMallocHost((void **)&image, *width * *height * sizeof(unsigned char));
if (status != cudaSuccess)
    printf("Error allocating pinned host memory\n");
CudaCheckError(__FILE__, __LINE__);

    if (image == NULL)
    {
        perror("Error allocating memory for image");
        fclose(fp);
        return NULL;
    }

    // read rest of bytes into the image buffer
    //..
    fread(image, sizeof(unsigned char), *width * *height, fp);

    fclose(fp);

    return image;
}

__device__ void draw_bounding_box_device(unsigned char *image, int width, int height, int x, int y, int width_box, int height_box, unsigned char grey_val)
{
    for (int i = y; i < (y + height_box); i += 4)
    {
        for (int j = x; j < (x + width_box); j += 4)
        {
            if (i >= 0 && i < height && j >= 0 && j < width)
            {
                image[i * width + j] = grey_val;
            }
        }
    }
}

// Function to save a PGM image to a file
void save_pgm_image(const char *filename, unsigned char *image, int width, int height)
{
#ifndef CSV
    fprintf(stderr, "Destination image: %s, %d x %d\n", filename, width, height);
#endif
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        perror("Error opening image file for saving");
        return;
    }

    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    // write the rest of bytes out to the file
    fwrite(image, sizeof(unsigned char), width * height, fp);

    fclose(fp);
}

__global__ void
device_count_coins(IN unsigned char *device_image,
                   IN unsigned char *device_coin,
                   OUT unsigned char *device_bounds_image,
                   IN unsigned int image_height,
                   IN unsigned int image_width,
                   IN unsigned int coin_height,
                   IN unsigned int coin_width,
                   OUT unsigned int *device_count_per_block)
{
    unsigned int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ unsigned short SAD_block[SAD_SIZE_X * SAD_SIZE_Y];

    int block_corner_x = blockIdx.x * coin_width;
    int block_corner_y = blockIdx.y * coin_height;

    int center_of_coin_x = coin_width / 2;
    int center_of_coin_y = coin_height / 2;

    int sad_corner_img_x = block_corner_x + (center_of_coin_x - (SAD_SIZE_X / 2));
    int sad_corner_img_y = block_corner_y + (center_of_coin_y - (SAD_SIZE_X / 2));

    int image_x = threadIdx.x + sad_corner_img_x;
    int image_y = threadIdx.y + sad_corner_img_y;

    int sad_corner_coin_x = center_of_coin_x - (SAD_SIZE_X / 2);
    int sad_corner_coin_y = center_of_coin_y - (SAD_SIZE_X / 2);

    int coin_x = threadIdx.x + sad_corner_coin_x;
    int coin_y = threadIdx.y + sad_corner_coin_y;

    unsigned int difference = abs(device_image[image_y * image_width + image_x] - device_coin[coin_y * coin_width + coin_x]);

    SAD_block[thread_id] = difference;

    __syncthreads();

#if defined BOUNDS_IMAGE
    draw_bounding_box_device(device_bounds_image, image_width, image_height, sad_corner_img_x, sad_corner_img_y, SAD_SIZE_X, SAD_SIZE_Y, 255);
#endif

    // Sum up SAD_block array
    for (unsigned int s = 1; s < SAD_SIZE_X * SAD_SIZE_Y; s *= 2)
    {
        int index = 2 * s * thread_id;

        if (index < SAD_SIZE_X * SAD_SIZE_Y)
        {
            SAD_block[index] += SAD_block[index + s];
        }
        __syncthreads();
    }

    if (thread_id == 0)
    {
        // Test if a coin is detected at this location
        if (SAD_block[0] < CORRELATION_THRESHOLD)
        {
            device_count_per_block[block_id] = 1;
        }
        else
        {
            device_count_per_block[block_id] = 0;
        }
    }
}

// Buffer allocation is 'expensive' in time.  Do this once before we start timing for performance
void allocate_device_buffers(unsigned int image_height,
                             unsigned int image_width,
                             unsigned int coin_height,
                             unsigned int coin_width)
{
    unsigned int x_blocks = image_width / coin_width;
    unsigned int y_blocks = image_height / coin_height;
    unsigned int num_blocks = x_blocks * y_blocks;
    size_t size_count = num_blocks * sizeof(unsigned int);

    // allocate one counter slot per CUDA block
    cudaMalloc(&device_count_per_block, (image_height / coin_height) * (image_width / coin_width) * sizeof(unsigned int));

    // allocate image buffers
    cudaMalloc(&device_image, image_height * image_width);

#if defined BOUNDS_IMAGE
    cudaMalloc(&device_bounds_image, image_height * image_width);
#endif
    cudaMalloc(&device_coin, coin_height * coin_width);
    CudaCheckError(__FILE__, __LINE__);
    count_per_block = (unsigned int *)malloc(size_count);

    if (count_per_block == NULL)
    {
        fprintf(stderr, "count_per_block malloc failed\n");
        exit(1);
    }
}

void free_device_buffers()
{
    // clean up device memory:
    cudaFree(device_image);
#if defined BOUNDS_IMAGE
    cudaFree(device_bounds_image);
#endif
    cudaFree(device_coin);
    cudaFree(device_count_per_block);
    CudaCheckError(__FILE__, __LINE__);
}

int count_coins_gpu(unsigned char *image,
                    unsigned char *coin,
                    unsigned char *bounds_image,
                    unsigned int image_height,
                    unsigned int image_width,
                    unsigned int coin_height,
                    unsigned int coin_width)
{
    unsigned int x_blocks = image_width / coin_width;
    unsigned int y_blocks = image_height / coin_height;
    unsigned int num_blocks = x_blocks * y_blocks;
    size_t size_count = num_blocks * sizeof(unsigned int);
    static bool bFirstTime = true;

    if (bFirstTime)
    {
        // copy host memory to the device:
        cudaMemcpy(device_image, image, image_height * image_width, cudaMemcpyHostToDevice);
        bFirstTime = false;
    }

    CudaCheckError(__FILE__, __LINE__);

#if defined BOUNDS_IMAGE
    cudaMemcpy(device_bounds_image, bounds_image, image_height * image_width, cudaMemcpyHostToDevice);
#endif
    cudaMemcpy(device_coin, coin, coin_height * coin_width, cudaMemcpyHostToDevice);

    CudaCheckError(__FILE__, __LINE__);

    // setup the execution parameters:
    dim3 threadsPerBlock(THREAD_GRID_X, THREAD_GRID_Y);
    dim3 numBlocks(x_blocks, y_blocks);

    // wait until all the data is copied over
    cudaDeviceSynchronize();
    CudaCheckError(__FILE__, __LINE__);

    // execute the kernel:
    device_count_coins<<<numBlocks, threadsPerBlock>>>(
        device_image,
        device_coin,
        device_bounds_image,
        image_height,
        image_width,
        coin_height,
        coin_width,
        device_count_per_block);

    cudaDeviceSynchronize();
    CudaCheckError(__FILE__, __LINE__);

    // copy result from the device to the host:
    cudaMemcpy(count_per_block, device_count_per_block, size_count, cudaMemcpyDeviceToHost);
#if defined BOUNDS_IMAGE
    cudaMemcpy(bounds_image, device_bounds_image, image_height * image_width, cudaMemcpyDeviceToHost);
#endif
    CudaCheckError(__FILE__, __LINE__);

    // compute the sum :
    int numCoins = 0;
    for (int i = 0; i < num_blocks; i++)
    {
        numCoins += count_per_block[i];
    }

    return numCoins;
}

int main(int argc, char *argv[])
{
    // filenames
    const char *input_filename = "assets/coins.pgm";
    const char *dime_head_template = "assets/dime_head.pgm";
    const char *dime_tail_template = "assets/dime_tail.pgm";

    // Dimensions
    int coin_height, coin_width;
    int image_height, image_width;
    int head_count = 0;
    int tail_count = 0;

    // timers
    double time_start, time_end;
    double perf, max_perf = 0;

    if (argc > 1)
    {
        input_filename = argv[1];
    }
    // Load the coin templates from pgm (greyscale) files.
    unsigned char *coin_head = load_pgm_image(dime_head_template, &coin_width, &coin_height);
    if (coin_head == NULL)
    {
        return 1;
    }

    unsigned char *coin_tail = load_pgm_image(dime_tail_template, &coin_width, &coin_height);
    if (coin_tail == NULL)
    {
        return 1;
    }

    unsigned char *image = load_pgm_image(input_filename, &image_width, &image_height);
    if (image == NULL)
    {
        return 1;
    }

// Allocate bounding box buffer
unsigned char *bounding_box_image = (unsigned char *)malloc(image_width * image_height * sizeof(unsigned char));

// Copy the image into the bounding box buffer
memcpy(bounding_box_image, image, image_width * image_height);

allocate_device_buffers(image_height, image_width, coin_height, coin_width);

#ifndef CSV
    fprintf(stderr, "\nUsing GPU with %dx%d threads per block and a grid of %dx%d blocks.\n\n", THREAD_GRID_X, THREAD_GRID_Y, image_width / coin_width, image_height / coin_height);
#endif

    for (int i = 0; i < NUMTRIES; i++)
    {

        time_start = omp_get_wtime();


    head_count = count_coins_gpu(image, coin_head, bounding_box_image, image_height, image_width, coin_height, coin_width);
    tail_count = count_coins_gpu(image, coin_tail, bounding_box_image, image_height, image_width, coin_height, coin_width);


    time_end = omp_get_wtime();
    perf = (double)(image_width * image_height) / (time_end - time_start) / 1000000.;
    if (max_perf < perf)
        max_perf = perf;
    }

    float percent_heads = ((float)head_count) / ((float)(head_count + tail_count)) * 100;
#ifdef CSV
    unsigned int image_size = image_width * image_height;
    fprintf(stderr, "%lu, %.2f, %.2lf\n", image_size, max_perf, percent_heads);
#else
    fprintf(stderr, "Number of heads: %d\n", head_count);
    fprintf(stderr, "Number of tails: %d\n", tail_count);
    fprintf(stderr, "Total coins: %d\n", head_count + tail_count);
    fprintf(stderr, "Percent heads: %.2f\n", percent_heads);
    fprintf(stderr, "Performance in Mpixels/sec: %.2lf\n", max_perf);
#endif

#ifdef BOUNDS_IMAGE
    // Save the bounding box image
    save_pgm_image("bounding-box.pgm", bounding_box_image, image_width, image_height);

#endif


free_device_buffers();
cudaFreeHost(image);
cudaFreeHost(coin_head);
cudaFreeHost(coin_tail);
free(bounding_box_image);
free(count_per_block);
return 0;
}
