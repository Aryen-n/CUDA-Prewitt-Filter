#include <thread>
#include <chrono>
#include <time.h>
#include <iostream>
#include <math.h>
#include "imageLoader.cpp"

#define GRIDVAL 20.0 

void prewitt_cpu(const byte* orig, byte* cpu, const unsigned int width, const unsigned int height);

__global__ void prewitt_gpu(const byte* orig, byte* cpu, const unsigned int width, const unsigned int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
        dx = (-1* orig[(y-1)*width + (x-1)]) + (-2*orig[y*width+(x-1)]) + (-1*orig[(y+1)*width+(x-1)]) +
             (    orig[(y-1)*width + (x+1)]) + ( 2*orig[y*width+(x+1)]) + (   orig[(y+1)*width+(x+1)]);
        dy = (    orig[(y-1)*width + (x-1)]) + ( 2*orig[(y-1)*width+x]) + (   orig[(y-1)*width+(x+1)]) +
             (-1* orig[(y+1)*width + (x-1)]) + (-2*orig[(y+1)*width+x]) + (-1*orig[(y+1)*width+(x+1)]);
        cpu[y*width + x] = sqrt( (dx*dx) + (dy*dy) );
    }
}

int main(int argc, char*argv[]) {
    /** Check command line arguments **/
    if(argc != 2) {
        printf("%s: Invalid number of command line arguments. Exiting program\n", argv[0]);
        printf("Usage: %s [image.png]", argv[0]);
        return 1;
    }
    /** Gather CUDA device properties **/
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	int cores = devProp.multiProcessorCount;
	switch (devProp.major)
	{
	case 2: // Fermi
		if (devProp.minor == 1) cores *= 48;
		else cores *= 32; break;
	case 3: // Kepler
		cores *= 192; break;
	case 5: // Maxwell
		cores *= 128; break;
	case 6: // Pascal
		if (devProp.minor == 1) cores *= 128;
		else if (devProp.minor == 0) cores *= 64;
		break;
    }
    
    /** Print out some header information (# of hardware threads, GPU info, etc) **/
    time_t rawTime;time(&rawTime);
    struct tm* curTime = localtime(&rawTime);
    char timeBuffer[80] = "";
    strftime(timeBuffer, 80, "edge map benchmarks (%c)\n", curTime);
    printf("%s", timeBuffer);
    printf("CPU: %d hardware threads\n", std::thread::hardware_concurrency());
    printf("GPGPU: %s, CUDA %d.%d, %zd Mbytes global memory, %d CUDA cores\n",
    devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / 1048576, cores);

    /** Load our img and allocate space for our modified images **/
    imgData origImg = loadImage(argv[1]);
    imgData cpuImg(new byte[origImg.width*origImg.height], origImg.width, origImg.height);
  
    imgData gpuImg(new byte[origImg.width*origImg.height], origImg.width, origImg.height);
    
    /** make sure all our newly allocated data is set to 0 **/
    memset(cpuImg.pixels, 0, (origImg.width*origImg.height));
  

    /** We first run the prewitt filter on just the CPU using only 1 thread **/
    auto c = std::chrono::system_clock::now();
    prewitt_cpu(origImg.pixels, cpuImg.pixels, origImg.width, origImg.height);
    std::chrono::duration<double> time_cpu = std::chrono::system_clock::now() - c;

    /** Next, we use OpenMP to parallelize it **/
    c = std::chrono::system_clock::now();
  
 

    /** Finally, we use the GPU to parallelize it further **/
    /** Allocate space in the GPU for our original img, new img, and dimensions **/
    byte *gpu_orig, *gpu_prewitt;
    cudaMalloc( (void**)&gpu_orig, (origImg.width * origImg.height));
    cudaMalloc( (void**)&gpu_prewitt, (origImg.width * origImg.height));
    /** Transfer over the memory from host to device and memset the prewitt array to 0s **/
    cudaMemcpy(gpu_orig, origImg.pixels, (origImg.width*origImg.height), cudaMemcpyHostToDevice);
    cudaMemset(gpu_prewitt, 0, (origImg.width*origImg.height));
   
    /** set up the dim3's for the gpu to use as arguments (threads per block & num of blocks)**/
    dim3 threadsPerBlock(GRIDVAL, GRIDVAL, 1);
    dim3 numBlocks(ceil(origImg.width/GRIDVAL), ceil(origImg.height/GRIDVAL), 1);

    /** Run the prewitt filter using the CPU **/
    c = std::chrono::system_clock::now();
    prewitt_gpu<<<numBlocks, threadsPerBlock>>>(gpu_orig, gpu_prewitt, origImg.width, origImg.height);
    cudaError_t cudaerror = cudaDeviceSynchronize(); // waits for completion, returns error code
    if ( cudaerror != cudaSuccess ) fprintf( stderr, "Cuda failed to synchronize: %s\n", cudaGetErrorName( cudaerror ) ); // if error, output error
    std::chrono::duration<double> time_gpu = std::chrono::system_clock::now() - c;
    /** Copy data back to CPU from GPU **/
    cudaMemcpy(gpuImg.pixels, gpu_prewitt, (origImg.width*origImg.height), cudaMemcpyDeviceToHost);

    /** Output runtimes of each method of prewitt filtering **/
    printf("\nProcessing %s: %d rows x %d columns\n", argv[1], origImg.height, origImg.width);
    printf("CPU execution time    = %*.1f msec\n", 5, 1000*time_cpu.count());

    printf("CUDA execution time   = %*.1f msec\n", 5, 1000*time_gpu.count());

    printf("\nCPU->GPU speedup:%*.1f X", 12, (1000*time_cpu.count())/(1000*time_gpu.count()));
    printf("\n");

    /** Output the images of each prewitt filter with an appropriate string appended to the original image name **/
    writeImage(argv[1], "gpu", gpuImg);
    writeImage(argv[1], "cpu", cpuImg);


    /** Free any memory leftover.. gpuImig, cpuImg get their pixels free'd while writing **/
    cudaFree(gpu_orig); cudaFree(gpu_prewitt);
    return 0;
}

void prewitt_cpu(const byte* orig, byte* cpu, const unsigned int width, const unsigned int height) {
    for(int y = 1; y < height-1; y++) {
        for(int x = 1; x < width-1; x++) {
            int dx = (-1*orig[(y-1)*width + (x-1)]) + (-2*orig[y*width+(x-1)]) + (-1*orig[(y+1)*width+(x-1)]) +
                 (orig[(y-1)*width + (x+1)]) + (2*orig[y*width+(x+1)]) + (orig[(y+1)*width+(x+1)]);
            int dy = (orig[(y-1)*width + (x-1)]) + (2*orig[(y-1)*width+x]) + (orig[(y-1)*width+(x+1)]) +
            (-1*orig[(y+1)*width + (x-1)]) + (-2*orig[(y+1)*width+x]) + (-1*orig[(y+1)*width+(x+1)]);
            cpu[y*width + x] = sqrt((dx*dx)+(dy*dy));
        }
    }
}


