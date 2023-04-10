#include <stdio.h>
#include "support.h"
#include "kernel.cu"

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
	
    unsigned int m;
    unsigned int k;
    unsigned int n;
    if(argc == 4) {
        m = atoi(argv[1]);
	k = atoi(argv[2]);
	n = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }

    float* A_h = (float*) malloc( m*k*sizeof(float));
    for (unsigned int i=0; i < m; i++)
	{
    		for (unsigned int j=0; j<k; j++)
	 		{ A_h[i*k+j] = (rand()%100)/100.00; }
	}

    float* B_h = (float*) malloc( k*n*sizeof(float));
    for (unsigned int i=0; i < k; i++) { 
		for(unsigned int j =0; j<n; j++)
		{
		B_h[i*n+j] = (rand()%100)/100.00; 
		}
	}

    float* C_h = (float*) malloc( m*n*sizeof(float) );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u,%u\n",m,n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    float* A_d;
    cuda_ret = cudaMalloc((void**) &A_d, m*k*sizeof(float));
	if(cuda_ret != cudaSuccess)
	{
	printf("CUDA error: %s\n", cudaGetErrorString(cuda_ret));
	 FATAL("Unable to allocate device memory");
	printf("CUDA error: %s\n", cudaGetErrorString(cuda_ret));
	}

    //INSERT CODE HERE for B and C
    float* B_d;
    cuda_ret = cudaMalloc((void**) &B_d, k*n*sizeof(float));
    	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    float* C_d;
    cuda_ret = cudaMalloc((void**) &C_d,m*n* sizeof(float));
    	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(A_d, A_h, (m*k)*sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    //INSERT CODE HERE for B
    cuda_ret = cudaMemcpy(B_d, B_h, (k*n)*sizeof(float), cudaMemcpyHostToDevice);
    	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device"); 
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    //const unsigned int THREADS_PER_BLOCK = 512;
    //const unsigned int numBlocks = (m - 1)/THREADS_PER_BLOCK + 1;
    //dim3 gridDim(numBlocks, numBlocks, 1), blockDim(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
    
    dim3 threadsPerBlock(m,n);
    dim3 blocksPerGrid(1,1);
    if(m > 512 || n>512){
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil( double(m)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(n)/double(threadsPerBlock.y));
     }

    //INSERT CODE HERE to call kernel
    vecMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, m, k, n);
    //matrixMultiplication(A_d, B_d, C_d, m, n);
    printf("Ended kernel");

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE to copy C
    cuda_ret = cudaMemcpy(C_h, C_d, m*n*sizeof(float), cudaMemcpyDeviceToHost);

    if(cuda_ret!=cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // print results -----------------------------------------------------
    printf("After copying...");
    for(int i1=0; i1<m;i1++)
	{
	for(int j1=0; j1<k; j1++)
	{
	printf("%f ",A_h[i1*k+j1]);
	}
	printf("\n");
	}
    printf("\n\n\n\n");
    for(int i1=0; i1<k;i1++)
        {
        for(int j1=0; j1<n; j1++)
        {
        printf("%f ",B_h[i1*n+j1]);
        }
        printf("\n");
        }
	printf("\n\n\n");
    for(int i1=0; i1<m;i1++)
        {
        for(int j1=0; j1<n; j1++)
        {
        printf("%f ",C_h[i1*n+j1]);
        }
        printf("\n");
        }



    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE to free device matrices
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;

}

