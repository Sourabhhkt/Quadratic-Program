#include <stdio.h>
#include "support.h"
#include "kernel.cu"
#include <float.h>

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;
    // Setting up input parameter for QP
    // W = np.array([
    //     [6, 3, 5, 0],
    //     [3, 6, 0, 1],
    //     [5, 0, 8, 0],
    //     [0, 1, 0, 10] 
    // ])
    // A = np.array([
    //     [3, -3, -2, 1],
    //     [4, 1, -1, -2] 
    // ])
    // E = np.array([
    //     [-1, 1, 0, 0],
    //     [3, 0, 1, 0] 
    // ])
    // C = np.array([-11,0,0,-5])
    // b = np.array([0,0])
    // l = np.array([-np.inf,-2])
    // h = np.array([-1,4])
    
    // W_inv = np.linalg.inv(W)

    //     ME_T = 
    //  [[-0.01105294  0.07155323]
    //  [ 0.03548575 -0.01919721]
    //  [-0.05759162  0.16230366]
    //  [ 0.02443281  0.05235602]]
    // s = 
    //  [0.26236184 0.26294357 0.2617801  0.52530541]

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
    
    int row_num = 4;
    int col_num = 2;

    // Constant parameter from the problem: l and h
    float* l; float* h; 
    float raw_l = {FLT_MIN, -2};float raw_h = {-1, 4};
    l = raw_l; h = raw_h;

    // Constant parameter from the problem: ME^T
    float** E = (float**) malloc( sizeof(float*)*col_num );
    for (int i = 0; i < row_num; i++){
        *(E+i) = (float*)malloc(sizeof(float)*row_num);
    }

    E[0][0] =  -1; E[0][1] =  1; E[0][2] =  0; E[0][3] =  0;
    E[1][0] =  3; E[1][1] =  0; E[1][2] =  1; E[1][3] =  0;
    printf("Matrix E: \n"); fflush(stdout);
    print_2d_array(row_num,col_num,E);

    // Constant parameter from the problem: ME^T
    float** ME_T = (float**) malloc( sizeof(float*)*row_num );
    for (int i = 0; i < row_num; i++){
        *(ME_T+i) = (float*)malloc(sizeof(float)*col_num);
    }

    ME_T[0][0] =  -0.01105294; ME_T[0][1] =  0.07155323;
    ME_T[1][0] =  0.03548575; ME_T[1][1] =  -0.01919721; 
    ME_T[2][0] =  -0.05759162; ME_T[2][1] =  0.16230366; 
    ME_T[3][0] =  0.02443281; ME_T[3][1] =  0.05235602; 
    printf("Matrix ME^T: \n"); fflush(stdout);
    print_2d_array(row_num,col_num,ME_T);

    // Constant parameter from the problem: s
    float raw_vecS[] = {0.26236184, 0.26294357, 0.2617801, 0.52530541};
    float* s = raw_vecS;
    
    printf("Vector s: \n"); fflush(stdout);
    print_1d_array(row_num,s);

    // Initialize u and x variable
    float raw_u[] = {10, -10};float raw_u_minus[] = {-10, 10};
    float* u_0 = raw_u;
    float* u_0_minus = raw_u_minus;

    float* x = (float*)malloc(sizeof(float)*row_num);
    x = vec_add_vec(row_num,mat_mul_vec(row_num, col_num, ME_T, u_0),s);
    printf("Vector ME^T*u0 + s: \n"); fflush(stdout);
    print_1d_array(row_num,x);

    // Initialize Ex
    float* Ex = (float*)malloc(sizeof(float)*col_num);
    Ex = mat_mul_vec(col_num,row_num, E, x);
    printf("Vector Ex: \n"); fflush(stdout);
    print_1d_array(col_num,Ex);

    // Initialize g_Ex_u
    float* g_Ex_u = (float*)malloc(sizeof(float)*row_num);
    g_Ex_u = g_function(row_num, vec_add_vec(row_num,Ex,u_0_minus))
    printf("Vector g_Ex_u: \n"); fflush(stdout);
    print_1d_array(col_num,g_Ex_u);
    // bool tolerance_met = false;
    // while (!tolerance_met)
    // {

    // }

    // Allocate device variables ----------------------------------------------

    // printf("Allocating device variables..."); fflush(stdout);
    // startTime(&timer);

    // float* u_d;
    // cuda_ret = cudaMalloc((void**) &u_d, sizeof(float)*col_num);
	// if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    // float* u_prev_d;
    // cuda_ret = cudaMalloc((void**) &u_prev_d, sizeof(float)*col_num);
	// if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");




    // //INSERT CODE HERE for B and C
    // float* B_d;
    // cuda_ret = cudaMalloc((void**) &B_d, sizeof(float)*n);
	// if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    // float* C_d;
    // cuda_ret = cudaMalloc((void**) &C_d, sizeof(float)*n);
	// if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    // cudaDeviceSynchronize();




    // float* A_h = (float*) malloc( sizeof(float)*n );
    // for (unsigned int i=0; i < n; i++) { A_h[i] = (rand()%100)/100.00; }

    // float* B_h = (float*) malloc( sizeof(float)*n );
    // for (unsigned int i=0; i < n; i++) { B_h[i] = (rand()%100)/100.00; }

    // float* C_h = (float*) malloc( sizeof(float)*n );

    // stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    // printf("    Vector size = %u\n", n);

    // // Allocate device variables ----------------------------------------------

    // printf("Allocating device variables..."); fflush(stdout);
    // startTime(&timer);

    // float* A_d;
    // cuda_ret = cudaMalloc((void**) &A_d, sizeof(float)*n);
	// if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    // //INSERT CODE HERE for B and C
    // float* B_d;
    // cuda_ret = cudaMalloc((void**) &B_d, sizeof(float)*n);
	// if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    // float* C_d;
    // cuda_ret = cudaMalloc((void**) &C_d, sizeof(float)*n);
	// if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    // cudaDeviceSynchronize();
    // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // // Copy host variables to device ------------------------------------------

    // printf("Copying data from host to device..."); fflush(stdout);
    // startTime(&timer);

    // cuda_ret = cudaMemcpy(A_d, A_h, sizeof(float)*n, cudaMemcpyHostToDevice);
	// if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    // //INSERT CODE HERE for B
    // cuda_ret = cudaMemcpy(B_d, B_h, sizeof(float)*n, cudaMemcpyHostToDevice);
	// if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    // cudaDeviceSynchronize();
    // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // // Launch kernel ----------------------------------------------------------

    // printf("Launching kernel..."); fflush(stdout);
    // startTime(&timer);

    // const unsigned int THREADS_PER_BLOCK = 512;
    // const unsigned int numBlocks = (n - 1)/THREADS_PER_BLOCK + 1;
    // dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);
    // //INSERT CODE HERE to call kernel
    // vecAddKernel<<<ceil(numBlocks),THREADS_PER_BLOCK>>>(A_d, B_d, C_d, n);

    // cuda_ret = cudaDeviceSynchronize();
	// if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // // Copy device variables from host ----------------------------------------

    // printf("Copying data from device to host..."); fflush(stdout);
    // startTime(&timer);

    // //INSERT CODE HERE to copy C
    // cuda_ret = cudaMemcpy(C_h, C_d, sizeof(float)*n, cudaMemcpyDeviceToHost);
	// if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    // cudaDeviceSynchronize();
    // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // // Verify correctness -----------------------------------------------------

    // printf("Verifying results..."); fflush(stdout);

    // verify(A_h, B_h, C_h, n);

    // // Free memory ------------------------------------------------------------

    // free(A_h);
    // free(B_h);
    // free(C_h);

    // //INSERT CODE HERE to free device matrices
    // cudaFree(A_d);
    // cudaFree(B_d);
    // cudaFree(C_d);

    return 0;

}

