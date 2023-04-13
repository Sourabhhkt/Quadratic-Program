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
    float EPSILON = atof(argv[argc-1]);
    printf("EPSILON = %f s\n", EPSILON); fflush(stdout);

    
    int row_num = 4; // this define dim of x
    int col_num = 2;

    // Constant parameter from the problem: l and h
    float* l; float* h; 
    float raw_l[] = {FLT_MIN, -2};
    float raw_h[] = {-1, 4};
    l = raw_l; h = raw_h;

    // Constant parameter from the problem: ME^T
    float** E = (float**) malloc( sizeof(float*)*col_num );
    for (int i = 0; i < col_num; i++){
        *(E+i) = (float*)malloc(sizeof(float)*row_num);
    }
    E[0][0] =  -1; E[0][1] =  1; E[0][2] =  0; E[0][3] =  0;
    E[1][0] =  3; E[1][1] =  0; E[1][2] =  1; E[1][3] =  0;
    printf("Matrix E: \n"); fflush(stdout);
    print_2d_array(col_num,row_num,E);

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
    float* ME_T_h_1d = (float*)malloc(sizeof(float)*row_num*col_num);
    ME_T_h_1d = convert_2d_mat_to_1d_arr(row_num, col_num, ME_T);


    // Constant parameter from the problem: s
    float raw_vecS[] = {0.26236184, 0.26294357, 0.2617801, 0.52530541};
    float* s = raw_vecS;
    
    printf("Vector s: \n"); fflush(stdout);
    print_1d_array(row_num,s);

    // Initialize u and x variable
    float raw_u[] = {10, -10};float raw_u_minus[] = {-10, 10};
    float* u_p_h = raw_u;
    float* u_p_minus = raw_u_minus;

    float* x_h = (float*)malloc(sizeof(float)*row_num);
    float* x_p_h = (float*)malloc(sizeof(float)*row_num);

    // Initialize Ex
    float* Ex = (float*)malloc(sizeof(float)*col_num);
    
    // Initialize g_Ex_u
    float* g_Ex_u = (float*)malloc(sizeof(float)*col_num);
    
    // Initialize u_c_h
    float* u_c_h = (float*)malloc(sizeof(float)*col_num);
    


    bool tolerance_met = false;
    // while (!tolerance_met)
    // {
    for (int iter = 0; iter < 100; iter++)
    {
        printf("Iter: [%d] ========================================\n",iter); fflush(stdout);
        // calculate x_h
        x_h = vec_add_vec(row_num,mat_mul_vec(row_num, col_num, ME_T, u_p_h),s);
        printf("Vector x_h: \n"); fflush(stdout);
        print_1d_array(row_num,x_h);

        Ex = mat_mul_vec(col_num,row_num, E, x_h);
        printf("Vector Ex: \n"); fflush(stdout);
        print_1d_array(col_num,Ex);

        g_Ex_u = g_function(col_num, vec_add_vec(col_num,Ex,u_p_minus),l,h);
        printf("Vector g_Ex_u: \n"); fflush(stdout);
        print_1d_array(col_num,g_Ex_u);


        // u_c_h =vec_add_vec(col_num,u_p_h,scale_vec(col_num, EPSILON, vec_add_vec(col_num,g_Ex_u, scale_vec(col_num,-1, Ex)))
        for (int _idx = 0; _idx< col_num; _idx++)
        {
            u_c_h[_idx] = u_p_h[_idx] + (1/EPSILON)*(g_Ex_u[_idx]-Ex[_idx]);
        }
        printf("Vector u_c_h: \n"); fflush(stdout);
        print_1d_array(col_num,u_c_h);


        // Allocate device variables ----------------------------------------------
        printf("Allocating device variables..."); fflush(stdout);
        startTime(&timer);

        float* x_d;
        cuda_ret = cudaMalloc((void**) &x_d, sizeof(float)*row_num);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate x_d device memory");

        float* u_d;
        cuda_ret = cudaMalloc((void**) &u_d, sizeof(float)*col_num);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate u_d device memory");

        float* ME_T_d; // this should be 2d mat stored in 1d (col-wised might be better)
        cuda_ret = cudaMalloc((void**) &ME_T_d, sizeof(float)*row_num*col_num);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate ME_T device memory");

        float* s_d;
        cuda_ret = cudaMalloc((void**) &s_d, sizeof(float)*col_num);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate s_d device memory");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // Copy host variables to device ------------------------------------------
        printf("Copying data from host to device..."); fflush(stdout);
        startTime(&timer);
        // x_current
        cuda_ret = cudaMemcpy(x_d, x_h, sizeof(float)*row_num, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory x_h to x_d device");
        // u_current
        cuda_ret = cudaMemcpy(u_d, u_c_h, sizeof(float)*col_num, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory u_c_h to u_d device");
        // ME_T_d
        cuda_ret = cudaMemcpy(ME_T_d, ME_T_h_1d, sizeof(float)*row_num*col_num, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory ME_T_h_1d to ME_T_d device");
        // s
        cuda_ret = cudaMemcpy(s_d, s, sizeof(float)*col_num, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory s to s_d device");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // Launch kernel ----------------------------------------------------------
        printf("Launching kernel..."); fflush(stdout);
        startTime(&timer);

        const unsigned int THREADS_PER_BLOCK = 512;
        const unsigned int numBlocks = (row_num - 1)/THREADS_PER_BLOCK + 1;
        dim3 gridDim(numBlocks, 1, 1), blockDim(THREADS_PER_BLOCK, 1, 1);
        //INSERT CODE HERE to call kernel
        sdnnIterationKernel<<<ceil(numBlocks),THREADS_PER_BLOCK>>>(x_d, u_d, ME_T_d, s_d, row_num, col_num);

        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // Copy device variables to host ----------------------------------------
        printf("Copying data from device to host..."); fflush(stdout);
        startTime(&timer);
        // x_current
        cuda_ret = cudaMemcpy(x_h, x_d, sizeof(float)*row_num, cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory back to host");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // Close kernel & free memory
        cudaFree(u_d);
        cudaFree(x_d);

        // Tolerance verification
        printf("Tolerance check..."); fflush(stdout);
        float* gradient = vec_add_vec(col_num, g_Ex_u, scale_vec(col_num,-1, Ex));
        float tol = vec_l1_norm(col_num, gradient);
        if (tol < 0.000001)
        {
            tolerance_met = true;
        }
        printf("Tol = %f \n", tol);

        // Updating rule
        // u_p_h
        u_p_h = u_c_h;
        u_p_minus = scale_vec(col_num,-1,u_p_h);
        x_p_h = x_h;

    }
    // get final solution

    printf("========================================\n"); fflush(stdout);
    printf("Tolerance_met = %s", tolerance_met ? "true" : "false");
    // printf("Optimal obj:%f s\n", );
    printf("Optimal sol X: \n"); print_1d_array(row_num,x_h);

    printf("========================================"); fflush(stdout);

    // }

    return 0;

}

