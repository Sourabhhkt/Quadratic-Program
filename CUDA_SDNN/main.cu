#include <stdio.h>
#include "support.h"
#include "kernel.cu"
#include <float.h>
#include <time.h>

int main(int argc, char**argv) {

    Timer timer;
    clock_t start, end;
    float cpu_time_used;
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
    int row_num = 50; // this define dim of x
    int col_num = 1;


    printf("\nTesting csv reader function.."); fflush(stdout);
    char inst_path[] = "/DataInstance/QPLOB_0018/";
    char inst_name[] = "QPLOB_0018";

    char f_name[] = "example.csv";
    read_W(inst_path,inst_name, row_num, col_num);



    // Initialize host variables ----------------------------------------------
    
    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
    float EPSILON; int ITERATIONLIM; float TIMELIMINSEC;
    if (argc==2){
        EPSILON = atof(argv[1]);
        ITERATIONLIM = 100;
        TIMELIMINSEC = 120; // 2mins
    } else if (argc==3){
        EPSILON = atof(argv[1]);
        ITERATIONLIM = atof(argv[2]);
        TIMELIMINSEC = 120; // 2mins
    } else if (argc==4){
        EPSILON = atof(argv[1]);
        ITERATIONLIM = atof(argv[2]);
        TIMELIMINSEC = atof(argv[3]);
    } else{
        EPSILON = 1;
        ITERATIONLIM = 100;
        TIMELIMINSEC = 120; // 2mins
    }
     
    printf("EPSILON = %f \n", EPSILON); fflush(stdout);
    printf("ITERATIONLIM = %d \n", ITERATIONLIM); fflush(stdout);
    printf("TIMELIMINSEC = %f \n", TIMELIMINSEC); fflush(stdout);

    


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
    printf("Matrix ME^T_1d: \n"); fflush(stdout);
    print_1d_array(row_num*col_num,ME_T_h_1d);

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
    x_p_h = vec_add_vec(row_num,mat_mul_vec(row_num, col_num, ME_T, u_p_h),s);

    float* x_optimal = (float*)malloc(sizeof(float)*row_num);

    // Initialize Ex
    float* Ex = (float*)malloc(sizeof(float)*col_num);
    
    // Initialize g_Ex_u
    float* g_Ex_u = (float*)malloc(sizeof(float)*col_num);
    
    // Initialize u_c_h
    float* u_c_h = (float*)malloc(sizeof(float)*col_num);
    
    int iter_count = 0; float iter_time = 0;float tol = FLT_MAX;

    bool tolerance_met = false;
    bool iteration_lim_met = false;
    bool time_lim_met = false;

    
    while ((!tolerance_met )& (!iteration_lim_met)&(!time_lim_met))
    {
    // for (int iter = 0; iter < ITERATIONLIM; iter++)
    // {
        start = clock();
        printf("Iter: [%d] ========================================\n",iter_count); fflush(stdout);
        
        printf("Vector x_p_h: \n"); fflush(stdout);
        print_1d_array(row_num,x_p_h);

        Ex = mat_mul_vec(col_num,row_num, E, x_p_h);
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
        cuda_ret = cudaMalloc((void**) &s_d, sizeof(float)*row_num);
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate s_d device memory");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // Copy host variables to device ------------------------------------------
        printf("Copying data from host to device..."); fflush(stdout);
        startTime(&timer);
        // x_current
        // cuda_ret = cudaMemcpy(x_d, x_h, sizeof(float)*row_num, cudaMemcpyHostToDevice);
        // if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory x_h to x_d device");
        // u_current
        cuda_ret = cudaMemcpy(u_d, u_c_h, sizeof(float)*col_num, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory u_c_h to u_d device");
        // ME_T_d
        cuda_ret = cudaMemcpy(ME_T_d, ME_T_h_1d, sizeof(float)*row_num*col_num, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory ME_T_h_1d to ME_T_d device");
        // s
        cuda_ret = cudaMemcpy(s_d, s, sizeof(float)*row_num, cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory s to s_d device");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

        // Launch kernel ----------------------------------------------------------
        printf("Launching kernel..."); fflush(stdout);
        startTime(&timer);

        const unsigned int THREADS_PER_BLOCK = 256;
        const unsigned int numBlocks = (row_num - 1)/THREADS_PER_BLOCK + 1;
        printf("Num blocks: %d, Num threads per block: %d \n",numBlocks,THREADS_PER_BLOCK);
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

        // cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
        // calculate x_h
        // x_h = vec_add_vec(row_num,mat_mul_vec(row_num, col_num, ME_T, u_p_h),s);
        printf("Vector x_h: \n"); fflush(stdout);
        print_1d_array(row_num,x_h);

        // Close kernel & free memory
        cudaFree(u_d);
        cudaFree(x_d);

        // Iteration check
        if (iter_count >= ITERATIONLIM)
        {
            iteration_lim_met = true;
            printf("Iter count limit reached! %d\n", iter_count); fflush(stdout);
        }

        // Time limit check
        end = clock();
        iter_time = ((float) (end - start)) / CLOCKS_PER_SEC;
        cpu_time_used+=iter_time;
        if (cpu_time_used >= TIMELIMINSEC)
        {
            time_lim_met = true;
            printf("Time limit reached! %f\n", cpu_time_used); fflush(stdout);
        }

        // Tolerance verification
        printf("Tolerance check..."); fflush(stdout);
        float* gradient = vec_add_vec(col_num, g_Ex_u, scale_vec(col_num,-1, Ex));
        tol = vec_l1_norm(col_num, gradient);
        if (tol < 0.000001)
        {
            tolerance_met = true;
            printf("Tolerance limit reached! %f\n", tol); fflush(stdout);
            for (int _dim = 0; _dim < row_num; _dim++){
                x_optimal[_dim] = x_h[_dim]; //copy the value!
                printf("x(%d) =  %f\n", _dim, x_h[_dim]);
            }
            printf("Optimal sol X after copy: \n");
            print_1d_array(row_num,x_optimal);
            
        }

        // Updating rule
        // u_p_h
        u_p_h = u_c_h;
        u_p_minus = scale_vec(col_num,-1,u_p_h);
        x_p_h = x_h;

        iter_count++;
    }
    // get final solution

    printf("\n========================================\n"); fflush(stdout);
    printf("-----Terminating condition------\n");
    printf("Iteration_lim_met = %s", iteration_lim_met ? "true \n" : "false \n");
    printf("Time_lim_met = %s", time_lim_met ? "true \n" : "false \n");
    printf("Tolerance_met = %s", tolerance_met ? "true \n" : "false \n");
    printf("-----Computational resource summary------\n");
    printf("#Iteration = %d \n", iter_count );
    printf("Time used = %f \n", cpu_time_used );
    printf("Tolerance = %f \n", tol );
    // printf("Optimal obj:%f s\n", );
    printf("-----Solution received------\n");
    printf("Optimal sol X: \n"); print_1d_array(row_num,x_h);
    printf("========================================\n"); fflush(stdout);

    // }

    return 0;

}

