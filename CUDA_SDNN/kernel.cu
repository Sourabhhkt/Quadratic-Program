__global__ void vecAddKernel(float* A, float* B, float* C, int n) {

    // Calculate global thread index based on the block and thread indices ----

    //INSERT KERNEL CODE HERE
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    // Use global index to determine which elements to read, add, and write ---

    //INSERT KERNEL CODE HERE
    if (i < n)
    {
        C[i] = A[i]+B[i];
    }
}


__global__ void sdnnIterationKernel(float* x_d, float* u_d, float*  ME_T_d, float* s_d, int row_num, int col_num) {

    // Calculate global thread index based on the block and thread indices ----
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < row_num)
    {
        x_d[i] = s_d[i];
        for (int j = 0; j < col_num; j++)
        {
            x_d[i] += (ME_T_d[i*row_num + j]*u_d[j]);
        }
    }
    __syncthreads();
}

