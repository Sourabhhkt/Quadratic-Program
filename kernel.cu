__global__ void vecMultiplyKernel(float* A, float* B, float* C, int m, int n, int k) {

    // Calculate global block and thread indices based on the block and thread indi ----
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    

    if (i<m && j<k){
	//Each thread calculates one index of the block submatrix
	float sum = 0.0f;
	for (int l = 0; l < k; l++) {
	sum = sum + A[i*k + l] * B[l*k+j];
	}
	
	C[i*m + j] = sum;
	}
}


