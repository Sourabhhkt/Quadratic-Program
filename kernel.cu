__global__ void vecMultiplyKernel(float* A, float* B, float* C, int m, int n) {

    // Calculate global block and thread indices based on the block and thread indi ----
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;

    float tmpSum = 0.0f;

    if (i<m && j<m){
	//Each thread calculates one index of the block submatrix
	for (int k = 0; k < n; k++) {
	tmpSum = tmpSum + A[i*n + k] * B[k*m+j];
	}
	
	C[i*m + j] = tmpSum;

	}
}

void matrixMultiplication(float *A, float *B, float *C, int m, int n){
//declare the number of blocks per thread and number of threads per block
dim3 threadsPerBlock(m,m);
dim3 blocksPerGrid(1,1);
	if(m*n > 512){
	threadsPerBlock.x = 512;
	threadsPerBlock.y = 512;
	blocksPerGrid.x = ceil( double(m)/double(threadsPerBlock.x));
	blocksPerGrid.y = ceil(double(m)/double(threadsPerBlock.y));
}
matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, m, n);
}

