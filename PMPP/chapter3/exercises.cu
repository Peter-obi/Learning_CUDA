__global__
//one thread per row of output matrix
void MatrixMulRow(float* M, float* N, float* P, int Width){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < Width){
        for(int col = 0; col < Width; ++col){
                float Pvalue = 0;
            for(int k = 0; k < Width; ++k){
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
        P[row * Width + col] = Pvalue;
        }
    }
}

//one thread per column of output matrix
void MatrixMulCol(float* M, float* N, float* P, int Width){
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(col < Width){
        for(int row = 0; row < Width; ++row){
                float Pvalue = 0;
            for(int k = 0; k < Width; ++k){
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
        P[row * Width + col] = Pvalue;
        }
    }
}

void MatrixMul(float* M_h, float* N_h, float* P_h, int width){

    int size = width * width* sizeof(float);
    float* M_d; float* N_d;float* P_d;

    dim3 dimGrid (ceil(width/256), 1, 1);
    dim3 dimBlock (256, 1, 1);

    cudaMalloc((void**)&M_d, size);
    cudaMalloc((void**)&N_d, size);
    cudaMalloc((void**)&P_d, size);

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    MatrixMulCol<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

}

//vector-matrix multiplication

void vecMatrixMulKernel(float* A, float* B, float* C, int Width){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < Width){
        float Avalue = 0;
        for(int k = 0; k < Width; ++k){
            Avalue += B[row*Width + k] * C[k];
        }
        A[row] = Avalue;
    }
}

dim3 dimGrid (ceil(width/256), 1, 1);
dim3 dimBlock (256, 1, 1);
