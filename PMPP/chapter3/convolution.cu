#define BLUR_SIZE 1 //3 x 3 patch
__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < w && row < h){
        int pixVal = 0;
        int pixels = 0;
        //Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for(int blurRow=-BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow){
            for(int blurCol=-BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                //Verify we have a valid image pixel
                if (curRow >= 0 && curRow < h && curCol >=0 && curCol < w){
                    pixVal += in[curRow * w + curCol];
                    ++pixels; // Keep track of number of pixels in the avg
                }
            }
        }
        //Write our new pixel value out
        out[row*w + col] = (unsigned char) (pixVal/pixels);
    }
}

void blur(unsigned char* in_h, unsigned char * out_h, int m, int n){
    int size = m * n * sizeof(unsigned char);
    unsigned char* in_d; unsigned char* out_d;

    dim3 dimGrid (ceil(m/16.0), ceil(n/16.0), 1);
    dim3 dimBlock (16, 16, 1);

    cudaMalloc((void**)&in_d, size);
    cudaMalloc((void**)&out_d, size);

    cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);

    blurKernel<<<dimGrid, dimBlock>>>(in_d, out_d, m, n);

    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);

}
