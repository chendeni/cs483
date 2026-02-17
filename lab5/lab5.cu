// MP5 Reduction
// Input: A num list of length n
// Output: Sum of the list = list[0] + list[1] + ... + list[n-1];

#include <wb.h>

#define BLOCK_SIZE 512 //@@ This value is not fixed and you can adjust it according to the situation

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the correct index
  __shared__ float data[BLOCK_SIZE];
  //int iterations = 4;
  int index = threadIdx.x + 2*blockIdx.x*blockDim.x;

  int stride = BLOCK_SIZE;//512

  if(index<len)
  {
    data[threadIdx.x] = input[index];
  }
  else
  {
    data[threadIdx.x] = 0;
  }
    

  if((index+stride) < len)
  {
    data[threadIdx.x] += input[index+stride];
  }
  
  stride = stride>>1;//256
  __syncthreads();


  while(stride>0)
  {
    if(threadIdx.x < stride)
    {
      data[threadIdx.x] += data[threadIdx.x + stride];
    }
    stride = stride>>1;
    __syncthreads();
  }

  //printf("result: %f\n", data[0]);
  if(threadIdx.x==0)
  {
    output[blockIdx.x] = data[0];
    //printf("Block:%d, Partial sum:%f\n", blockIdx.x, data[0]);
  }
  


}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  //@@ Initialize device input and output pointers
  float *deviceInput;
  float *deviceOutput;

  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  //Import data and create memory on host
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  // The number of input elements in the input is numInputElements
  // The number of output elements in the input is numOutputElements

  //@@ Allocate GPU memory
  cudaMalloc((void **) &deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void **) &deviceOutput, numOutputElements * sizeof(float));

  //printf("\n\n");
  //printf("Input Size: %d\n",numInputElements);
  //printf("Output Size: %d\n",numOutputElements);


  //@@ Copy input memory to the GPU
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGrid(numOutputElements, 1, 1);//Dimensions of grid in blocks

  //@@ Launch the GPU Kernel and perform CUDA computation
  /*
  for(int i=0; i<numInputElements; i++)
  {
    if(i%(BLOCK_SIZE<<1) == 0)
    {
      printf("\n\n");
    }
    printf("%f, ", hostInput[i]);
  }
  */

  total<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, numInputElements);
  cudaDeviceSynchronize();  

  //@@ Copy the GPU output memory back to the CPU
  //cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);


  
  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. 
   * For simplicity, we do not require that for this lab.
   ********************************************************************/
  //printf("Outputs: %f, ", hostOutput[0]);
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
    //printf("%f, ", hostOutput[ii]);
  }
  //printf("\nResult: %f\n", hostOutput[0]);

  //@@ Free the GPU memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}

