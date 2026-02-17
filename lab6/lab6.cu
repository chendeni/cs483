// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void addSums(float *input, float *partialSums, int len)
  {
    int index = threadIdx.x + ((blockIdx.x*blockDim.x)<<1);

    float partial_sum = 0;
    if(blockIdx.x > 0)
    {
      partial_sum = partialSums[blockIdx.x-1];
    }


    if(index<len)
    {
      input[index] += partial_sum;
    }
    if((index+blockDim.x)<len)
    {
      input[index+blockDim.x] += partial_sum;
    }
    

  }




__global__ void scan(float *input, float *output, float *partial_sums, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float data[BLOCK_SIZE<<1];

  //__shared__ float data_b[2*BLOCK_SIZE];
  int index = threadIdx.x + ((blockIdx.x*blockDim.x)<<1);

  int from_index;
  int to_index;

  if(index<len)
  {
    data[threadIdx.x] = input[index];
  }
  else
  {
    data[threadIdx.x] = 0;
  }

  if((index+blockDim.x)<len)
  {
    data[threadIdx.x+blockDim.x] = input[index+blockDim.x];
  }
  else
  {
    data[threadIdx.x+blockDim.x] = 0;
  }

  __syncthreads();

  from_index = threadIdx.x << 1;
  to_index = from_index + 1;

  int stride = 1;
  int max_index = len-((blockIdx.x*blockDim.x)<<1);

  if(max_index > blockDim.x<<1)
  {
    max_index = blockDim.x<<1;
  }


  while(stride < blockDim.x<<1)
  {
    from_index = to_index - stride;
    
    stride = stride << 1;
    if((to_index+1)%stride == 0)
    {
      /*
      if(from_index >= 1024 || from_index<0)
      {
        printf("from_index:%d", from_index);
        printf("to_index:%d", to_index);
      }
      */
      data[to_index] += data[from_index];
    }
    
    __syncthreads();
  }

  stride = stride >> 2;
  from_index = to_index;

  while(stride>0)
  {
    to_index = from_index + stride;

    if(((from_index+1)%(stride<<1) == 0) && (to_index < blockDim.x<<1))
    {
      data[to_index] += data[from_index];
    }
    stride = stride >> 1;
    __syncthreads();
  }

  if(index<len)
  {
    output[index] = data[threadIdx.x];
  }
  
  if((index+blockDim.x)<len)
  {
    output[index+blockDim.x] = data[threadIdx.x+blockDim.x];
  }

  if(threadIdx.x == blockDim.x - 1)
  {
    partial_sums[blockIdx.x] = data[threadIdx.x+blockDim.x];
    /*
    if(len<100)
    {
      printf("\nSum:%f", data[threadIdx.x+blockDim.x]);
    }
    */
  }

  


}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *devicePartialSums;
  float *deviceSum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));





  //@@ Initialize the grid and block dimensions here
  /*
  printf("Input%d",numElements);
  for(int i=0; i<numElements; i++)
  {
    if(i%(BLOCK_SIZE*2) == 0)
    {
      printf("\n");
    }
    printf("%f, ", hostInput[i]);
  }
  printf("\n");
*/



  int numBlocks = numElements/(BLOCK_SIZE<<1);
  if((numElements%(BLOCK_SIZE<<1)) != 0)
  {
    numBlocks += 1;
  }

  cudaMalloc((void **)&devicePartialSums, numBlocks * sizeof(float));
  cudaMalloc((void **)&deviceSum, sizeof(float));

  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGrid(numBlocks, 1, 1);

  scan<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, devicePartialSums, numElements);
  cudaDeviceSynchronize();

  

  dim3 DimBlock2(BLOCK_SIZE, 1, 1);
  dim3 DimGrid2(1, 1, 1);
  scan<<<DimGrid2,DimBlock2>>>(devicePartialSums, devicePartialSums, deviceSum, numBlocks);
  cudaDeviceSynchronize();

  
  /*
  numBlocks = numElements/(BLOCK_SIZE);
  if((numElements%(BLOCK_SIZE)) != 0)
  {
    numBlocks += 1;
  }
  
  dim3 DimBlock3(BLOCK_SIZE, 1, 1);
  dim3 DimGrid3(numBlocks, 1, 1);
  */
  addSums<<<DimGrid,DimBlock>>>(deviceOutput, devicePartialSums, numElements);
  


  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));

  /*
  printf("Output:");
  for(int i=0; i<numElements; i++)
  {
    if(i%(BLOCK_SIZE*2) == 0)
    {
      printf("\n");
    }
    printf("%f, ", hostOutput[i]);
  }
  printf("\n\n");
*/

  //@@  Free GPU Memory

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(devicePartialSums);
  cudaFree(deviceSum);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

