// LAB 1
#include <wb.h>
#define THREADS_PER_BLOCK 256

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here

  //blockIdx: index of block in grid
  //threadIdx: index of thread in block
  //blockDim: Number of threads per block
  //i = index of inputs and outputs
  int i = blockIdx.x * blockDim.x + threadIdx.x; 

  //if i>=len in partially filled block, don't do anything.
  if(i<len)
  {
    out[i] = in1[i] + in2[i];
  }
  
}


int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here

  //inputLength is the number of floating point values
  //hostInput1, hostInput2, and hostOutput have the same length
  //size is in bytes
  int size = inputLength * sizeof(float);

      //pointers to device inputs and output
  float *deviceInput1, *deviceInput2, *deviceOutput; 

      //allocate memory on device for inputs and output
  cudaMalloc((void **) &deviceInput1, size);
  cudaMalloc((void **) &deviceInput2, size);
  cudaMalloc((void **) &deviceOutput, size);

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
  
  //@@ Initialize the grid and block dimensions here

  dim3 DimGrid(inputLength / THREADS_PER_BLOCK, 1, 1);//Dimensions of grid in blocks
  dim3 DimBlock(THREADS_PER_BLOCK, 1, 1);//Dimensions of block in threads
  
  if (0 != (inputLength % 256)) //round up if there is a partially filled block
  { DimGrid.x++; } 

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid,DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
