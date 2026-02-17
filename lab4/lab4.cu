#include <wb.h>
#include <math.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define BLOCK_X_DIM 8 //block x dimension equals y and z dimension
#define MASK_WIDTH 3
#define MASK_SIZE MASK_WIDTH*MASK_WIDTH*MASK_WIDTH
#define MASK_RADIUS MASK_WIDTH/2

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_SIZE];



__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float Tile[BLOCK_X_DIM + 2*MASK_RADIUS][BLOCK_X_DIM + 2*MASK_RADIUS][BLOCK_X_DIM + 2*MASK_RADIUS];

  int x = blockIdx.x * BLOCK_X_DIM + threadIdx.x - MASK_RADIUS; 
  int y = blockIdx.y * BLOCK_X_DIM + threadIdx.y - MASK_RADIUS; 
  int z = blockIdx.z * BLOCK_X_DIM + threadIdx.z - MASK_RADIUS;
  int index = z*x_size*y_size + y*x_size + x;

  
  bool is_ghost = (x >= x_size || x < 0 || y >= y_size || y < 0 || z >= z_size || z < 0);
  bool is_halo = (threadIdx.x < MASK_RADIUS || threadIdx.x >= (MASK_RADIUS+BLOCK_X_DIM) || 
                  threadIdx.y < MASK_RADIUS || threadIdx.y >= (MASK_RADIUS+BLOCK_X_DIM) || 
                  threadIdx.z < MASK_RADIUS || threadIdx.z >= (MASK_RADIUS+BLOCK_X_DIM));
  /*
  if(index>=0 && index<9)
  {
    printf("%d,%d,%d: %d,%d,%d: %d,%d\n",x,y,z,threadIdx.x,threadIdx.y,threadIdx.z,is_ghost,is_halo);
  }
  */
  /*
  if(threadIdx.x == 1 && threadIdx.y == 1 && threadIdx.z == 1)
  {
    printf("%d,%d,%d: %d,%d,%d: %d,%d\n",x,y,z,threadIdx.x,threadIdx.y,threadIdx.z,is_ghost,is_halo);
  }
  */
  if(is_ghost)
  {
    Tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
  }
  else
  {
    Tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[index];
  }
  __syncthreads();

  if(!is_halo && !is_ghost)
  {
    float total = 0;
    for(int i=0; i<MASK_WIDTH; i++)
    {
      for(int j=0; j<MASK_WIDTH; j++)
      {
        for(int k=0; k<MASK_WIDTH; k++)
        {
          int offset_x = threadIdx.x - MASK_RADIUS + k;
          int offset_y = threadIdx.y - MASK_RADIUS + j;
          int offset_z = threadIdx.z - MASK_RADIUS + i;
          int mask_index = k + j*MASK_WIDTH + i*MASK_WIDTH*MASK_WIDTH;

          total += deviceKernel[mask_index] * Tile[offset_z][offset_y][offset_x];

        }
      }
    }


    output[index] = total;
    
  }
  __syncthreads();


}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  int blockWidth;
  
  //@@ Initial deviceInput and deviceOutput here.
  
  args = wbArg_read(argc, argv);
  
  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == MASK_SIZE);// = 27


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  float *deviceInput, *deviceOutput;
  cudaMalloc((void **) &deviceInput, (inputLength-3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength-3) * sizeof(float));


  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength-3) * sizeof(float), cudaMemcpyHostToDevice);
  

  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength*sizeof(float));


  //@@ Initialize grid and block dimensions here
  blockWidth = BLOCK_X_DIM + MASK_RADIUS*2;// BLOCK_X_DIM + ((MASK_WIDTH >> 1) << 1);
  //wbLog(TRACE, "blockWidth", blockWidth);
  


  dim3 DimBlock(blockWidth, blockWidth, blockWidth);
  dim3 DimGrid(x_size/BLOCK_X_DIM, y_size/BLOCK_X_DIM, z_size/BLOCK_X_DIM);//Dimensions of grid in blocks

  if (0 != (x_size % BLOCK_X_DIM)) //round up if there is a partially filled block
  { DimGrid.x++; } 
  if (0 != (y_size % BLOCK_X_DIM)) //round up if there is a partially filled block
  { DimGrid.y++; } 
  if (0 != (z_size % BLOCK_X_DIM)) //round up if there is a partially filled block
  { DimGrid.z++; } 

  //assert(blockWidth == 10);
  
  //printf("DimGrid: %d,%d,%d\n",DimGrid.x,DimGrid.y,DimGrid.z);
  //printf("DimBlock: %d,%d,%d\n",DimBlock.x,DimBlock.y,DimBlock.z);
  
  /*
  for(int i=0; i<z_size; i++ )
  {
    for(int j=0; j<y_size; j++ )
    {
      for(int k=0; k<x_size; k++ )
      {

        printf("%f,", hostInput[i*y_size*x_size + j*x_size + k + 3]);
      }
      //printf("\n");
      wbLog(TRACE, "");
    }
    //printf("///////////////////////////////////\n");
    wbLog(TRACE, "-----------------------------------------\n");
  }
  */
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();






  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  //cudaMemcpy(hostOutput + (3 * sizeof(float)), deviceOutput, (inputLength-3) * sizeof(float), cudaMemcpyDeviceToHost);
  //printf("size: %d\n", (3 * sizeof(float)));
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength-3) * sizeof(float), cudaMemcpyDeviceToHost);

  //cudaMemcpy(hostOutput, deviceOutput, sizeof(float), cudaMemcpyDeviceToHost);



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;

  /*
  printf("Output:\n");
  for(int i=0; i<z_size; i++ )
  {
    for(int j=0; j<y_size; j++ )
    {
      for(int k=0; k<x_size; k++ )
      {

        printf("%f,", hostOutput[i*y_size*x_size + j*x_size + k + 3]);
        
        if(hostOutput[i*y_size*x_size + j*x_size + k + 3] != hostInput[i*y_size*x_size + j*x_size + k + 3])
        {
          printf("not equal:%f\n", hostInput[i*y_size*x_size + j*x_size + k + 3]);
        }
        

      }
      //printf("\n");
      wbLog(TRACE, "");
    }
    //printf("///////////////////////////////////\n");
    wbLog(TRACE, "-----------------------------------------\n");
  }
  */
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

