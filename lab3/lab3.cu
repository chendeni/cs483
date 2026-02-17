#include <wb.h>

//#define THREADS_PER_BLOCK 256
#define BLOCK_X_DIM 16 //block x dimension equals y dimension

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) 
{
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileA[BLOCK_X_DIM][BLOCK_X_DIM];
  __shared__ float subTileB[BLOCK_X_DIM][BLOCK_X_DIM];

  int rowC = blockIdx.y * blockDim.y + threadIdx.y; 
  int colC = blockIdx.x * blockDim.x + threadIdx.x; 

  int num_steps = (numAColumns / blockDim.x);
  if(num_steps % blockDim.x != 0)
  {
    num_steps += 1;
  }

  float count = 0;

  for(int i=0; i<num_steps; i++)
  {
    
    if(rowC >= numARows || ( blockDim.x*i + threadIdx.x ) >= numAColumns)
    {
      subTileA[threadIdx.y][threadIdx.x] = 0;
    }
    else
    {
      subTileA[threadIdx.y][threadIdx.x] = A[numAColumns * rowC + blockDim.x*i + threadIdx.x];
    }
    if(colC >= numBColumns || ( blockDim.y*i + threadIdx.y) >= numBRows)
    {
      subTileB[threadIdx.y][threadIdx.x] = 0;
    }
    else
    {
      subTileB[threadIdx.y][threadIdx.x] = B[numBColumns * ( blockDim.y*i + threadIdx.y) + colC];
    }
    __syncthreads();


    for(int j=0; j < blockDim.x; j++)
    {
      count += subTileA[threadIdx.y][j] * subTileB[j][threadIdx.x];
    }

    __syncthreads();

    
  }

  if(rowC < numCRows && colC < numCColumns)
  {
    C[rowC*numCColumns + colC] = count;
  }
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  int lengthA = numARows * numAColumns;
  int lengthB = numBRows * numBColumns;
  int lengthC = numCRows * numCColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(lengthC * sizeof(float));

  //@@ Allocate GPU memory here
  float *deviceA, *deviceB, *deviceC;
  cudaMalloc((void **) &deviceA, lengthA * sizeof(float));
  cudaMalloc((void **) &deviceB, lengthB * sizeof(float));
  cudaMalloc((void **) &deviceC, lengthC * sizeof(float));


  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, lengthA * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, lengthB * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here


  dim3 DimGrid(numCColumns/BLOCK_X_DIM, numCRows/BLOCK_X_DIM, 1);//Dimensions of grid in blocks
  dim3 DimBlock(BLOCK_X_DIM, BLOCK_X_DIM, 1);//Dimensions of block in threads

  if (0 != (numCRows % BLOCK_X_DIM)) //round up if there is a partially filled block
  { DimGrid.y++; } 
  if (0 != (numCColumns % BLOCK_X_DIM)) //round up if there is a partially filled block
  { DimGrid.x++; } 

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC, 
                                numARows,  numAColumns, 
                                numBRows,  numBColumns, 
                                numCRows,  numCColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, lengthC * sizeof(float), cudaMemcpyDeviceToHost);
/*
  if(numCColumns == 257)
  {
    for(int i=0; i<numCRows; i++)
    {
      for(int j=0; j<numCColumns; j++)
      {
        wbLog(TRACE, hostC[numCColumns*i + j], ",");
      }
      wbLog(TRACE, "////////////////////////");
    }
  }
*/
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);


  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);

  return 0;
}
