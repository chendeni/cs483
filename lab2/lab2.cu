#include <wb.h>

#define THREADS_PER_BLOCK 256

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, 
                              int numARows, int numAColumns, 
                              int numBRows, int numBColumns, 
                              int numCRows, int numCColumns)
{
  

  //@@ Implement matrix multiplication kernel here
  int indexC = blockIdx.x * blockDim.x + threadIdx.x; 

  //In the last block, don't run if index of matrix C is out of bounds
  if(indexC < numCRows*numCColumns)
  {


    int rowA = indexC / numCColumns; 
    int colB = indexC % numCColumns;

    float countC = 0;
    for(int i=0; i<numAColumns; i++)
    {
      int indexA = rowA*numAColumns+i;
      int indexB = i*numBColumns + colB;
      countC += A[indexA] * B[indexB];

    }

    C[indexC] = countC;

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
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);



  if(numAColumns != numBRows)
  {
    wbLog(TRACE, "Cannot multiply matrix: Invalid dimensions");
    return -1;
  }
  
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
  dim3 DimGrid(lengthC / THREADS_PER_BLOCK, 1, 1);//Dimensions of grid in blocks
  dim3 DimBlock(THREADS_PER_BLOCK, 1, 1);//Dimensions of block in threads
  if (0 != (lengthC % 256)) //round up if there is a partially filled block
  { DimGrid.x++; } 

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC, 
                                numARows,  numAColumns, 
                                numBRows,  numBColumns, 
                                numCRows,  numCColumns);

  cudaDeviceSynchronize();
  
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, lengthC * sizeof(float), cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  /*
  for(int i=0; i<numCRows; i++)
  {
    for(int j=0; j<numCColumns; j++)
    {
      wbLog(TRACE, hostC[numCColumns*i + j], ",");
    }
    wbLog(TRACE, "////////////////////////");
  }
  */
  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);

  return 0;
}

