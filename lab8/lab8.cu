#include <wb.h>

#define BLOCK_SIZE 512

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void spmvJDSKernel(float *out, int *matColStart, int *matCols,
                              int *matRowPerm, int *matRows,
                              float *matData, float *vec, int dim) 
{
  //@@ insert spmv kernel for jds format
  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int i=0;
  int num_iterations = 0; 
  int data_index;// = matColStart[index];
  float data_elem;
  float vector_elem;
  int output_index;
  float total = 0;

  if(index<dim)
  {
    num_iterations = matRows[index];
    output_index = matRowPerm[index];
  }
  //__shared__ float vecShared[dim];
  //__shared__ float matColStart[dim];


  while(i < num_iterations)
  {
    data_index = matColStart[i]+index;
    data_elem = matData[data_index];
    vector_elem = vec[matCols[data_index]];
    total += data_elem*vector_elem;

    i+=1;
    //num_threads = matRows[i];
  }

  //printf("test:");
  //printf("%f, ",total);
  if(index<dim)
  {
    out[output_index] = total;
  }
  


}

static void spmvJDS(float *out, int *matColStart, int *matCols,
                    int *matRowPerm, int *matRows, float *matData,
                    float *vec, int dim)
{
  
  int numBlocks = ((dim-1) / BLOCK_SIZE) + 1;


  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGrid(numBlocks, 1, 1);

  spmvJDSKernel<<<DimGrid, DimBlock>>>(out, matColStart, matCols, 
      matRowPerm, matRows, matData, vec, dim);



  //@@ invoke spmv kernel for jds format
}

int main(int argc, char **argv) {
  wbArg_t args;
  int *hostCSRCols;
  int *hostCSRRows;
  float *hostCSRData;
  int *hostJDSColStart;
  int *hostJDSCols;
  int *hostJDSRowPerm;
  int *hostJDSRows;
  float *hostJDSData;
  float *hostVector;
  float *hostOutput;
  int *deviceJDSColStart;
  int *deviceJDSCols;
  int *deviceJDSRowPerm;
  int *deviceJDSRows;
  float *deviceJDSData;
  float *deviceVector;
  float *deviceOutput;
  int dim, ncols, nrows, ndata;
  int maxRowNNZ;

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  hostCSRCols = (int *)wbImport(wbArg_getInputFile(args, 0), &ncols, "Integer");
  hostCSRRows = (int *)wbImport(wbArg_getInputFile(args, 1), &nrows, "Integer");
  hostCSRData = (float *)wbImport(wbArg_getInputFile(args, 2), &ndata, "Real");
  hostVector = (float *)wbImport(wbArg_getInputFile(args, 3), &dim, "Real");

  hostOutput = (float *)malloc(sizeof(float) * dim);



  CSRToJDS(dim, hostCSRRows, hostCSRCols, hostCSRData, &hostJDSRowPerm, &hostJDSRows,
           &hostJDSColStart, &hostJDSCols, &hostJDSData);
  maxRowNNZ = hostJDSRows[0];

  // Allocate GPU memory.
  cudaMalloc((void **)&deviceJDSColStart, sizeof(int) * maxRowNNZ);
  cudaMalloc((void **)&deviceJDSCols, sizeof(int) * ndata);
  cudaMalloc((void **)&deviceJDSRowPerm, sizeof(int) * dim);
  cudaMalloc((void **)&deviceJDSRows, sizeof(int) * dim);
  cudaMalloc((void **)&deviceJDSData, sizeof(float) * ndata);

  cudaMalloc((void **)&deviceVector, sizeof(float) * dim);
  cudaMalloc((void **)&deviceOutput, sizeof(float) * dim);


  // Copy input memory to the GPU.
  cudaMemcpy(deviceJDSColStart, hostJDSColStart, sizeof(int) * maxRowNNZ,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSCols, hostJDSCols, sizeof(int) * ndata, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSRowPerm, hostJDSRowPerm, sizeof(int) * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSRows, hostJDSRows, sizeof(int) * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSData, hostJDSData, sizeof(float) * ndata, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceVector, hostVector, sizeof(float) * dim, cudaMemcpyHostToDevice);

/*
  printf("\n:Rows");
  for(int i=0; i<dim; i++)
  {
    if(i%(BLOCK_SIZE) == 0)
    {
      printf("\n");
    }
    printf("%d, ",hostJDSRows[i]);
  }
  
  printf("\n");
*/

  // Perform CUDA computation
  spmvJDS(deviceOutput, deviceJDSColStart, deviceJDSCols, deviceJDSRowPerm, deviceJDSRows,
          deviceJDSData, deviceVector, dim);
  cudaDeviceSynchronize();

  // Copy output memory to the CPU
  cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * dim, cudaMemcpyDeviceToHost);

  // Free GPU Memory
  cudaFree(deviceVector);
  cudaFree(deviceOutput);
  cudaFree(deviceJDSColStart);
  cudaFree(deviceJDSCols);
  cudaFree(deviceJDSRowPerm);
  cudaFree(deviceJDSRows);
  cudaFree(deviceJDSData);

  /*
  printf("\nOutput:");
  for(int i=0; i<dim; i++)
  {
    if(i%(BLOCK_SIZE) == 0)
    {
      printf("\n");
    }
    printf("%f, ",hostOutput[i]);
  }
  
  printf("\n");
*/
  wbSolution(args, hostOutput, dim);

  free(hostCSRCols);
  free(hostCSRRows);
  free(hostCSRData);
  free(hostVector);
  free(hostOutput);
  free(hostJDSColStart);
  free(hostJDSCols);
  free(hostJDSRowPerm);
  free(hostJDSRows);
  free(hostJDSData);

  return 0;
}
