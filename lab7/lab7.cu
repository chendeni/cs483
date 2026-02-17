// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

#define GRID_SIZE 64

//@@ insert code here

__global__ void castToUChar(float *input, unsigned char *output, int len)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x; 
  while(index < len)
  {
    output[index] = (unsigned char) (255 * input[index]);
    index += stride;
  }



}

__global__ void castToFloat(unsigned char *input, float *output, int len)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x; 

  while(index < len)
  {
    output[index] = (float) (input[index]/255.0);
    index += stride;
  }

}

__global__ void convertGrayscale(unsigned char *input, unsigned char *output, int len)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int index2;
  int stride = blockDim.x*gridDim.x; 

  while(index < len)
  {
    index2 = index*3;
    output[index] = (unsigned char) (0.21*input[index2] + 0.71*input[index2+1] + 0.07*input[index2+2]);
    index += stride;
  }
}

__global__ void histogram(unsigned char *input, unsigned int *histogram, int len)
{
  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x; 

  __shared__ unsigned int privateHistogram[HISTOGRAM_LENGTH];

  if(threadIdx.x < HISTOGRAM_LENGTH)
  {
    privateHistogram[threadIdx.x] = 0;
  }
  __syncthreads();
  while(index < len)
  {
    atomicAdd(&(privateHistogram[input[index]]), 1);
    index += stride;
  }
  __syncthreads();
  if(threadIdx.x < HISTOGRAM_LENGTH)
  {
    atomicAdd(&(histogram[threadIdx.x]), privateHistogram[threadIdx.x]);
  }


  /*
  __syncthreads();
  if(threadIdx.x==0 && blockIdx.x == 0)
  {
    printf("Histogram: ");
    int sum=0;
    for(int i=0; i<256; i++)
    {
      printf("%u, ",histogram[i]);
      sum+=histogram[i];
    }
    printf("Sum:%d", sum);
  }

  */
}

__global__ void cdf(unsigned int *input, float *output, int len, int size) 
{
  __shared__ float data[HISTOGRAM_LENGTH];

  
  int index = threadIdx.x + ((blockIdx.x*blockDim.x)<<1);

  int from_index;
  int to_index;

  if(index<len)
  {
    data[threadIdx.x] = (float)input[index];
  }
  else
  {
    data[threadIdx.x] = 0;
  }

  if((index+blockDim.x)<len)
  {
    data[threadIdx.x+blockDim.x] = (float)input[index+blockDim.x];
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
    output[index] = data[threadIdx.x]/size;
  }
  
  if((index+blockDim.x)<len)
  {
    output[index+blockDim.x] = data[threadIdx.x+blockDim.x]/size;
  }

/*
  __syncthreads();
  if(threadIdx.x==0 && blockIdx.x == 0)
  {
    printf("CDF: ");
    float sum=0;
    for(int i=0; i<256; i++)
    {
      printf("%f, ",output[i]);
      sum+=output[i];
    }
  }
*/


}

__global__ void correctColor(unsigned char *input, unsigned char *output, float* cdf, int len) 
{

  int index = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x; 
  float cdf_min = cdf[0];

  while(index < len)
  {
    float correct_val = 255*(cdf[input[index]] - cdf_min)/(1.0-cdf_min);

    if(correct_val > 255)
    {
      correct_val = 255;
    }
    if(correct_val < 0)
    {
      correct_val = 0;
    }

    output[index] = (unsigned char)correct_val;
    index += stride;
  }


}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  int imageSize;
  int imageSizeGrayscale;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;


  float *deviceInputImageData;
  unsigned char *deviceImage;
  unsigned char *deviceImageGrayscale;
  unsigned char *deviceImageCorrected;
  float *deviceOutputImageData;

  unsigned int *deviceHistogram;
  float *deviceCdf;



  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */
  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  
  imageSizeGrayscale = imageWidth*imageHeight;
  imageSize = imageSizeGrayscale*imageChannels;

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = (float *)malloc(imageSize * sizeof(float));

  //@@ insert code here

  cudaMalloc((void **)&deviceInputImageData, imageSize * sizeof(float));
  cudaMalloc((void **)&deviceImage, imageSize * sizeof(unsigned char));
  cudaMalloc((void **)&deviceImageGrayscale, imageSizeGrayscale * sizeof(unsigned char));
  cudaMalloc((void **)&deviceImageCorrected, imageSize * sizeof(unsigned char));
  cudaMalloc((void **)&deviceOutputImageData, imageSize * sizeof(float));

  cudaMalloc((void **)&deviceHistogram,  HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceCdf,  HISTOGRAM_LENGTH * sizeof(float));




/*
  printf("\nWidth:%d, Height:%d, Size:%d", imageWidth, imageHeight, imageSize);
  for(int i=0; i<imageSize; i++)
  {
    
    if(i%(imageWidth*imageChannels) == 0)
    {
      printf("\n");
    }
    printf("%f, ",hostInputImageData[i]);
  }
  */

  cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize * sizeof(float), cudaMemcpyHostToDevice);

  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGrid(GRID_SIZE, 1, 1);

  dim3 DimBlock2(HISTOGRAM_LENGTH>>1, 1, 1);
  dim3 DimGrid2(1, 1, 1);

  castToUChar<<<DimGrid,DimBlock>>>(deviceInputImageData, deviceImage, imageSize);
  cudaDeviceSynchronize();
  assert(imageChannels == 3);
  convertGrayscale<<<DimGrid,DimBlock>>>(deviceImage, deviceImageGrayscale, imageSizeGrayscale);
  cudaDeviceSynchronize();
  histogram<<<DimGrid,DimBlock>>>(deviceImageGrayscale, deviceHistogram, imageSizeGrayscale);
  cudaDeviceSynchronize();


  cdf<<<DimGrid2,DimBlock2>>>(deviceHistogram, deviceCdf, HISTOGRAM_LENGTH, imageSizeGrayscale);
  cudaDeviceSynchronize();

  correctColor<<<DimGrid,DimBlock>>>(deviceImage, deviceImageCorrected, deviceCdf, imageSize);
  cudaDeviceSynchronize();

  castToFloat<<<DimGrid,DimBlock>>>(deviceImageCorrected, deviceOutputImageData, imageSize);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

  /*
  printf("\nOutput:");
  for(int i=0; i<imageSize; i++)
  {
    
    if(i%(imageWidth*imageChannels) == 0)
    {
      printf("\n");
    }
    printf("%f, ",hostOutputImageData[i]);
  }
  */
  //printf("Width:%d",inputImage.width);
  //printf("Height:%d",inputImage.height);
  wbImage_setData(outputImage, hostOutputImageData);
  


  wbSolution(args, outputImage);

  //@@ insert code here

  cudaFree(deviceInputImageData);
  cudaFree(deviceImage);
  cudaFree(deviceImageGrayscale);
  cudaFree(deviceImageCorrected);
  cudaFree(deviceOutputImageData);

  cudaFree(deviceHistogram);
  cudaFree(deviceCdf);
  



  return 0;
}

