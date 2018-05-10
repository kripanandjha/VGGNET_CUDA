/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//		Author:    Harshitha Bura, Kripanand Jha
//		File:      CUDA implementation of Vggnet16
//		Objective: Implementing vggnet layer with CUDA support
/////////////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <assert.h>
#include <cuda_profiler_api.h>
using namespace std;

#define IMAGE_WIDTH 224
#define CHANNELS 3
#define LAYER1_PARAMS 1728 //params: (3*3*3)*64
#define LAYER1_BIAS_PARAMS 64 //bias: 64
#define LAYER2_PARAMS 36864 //params: (3*3*64)*64
#define LAYER2_BIAS_PARAMS 64 //bias: 64
#define LAYER3_PARAMS 73728 //params: (3*3*64)*128
#define LAYER3_BIAS_PARAMS 128 //bias: 128
#define LAYER4_PARAMS 147456 //params: (3*3*128)*128
#define LAYER4_BIAS_PARAMS 128 //bias: 128
#define LAYER5_PARAMS 294912 //params: (3*3*128)*256
#define LAYER5_BIAS_PARAMS 256 //bias: 256
#define LAYER6_PARAMS 589824 //params: (3*3*256)*256
#define LAYER6_BIAS_PARAMS 256 //bias: 256
#define LAYER7_PARAMS 589824 //params: (3*3*256)*256
#define LAYER7_BIAS_PARAMS 256 //bias: 256
#define LAYER8_PARAMS 1179648 //params: (3*3*256)*512
#define LAYER8_BIAS_PARAMS 512 //bias: 512
#define LAYER9_PARAMS 2359296 //params: (3*3*512)*512
#define LAYER9_BIAS_PARAMS 512 //bias: 512
#define LAYER10_PARAMS 2359296 //params: (3*3*512)*512
#define LAYER10_BIAS_PARAMS 512 //bias: 512
#define LAYER11_PARAMS 2359296 //params: (3*3*512)*512
#define LAYER11_BIAS_PARAMS 512 //bias: 512
#define LAYER12_PARAMS 2359296 //params: (3*3*512)*512
#define LAYER12_BIAS_PARAMS 512 //bias: 512
#define LAYER13_PARAMS 2359296 //params: (3*3*512)*512
#define LAYER13_BIAS_PARAMS 512 //bias: 512
#define LAYER14_PARAMS 102760448 //params: (7*7*512)*4096
#define LAYER14_BIAS_PARAMS 4096
#define LAYER15_PARAMS 16777216 //params: 4096*4096
#define LAYER15_BIAS_PARAMS 4096
#define LAYER16_PARAMS 4096000 //params: 4096*4096
#define LAYER16_BIAS_PARAMS 1000
#define MASK_WIDTH 3


extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
void NeuralNetwork(char* file_path);
unsigned NUM;

__device__ double atomicAdd1(double* address, double val) {
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val +
					__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}

/////////////////////////////////////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	int commandline_error;
	commandline_error = 0;
	if (argc >= 2) {
		NUM = atoi(argv[1]);
	} else commandline_error=1;

	if (commandline_error || !NUM) {
		printf("Usage: ./NN <NUM> image_path \n");
		printf("where NUM is the number of images to process in parallel. \n");
		return 1;
	}

	NeuralNetwork(argv[2]);
}

/////////////////////////////////////////////////////////////////////////////////////////
// Read all the weights from the weight files for all layers to the intialised host memory
/////////////////////////////////////////////////////////////////////////////////////////
void InitWeights_Biases(double *Weights_CPU, int size, char* file_path) {
	//Layer Weights
	FILE * pFile = fopen (file_path,"r");
	if (!pFile) {
		printf("FAIL! INPUT WEIGHTS NOT FOUND! %s\n",file_path);
		exit(1);
	}

	long int i = 0;
	if (pFile != NULL){
		size_t len = 99;
		char *line = NULL;
		while ((getline(&line, &len, pFile)) != -1) {
			double temp_num = atof(line);
			Weights_CPU[i] = temp_num;
			i++;
			if(i==size) {
				break;
			}
		}
		fclose(pFile);
	}
	
}

/////////////////////////////////////////////////////////////////////////////////////////
// Read the input image file, which is a txt file with R, G and B values
/////////////////////////////////////////////////////////////////////////////////////////
void LoadInput(double *Data_Layer_CPU,char* file_path) {
#if 0
// tested with below images
"data/vgg_rgb.txt"
"data/cat.1135.jpg.txt"
"data/dog.4.jpg.txt"
"data/dog.20.jpg.txt"
"data/zebra.png.txt"
"data/zebr1.png.txt"
"data/new_cat.0.txt"
#endif
	FILE * pFile = fopen (file_path,"rb");
	if (!pFile) {
		printf("FAIL! INPUT FILE NOT FOUND!\n");
		exit(1);
	}
	if (pFile != NULL){
		long int i = 0;
		size_t len = 99;
		char *line = NULL;
		while ((getline(&line, &len, pFile)) != -1) {
			double temp_num = atof(line);
			Data_Layer_CPU[i] = temp_num;
			i++;
			if(i==IMAGE_WIDTH*IMAGE_WIDTH*3) {
				printf("compeleted reading img file\n");
				break;
			}
		}
		fclose(pFile);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////
// Split the RGB array to separate R, G amd B channel arrays
/////////////////////////////////////////////////////////////////////////////////////////
void ConvertInput(double *Data_Layer_CPU_R, double *Data_Layer_CPU_G, double *Data_Layer_CPU_B, double *Data_Layer_CPU)
{
	for(int i=0; i<IMAGE_WIDTH*IMAGE_WIDTH*CHANNELS; i+=3)
	{
		Data_Layer_CPU_R[i/3] = Data_Layer_CPU[i];
		Data_Layer_CPU_G[i/3] = Data_Layer_CPU[i+1];
		Data_Layer_CPU_B[i/3] = Data_Layer_CPU[i+2];
	}
}

/////////////////////////////////////////////////////////////////////////////////////////
// Device function to execute first convolutional layer
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void ExecuteFirstLayer(double *Layer1_Weights_CPU, double *Data_Layer_CPU_R, double *Data_Layer_CPU_G, double *Data_Layer_CPU_B, double *Layer1_Features,double* bias) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	for(int f=0; f<64; f++) { //Number of filters = 64
		double result = 0;
		for(int i = x-1; i<=x+1; i++) { //padding = 1
			for(int j=y-1; j<=y+1; j++){
				int x_index = i-x+1;
				int y_index = j-y+1;
				int m = (y_index)+(x_index)*MASK_WIDTH;
				if(i<0 || j<0) {
					result+= 0;
				}
				else if(j>IMAGE_WIDTH-1 || i>IMAGE_WIDTH-1) {
					result+= 0;
				}
				else {
					result += Data_Layer_CPU_R[(y_index-1) + (x*IMAGE_WIDTH + y) + (x_index-1)*IMAGE_WIDTH]*Layer1_Weights_CPU[m+f*MASK_WIDTH*MASK_WIDTH*CHANNELS] + Data_Layer_CPU_G[(y_index-1) + x*IMAGE_WIDTH + y + (x_index-1)*IMAGE_WIDTH]*Layer1_Weights_CPU[m+MASK_WIDTH*MASK_WIDTH+f*MASK_WIDTH*MASK_WIDTH*CHANNELS] + Data_Layer_CPU_B[(y_index-1) + x*IMAGE_WIDTH + y + (x_index-1)*IMAGE_WIDTH]*Layer1_Weights_CPU[m+MASK_WIDTH*MASK_WIDTH* 2+ f*MASK_WIDTH*MASK_WIDTH*CHANNELS];
				}
			}
		}
		result += bias[f];
		//Relu Activation Function
		if(result < 0)
			result = 0;
		Layer1_Features[f*IMAGE_WIDTH*IMAGE_WIDTH+x*IMAGE_WIDTH+y] = result;
	}
}

/////////////////////////////////////////////////////////////////////////////////////////
// Device function to execute second convolutional layer
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void ExecuteConvLayer(double *Layer_Weights_CPU, double *Layer2_Features, double *Layer1_Features, int output_image_depth, int input_image_depth, int image_size,double* bias) {
	int kernel_depth = input_image_depth;
	int padding = 1;
	int mask_width = 3; //kernel width is 3 for all convolution layers in VGGNET
	double Features = 0;
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	for(int f=0; f<output_image_depth; f++) {
		Features = 0;
		for(int n=0; n<input_image_depth; n++) {// Input Image Depth = 64
			if(x<image_size){
				if(y<image_size) {
					double result = 0;
					for(int i = x-padding; i<=x+padding; i++) {// Padding = 1
						for(int j=y-padding; j<=y+padding; j++) {
							int x_index = i-x+padding;
							int y_index = j-y+padding;
							int m = (y_index)+(x_index)*mask_width;
							if(i<0 || j<0) {
								result+=0;
							}
							else if(j>image_size-padding || i>image_size-padding) {
								result+=0;
							}
							else {
								result+= Layer1_Features[n*image_size*image_size + (x_index+x-padding)*image_size + (y_index+y-padding)]*Layer_Weights_CPU[m+f*mask_width*mask_width*kernel_depth+n*mask_width*mask_width]; //Number of kernels =64
							}
						}
					}
					Features += result;
				}
			}
		}
		Features += bias[f];
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		Layer2_Features[f*image_size*image_size + x*image_size + y] = Features;
	}
}


/////////////////////////////////////////////////////////////////////////////////////////
// Device function to execute fourth layer, which is a fully connected layer
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void ExecuteFcLayer(double *Layer_Weights_CPU, double *Layer_Features, double *Pool_Layer_Features,int width, int image_depth, double* Layer_Bias) {
	int n = blockDim.x*blockIdx.x + threadIdx.x; // n is number of kernel
	double result = 0;
	for(int f=0; f<image_depth; f++) {// depth of image and kernel
		for(int x=0; x<width; x++){
			for(int y=0; y<width; y++){
				result+= Pool_Layer_Features[f*width*width +x*width + y] * Layer_Weights_CPU[y+(x*width)+(f*width*width)+(n*width*width*image_depth)];
			}
		}
	}
		result += Layer_Bias[n];
	if(result < 0 ) { // reLu
		result  = 0 ;
	}
	Layer_Features[n] = result ;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Device function to execute fifth layer, which is a fully connected layer
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void ExecuteFc1Layer(double *Layer5_Weights_CPU, double *Layer5_Features, double *Layer4_Features, int image_depth, double* Layer_Bias,bool use_relu){
	int n = blockDim.x*blockIdx.x + threadIdx.x;
	double result = 0;
	for(int f=0; f<image_depth; f++){
		result+= Layer4_Features[f] * Layer5_Weights_CPU[f+n*image_depth];
	}

	if(use_relu ) {
		if(result < 0 ) { // reLu
			result  = 0 ;
		}
	}

	Layer5_Features[n] = result + Layer_Bias[n];
	result = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Device function to execute max pooling compuation for the first layer
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void pooling(double *Layer_Neurons_GPU,double *Layer_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc){
	int row = (blockIdx.x * blockDim.x) + threadIdx.x;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y;
	int padding = 1;
	double max = 0.0;
	for(int output =0;output < out ;output++){

		if(row%stride_width != 0) {//stride = 2

			if(col%stride_width != 0){ //stride = 2

				for(int i = row-padding; i <= row; i++){

					if(i>(in_fr-1)) break;
					for(int j = col-padding; j <= col; j++){

						if(j>(in_fr-1)) break;
						if(max < ((Layer_Neurons_GPU[output*in_fr*in_fc+i*in_fr+j])))
							max =   ((Layer_Neurons_GPU[output*in_fr*in_fc+i*in_fr+j])) ;

					}
				}
				//ReLU activation function compuation
				if(max<0)
					max = 0;
				Layer_pool_GPU[(output*out_fr*out_fc) + (row-1)*(out_fr/2) + (col-1)/2] = max;
				max = 0.0;
			}
		}
	}
}

__device__ void findmax(double* result, double* in){
	int TotalThreads = blockDim.x;	// Total number of active threads
	__shared__ double max[1000];
	max[threadIdx.x] = in[threadIdx.x];
	__syncthreads();

	while(TotalThreads > 1)
	{
		int halfPoint = (TotalThreads >> 1);	// divide by two
		// only the first half of the threads will be active.

		if (threadIdx.x < halfPoint)
		{
			double temp = max[threadIdx.x + halfPoint];
			if (temp > max[threadIdx.x]) max[threadIdx.x] = temp;

		}
		__syncthreads();

		TotalThreads = (TotalThreads >> 1);	// divide by two.

	}

	if (threadIdx.x == 0)
	{
		result[0] = max[0];
	}

}
//__global__ void softmax(double* Layer17_Features, double* Layer16_Features,double m){
__global__ void softmax(double* Layer17_Features, double* Layer16_Features){

	int tid = threadIdx.x;
	__shared__ double feat_exp[1000];
	__shared__ double sum_exp ;
	__shared__ double m;

	findmax(&m, Layer16_Features);
	__syncthreads();
	feat_exp[tid]  = exp(Layer16_Features[tid] - m);

	atomicAdd1(&sum_exp, feat_exp[tid]);
	__syncthreads();
	double offset = m + log(sum_exp);

	Layer17_Features[tid] = exp(Layer16_Features[tid] - offset);


}

void LoadImageNetClass(char **image_class, char *file_path){
	FILE * pFile = fopen (file_path,"r");
	if (!pFile) {
		printf("FAIL! INPUT WEIGHTS NOT FOUND! %s\n",file_path);
		exit(1);
	}

	long int i = 0;
	if (pFile != NULL){
		size_t len = 99;
		char *line = NULL;
		while ((getline(&line, &len, pFile)) != -1) {
			strcpy(image_class[i],line);
			i++;
			if(i==1000) {
				break;
			}
		}
		fclose(pFile);
	}
}

void NeuralNetwork(char *file_path) { 
	cudaError_t err;
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "There is no device.\n");
		exit(EXIT_FAILURE);
	}
	int dev;
	for (dev = 0; dev < deviceCount; ++dev) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (deviceProp.major >= 1)
			break;
	}
	if (dev == deviceCount) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
	else
		cudaSetDevice(dev);
	//Allocation of host memory for weights
	double *Layer1_Weights_CPU = (double*) malloc (LAYER1_PARAMS * NUM * sizeof(double)); //no. of features in nth layer
	double *Layer1_Weights_Bias_CPU = (double*) malloc (LAYER1_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer2_Weights_CPU = (double*) malloc (LAYER2_PARAMS * NUM * sizeof(double)); // no. of features in nth layer
	double *Layer2_Weights_Bias_CPU = (double*) malloc (LAYER2_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer3_Weights_CPU = (double*) malloc (LAYER3_PARAMS * NUM * sizeof(double));
	double *Layer3_Weights_Bias_CPU = (double*) malloc (LAYER3_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer4_Weights_CPU = (double*) malloc (LAYER4_PARAMS * NUM * sizeof(double));
	double *Layer4_Weights_Bias_CPU = (double*) malloc (LAYER4_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer5_Weights_CPU = (double*) malloc (LAYER5_PARAMS * NUM * sizeof(double));
	double *Layer5_Weights_Bias_CPU = (double*) malloc (LAYER5_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer6_Weights_CPU = (double*) malloc (LAYER6_PARAMS * NUM * sizeof(double));
	double *Layer6_Weights_Bias_CPU = (double*) malloc (LAYER6_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer7_Weights_CPU = (double*) malloc (LAYER7_PARAMS * NUM * sizeof(double));
	double *Layer7_Weights_Bias_CPU = (double*) malloc (LAYER7_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer8_Weights_CPU = (double*) malloc (LAYER8_PARAMS * NUM * sizeof(double));
	double *Layer8_Weights_Bias_CPU = (double*) malloc (LAYER8_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer9_Weights_CPU = (double*) malloc (LAYER9_PARAMS * NUM * sizeof(double));
	double *Layer9_Weights_Bias_CPU = (double*) malloc (LAYER9_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer10_Weights_CPU = (double*) malloc (LAYER10_PARAMS * NUM * sizeof(double));
	double *Layer10_Weights_Bias_CPU = (double*) malloc (LAYER10_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer11_Weights_CPU = (double*) malloc (LAYER11_PARAMS * NUM * sizeof(double));
	double *Layer11_Weights_Bias_CPU = (double*) malloc (LAYER11_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer12_Weights_CPU = (double*) malloc (LAYER12_PARAMS * NUM * sizeof(double));
	double *Layer12_Weights_Bias_CPU = (double*) malloc (LAYER12_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer13_Weights_CPU = (double*) malloc (LAYER13_PARAMS * NUM * sizeof(double));
	double *Layer13_Weights_Bias_CPU = (double*) malloc (LAYER13_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer14_Weights_CPU = (double*) malloc (LAYER14_PARAMS * NUM * sizeof(double));
	double *Layer14_Weights_Bias_CPU = (double*) malloc (LAYER14_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer15_Weights_CPU = (double*) malloc (LAYER15_PARAMS * NUM * sizeof(double));
	double *Layer15_Weights_Bias_CPU = (double*) malloc (LAYER15_BIAS_PARAMS * NUM * sizeof(double));
	double *Layer16_Weights_CPU = (double*) malloc (LAYER16_PARAMS * NUM * sizeof(double));
	double *Layer16_Weights_Bias_CPU = (double*) malloc (LAYER16_BIAS_PARAMS * NUM * sizeof(double));
	//Allocation of host memory for input data
	double *Data_Layer_CPU_R = (double*) malloc (IMAGE_WIDTH*IMAGE_WIDTH*NUM*sizeof(double));
	double *Data_Layer_CPU_G = (double*) malloc (IMAGE_WIDTH*IMAGE_WIDTH*NUM*sizeof(double));
	double *Data_Layer_CPU_B = (double*) malloc (IMAGE_WIDTH*IMAGE_WIDTH*NUM*sizeof(double));
	//Allocation of device memory for input data
	double *Data_Layer_GPU_R;
	double *Data_Layer_GPU_G;
	double *Data_Layer_GPU_B;
	double *Data_Layer_CPU = (double*) malloc (CHANNELS*IMAGE_WIDTH*IMAGE_WIDTH*NUM*sizeof(double));
#if 1
	InitWeights_Biases(Layer1_Weights_CPU,LAYER1_PARAMS, (char *)"data/conv1_1_v.txt");
	InitWeights_Biases(Layer1_Weights_Bias_CPU,LAYER1_BIAS_PARAMS, (char *)"data/conv1_1_v_bias.txt");
	InitWeights_Biases(Layer2_Weights_CPU,LAYER2_PARAMS, (char *)"data/conv1_2_v.txt");
	InitWeights_Biases(Layer2_Weights_Bias_CPU,LAYER2_BIAS_PARAMS, (char *)"data/conv1_2_v_bias.txt");
	InitWeights_Biases(Layer3_Weights_CPU,LAYER3_PARAMS, (char *)"data/conv2_1_v.txt");
	InitWeights_Biases(Layer3_Weights_Bias_CPU,LAYER3_BIAS_PARAMS, (char *)"data/conv2_1_v_bias.txt");
	InitWeights_Biases(Layer4_Weights_CPU,LAYER4_PARAMS, (char *)"data/conv2_2_v.txt");
	InitWeights_Biases(Layer4_Weights_Bias_CPU,LAYER4_BIAS_PARAMS, (char *)"data/conv2_2_v_bias.txt");
	InitWeights_Biases(Layer5_Weights_CPU,LAYER5_PARAMS, (char *)"data/conv3_1_v.txt");
	InitWeights_Biases(Layer5_Weights_Bias_CPU,LAYER5_BIAS_PARAMS, (char *)"data/conv3_1_v_bias.txt");
	InitWeights_Biases(Layer6_Weights_CPU,LAYER6_PARAMS, (char *)"data/conv3_2_v.txt");
	InitWeights_Biases(Layer6_Weights_Bias_CPU,LAYER6_BIAS_PARAMS, (char *)"data/conv3_2_v_bias.txt");
	InitWeights_Biases(Layer7_Weights_CPU,LAYER7_PARAMS, (char *)"data/conv3_3_v.txt");
	InitWeights_Biases(Layer7_Weights_Bias_CPU,LAYER7_BIAS_PARAMS, (char *)"data/conv3_3_v_bias.txt");
	InitWeights_Biases(Layer8_Weights_CPU,LAYER8_PARAMS, (char *)"data/conv4_1_v.txt");
	InitWeights_Biases(Layer8_Weights_Bias_CPU,LAYER8_BIAS_PARAMS, (char *)"data/conv4_1_v_bias.txt");
	InitWeights_Biases(Layer9_Weights_CPU,LAYER9_PARAMS, (char *)"data/conv4_2_v.txt");
	InitWeights_Biases(Layer9_Weights_Bias_CPU,LAYER9_BIAS_PARAMS, (char *)"data/conv4_2_v_bias.txt");
	InitWeights_Biases(Layer10_Weights_CPU,LAYER10_PARAMS, (char *)"data/conv4_3_v.txt");
	InitWeights_Biases(Layer10_Weights_Bias_CPU,LAYER10_BIAS_PARAMS, (char *)"data/conv4_3_v_bias.txt");
	InitWeights_Biases(Layer11_Weights_CPU,LAYER11_PARAMS, (char *)"data/conv5_1_v.txt");
	InitWeights_Biases(Layer11_Weights_Bias_CPU,LAYER11_BIAS_PARAMS, (char *)"data/conv5_1_v_bias.txt");
	InitWeights_Biases(Layer12_Weights_CPU,LAYER12_PARAMS, (char *)"data/conv5_2_v.txt");
	InitWeights_Biases(Layer12_Weights_Bias_CPU,LAYER12_BIAS_PARAMS, (char *)"data/conv5_2_v_bias.txt");
	InitWeights_Biases(Layer13_Weights_CPU,LAYER13_PARAMS, (char *)"data/conv5_3_v.txt");
	InitWeights_Biases(Layer13_Weights_Bias_CPU,LAYER13_BIAS_PARAMS, (char *)"data/conv5_3_v_bias.txt");
	InitWeights_Biases(Layer14_Weights_CPU,LAYER14_PARAMS, (char *)"data/fc6_v.txt");
	InitWeights_Biases(Layer14_Weights_Bias_CPU,LAYER14_BIAS_PARAMS, (char *)"data/fc6_v_bias.txt");
	InitWeights_Biases(Layer15_Weights_CPU,LAYER15_PARAMS, (char *)"data/fc7_v.txt");
	InitWeights_Biases(Layer15_Weights_Bias_CPU,LAYER15_BIAS_PARAMS, (char *)"data/fc7_v_bias.txt");
	InitWeights_Biases(Layer16_Weights_CPU,LAYER16_PARAMS, (char *)"data/fc8_v.txt");
	InitWeights_Biases(Layer16_Weights_Bias_CPU,LAYER16_BIAS_PARAMS, (char *)"data/fc8_v_bias.txt");
#endif
	LoadInput(Data_Layer_CPU,file_path);
	ConvertInput(Data_Layer_CPU_R, Data_Layer_CPU_G, Data_Layer_CPU_B, Data_Layer_CPU);
#if 1
	double *Layer1_Features; // output no. of features of layer 1
	double *Layer1_Weights_GPU; // copied layer1 weights from cpu to this gpu variable
	double *Layer1_Features_CPU = (double*) malloc (IMAGE_WIDTH*IMAGE_WIDTH*64* NUM * sizeof(double)); // only used for displaying layer1 output
	err = cudaMalloc((void**) &Layer1_Features, IMAGE_WIDTH*IMAGE_WIDTH*64* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void**) &Layer1_Weights_GPU, LAYER1_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void**) &Data_Layer_GPU_R, IMAGE_WIDTH*IMAGE_WIDTH* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void**) &Data_Layer_GPU_G, IMAGE_WIDTH*IMAGE_WIDTH* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void**) &Data_Layer_GPU_B, IMAGE_WIDTH*IMAGE_WIDTH* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("Malloc completed\n");
	cudaMemcpy(Layer1_Weights_GPU,Layer1_Weights_CPU, sizeof(double)*LAYER1_PARAMS*NUM, cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to copy Layer1 weights (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Data_Layer_GPU_R,Data_Layer_CPU_R, IMAGE_WIDTH*IMAGE_WIDTH* NUM * sizeof(double), cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to copy R (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Data_Layer_GPU_G,Data_Layer_CPU_G, IMAGE_WIDTH*IMAGE_WIDTH* NUM * sizeof(double), cudaMemcpyHostToDevice);
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to copy G (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(Data_Layer_GPU_B,Data_Layer_CPU_B, IMAGE_WIDTH*IMAGE_WIDTH* NUM * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("Memcpy completed\n");
	double *Layer1_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer1_Weights_Bias_GPU, LAYER1_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer1_Weights_Bias_GPU,Layer1_Weights_Bias_CPU, LAYER1_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);

	dim3 n_blocks(7,7,1);
	dim3 n_threads(32,32,1);
	cudaThreadSynchronize();

	printf("Executing First Layer\n");
	//Execute First Layer
	ExecuteFirstLayer<<<n_blocks,n_threads>>>(Layer1_Weights_GPU, Data_Layer_GPU_R, Data_Layer_GPU_G, Data_Layer_GPU_B, Layer1_Features,Layer1_Weights_Bias_GPU);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "1st LayerKernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer1_Features, 64*IMAGE_WIDTH*IMAGE_WIDTH* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<64*IMAGE_WIDTH*IMAGE_WIDTH* NUM; i++){

		//if(Layer1_Features_CPU[i] == 0)
			printf("\n%.17f", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer1_Weights_GPU);
	double *Layer2_Weights_GPU;
	err = cudaMalloc((void**) &Layer2_Weights_GPU, LAYER2_PARAMS * NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer2_Weights_GPU,Layer2_Weights_CPU, LAYER2_PARAMS * NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer2_Features;
	err = cudaMalloc((void**) &Layer2_Features, 64*IMAGE_WIDTH*IMAGE_WIDTH* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer2_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer2_Weights_Bias_GPU, LAYER2_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer2_Weights_Bias_GPU,Layer2_Weights_Bias_CPU, LAYER2_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);

	printf("Executing Second Layer\n");
	//Execute Second Layer
	ExecuteConvLayer<<<n_blocks,n_threads>>>(Layer2_Weights_GPU, Layer2_Features, Layer1_Features, 64, 64, IMAGE_WIDTH,Layer2_Weights_Bias_GPU);
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "2nd Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer2_Features, IMAGE_WIDTH*IMAGE_WIDTH*64* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//for(int i=0; i<IMAGE_WIDTH*IMAGE_WIDTH*64* NUM; i++){

//		if(Layer1_Features_CPU[i] < 0)
			printf("%.17f\n", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer2_Weights_GPU);
//--------------------------------------
	double *Pool_Layer_Features;
	err = cudaMalloc((void**) &Pool_Layer_Features, 64*(IMAGE_WIDTH/2)*(IMAGE_WIDTH/2)* NUM * sizeof(double));
	printf("Executing First Pool Layer\n");
	pooling<<<n_blocks,n_threads>>>(Layer2_Features, Pool_Layer_Features, 64 /*IMAGE DEPTH*/, IMAGE_WIDTH/2, IMAGE_WIDTH/2, 2/*KERENEL*/, 2/*STRIDE*/, IMAGE_WIDTH, IMAGE_WIDTH);
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "1st Pool Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();
#if 0
	double *Pool_Layer_Features_cpu = (double*) malloc(64*(IMAGE_WIDTH/2)*(IMAGE_WIDTH/2)* NUM * sizeof(double));
	cudaMemcpy(Pool_Layer_Features_cpu, Pool_Layer_Features, 64*(IMAGE_WIDTH/2)*(IMAGE_WIDTH/2)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<64*(IMAGE_WIDTH/2)*(IMAGE_WIDTH/2)* NUM; i++){

		//if(Pool_Layer_Features_cpu[i] < 0)
			printf("\n%lf", Pool_Layer_Features_cpu[i]);
	}
#endif
	cudaFree(Layer2_Features);

	//---------------------------------------------------------------------------------------------------------------------------------------

	double *Layer3_Weights_GPU;
	err = cudaMalloc((void**) &Layer3_Weights_GPU, LAYER3_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer3_Weights_GPU,Layer3_Weights_CPU, LAYER3_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer3_Features;
	err = cudaMalloc((void**) &Layer3_Features, 128*(IMAGE_WIDTH/2)*(IMAGE_WIDTH/2)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer3_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer3_Weights_Bias_GPU, LAYER3_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer3_Weights_Bias_GPU,Layer3_Weights_Bias_CPU, LAYER3_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);

	printf("Executing Third Layer\n");
	//Execute Third Layer
	dim3 n1_blocks(4,4,1);
	dim3 n1_threads(28,28,1);
	ExecuteConvLayer<<<n1_blocks,n1_threads>>>(Layer3_Weights_GPU, Layer3_Features, Pool_Layer_Features, 128, 64, IMAGE_WIDTH/2,Layer3_Weights_Bias_GPU);
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "3rd Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer3_Features, 128*(IMAGE_WIDTH/2)*(IMAGE_WIDTH/2)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<128*(IMAGE_WIDTH/2)*(IMAGE_WIDTH/2)* NUM; i++){

		//if(Layer1_Features_CPU[i] < 0)
			printf("%.17f\n", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Pool_Layer_Features);
	cudaFree(Layer3_Weights_GPU);
//-------------------------------------------------------------------------------

	double *Layer4_Weights_GPU;
	err = cudaMalloc((void**) &Layer4_Weights_GPU, LAYER4_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer4_Weights_GPU,Layer4_Weights_CPU, LAYER4_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer4_Features;
	err = cudaMalloc((void**) &Layer4_Features, 128*(IMAGE_WIDTH/2)*(IMAGE_WIDTH/2)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer4_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer4_Weights_Bias_GPU, LAYER4_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(Layer4_Weights_Bias_GPU,Layer4_Weights_Bias_CPU, LAYER4_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);

	printf("Executing Fourth Layer\n");
	//Execute Fourth Layer
	ExecuteConvLayer<<<n1_blocks,n1_threads>>>(Layer4_Weights_GPU, Layer4_Features, Layer3_Features, 128, 128, IMAGE_WIDTH/2,Layer4_Weights_Bias_GPU);
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "4th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer4_Features, 128*(IMAGE_WIDTH/2)*(IMAGE_WIDTH/2)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<128*(IMAGE_WIDTH/2)*(IMAGE_WIDTH/2)* NUM; i++){

		//if(Layer1_Features_CPU[i] < 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer4_Weights_GPU);
	cudaFree(Layer3_Features);
	//----------------------------------------------------------------------------------------------
	double *Pool2_Layer_Features;
	err = cudaMalloc((void**) &Pool2_Layer_Features, 128*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM * sizeof(double));
	printf("Executing Second Pool\n");
	pooling<<<n1_blocks,n1_threads>>>(Layer4_Features, Pool2_Layer_Features, 128 /*IMAGE DEPTH*/, IMAGE_WIDTH/4, IMAGE_WIDTH/4, 2/*KERENEL*/, 2/*STRIDE*/, IMAGE_WIDTH/2, IMAGE_WIDTH/2);
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "2nd Pool Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaThreadSynchronize();
#if 0
	double *Pool2_Layer_Features_cpu = (double*) malloc(128*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM * sizeof(double));
	cudaMemcpy(Pool2_Layer_Features_cpu, Pool2_Layer_Features, 128*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<128*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM; i++){

		//if(Pool_Layer_Features_cpu[i] < 0)
			printf("\n%lf", Pool2_Layer_Features_cpu[i]);
	}
#endif
	cudaFree(Layer4_Features);

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------
	double *Layer5_Weights_GPU;
	err = cudaMalloc((void**) &Layer5_Weights_GPU, LAYER5_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer5_Weights_GPU,Layer5_Weights_CPU, LAYER5_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer5_Features;
	err = cudaMalloc((void**) &Layer5_Features, 256*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer5_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer5_Weights_Bias_GPU, LAYER5_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(Layer5_Weights_Bias_GPU,Layer5_Weights_Bias_CPU, LAYER5_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);

	printf("Executing Fifth Layer\n");
	//Execute Fifth Layer
	dim3 n2_blocks(7,7,1);
	dim3 n2_threads(8,8,1);
	ExecuteConvLayer<<<n2_blocks,n2_threads>>>(Layer5_Weights_GPU, Layer5_Features, Pool2_Layer_Features, 256, 128, IMAGE_WIDTH/4,Layer5_Weights_Bias_GPU);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "5th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer5_Features, 256*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<256*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM; i++){

		//if(Layer1_Features_CPU[i] < 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer5_Weights_GPU);
	cudaFree(Pool2_Layer_Features);
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------

	double *Layer6_Weights_GPU;
	err = cudaMalloc((void**) &Layer6_Weights_GPU, LAYER6_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer6_Weights_GPU,Layer6_Weights_CPU, LAYER6_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer6_Features;
	err = cudaMalloc((void**) &Layer6_Features, 256*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer6_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer6_Weights_Bias_GPU, LAYER6_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemcpy(Layer6_Weights_Bias_GPU,Layer6_Weights_Bias_CPU, LAYER6_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);


	printf("Executing Sixth Layer\n");
	//Execute Sixth Layer
	ExecuteConvLayer<<<n2_blocks,n2_threads>>>(Layer6_Weights_GPU, Layer6_Features, Layer5_Features, 256, 256, IMAGE_WIDTH/4,Layer6_Weights_Bias_GPU);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "6th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer6_Features, 256*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<256*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM; i++){

		//if(Layer1_Features_CPU[i] < 0)
			printf("%.17f\n", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer6_Weights_GPU);
	cudaFree(Layer5_Features);
	
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------

	double *Layer7_Weights_GPU;
	err = cudaMalloc((void**) &Layer7_Weights_GPU, LAYER7_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer7_Weights_GPU,Layer7_Weights_CPU, LAYER7_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer7_Features;
	err = cudaMalloc((void**) &Layer7_Features, 256*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	
	double *Layer7_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer7_Weights_Bias_GPU, LAYER7_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer7_Weights_Bias_GPU,Layer7_Weights_Bias_CPU, LAYER7_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	//Execute Seventh Layer
	printf("Executing Seventh Layer\n");
	ExecuteConvLayer<<<n2_blocks,n2_threads>>>(Layer7_Weights_GPU, Layer7_Features, Layer6_Features, 256, 256, IMAGE_WIDTH/4,Layer7_Weights_Bias_GPU);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "7th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer7_Features, 256*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<256*(IMAGE_WIDTH/4)*(IMAGE_WIDTH/4)* NUM; i++){

	//	if(Layer1_Features_CPU[i] < 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer7_Weights_GPU);
	cudaFree(Layer6_Features);
	//-------------------------------------------------------------------------------
	double *Pool3_Layer_Features;
	err = cudaMalloc((void**) &Pool3_Layer_Features, 256*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM * sizeof(double));
	printf("Executing Third Pool Layer\n");
	pooling<<<n2_blocks,n2_threads>>>(Layer7_Features, Pool3_Layer_Features, 256 /*IMAGE DEPTH*/, IMAGE_WIDTH/8, IMAGE_WIDTH/8, 2/*KERENEL*/, 2/*STRIDE*/, IMAGE_WIDTH/4, IMAGE_WIDTH/4);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "3rd Pool Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	double *Pool3_Layer_Features_cpu = (double*) malloc(256*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM * sizeof(double));
	cudaMemcpy(Pool3_Layer_Features_cpu, Pool3_Layer_Features, 256*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	for(int i=0; i<256*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM; i++){

		if(Pool3_Layer_Features_cpu[i] < 0)
			printf("\n%lf", Pool3_Layer_Features_cpu[i]);
	}
#endif
	cudaFree(Layer7_Features);

	//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#if 1
	double *Layer8_Weights_GPU;
	err = cudaMalloc((void**) &Layer8_Weights_GPU, LAYER8_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer8_Weights_GPU,Layer8_Weights_CPU, LAYER8_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer8_Features;
	err = cudaMalloc((void**) &Layer8_Features, 512*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	double *Layer8_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer8_Weights_Bias_GPU, LAYER8_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer8_Weights_Bias_GPU,Layer8_Weights_Bias_CPU, LAYER8_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	printf("Executing Eighth Layer\n");
	//Execute Eighth Layer
	dim3 n3_blocks(1,1,1);
	dim3 n3_threads(28,28,1);
	ExecuteConvLayer<<<n3_blocks,n3_threads>>>(Layer8_Weights_GPU, Layer8_Features, Pool3_Layer_Features, 512, 256, IMAGE_WIDTH/8,Layer8_Weights_Bias_GPU);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "8th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer8_Features, 512*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<512*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM; i++){

		if(Layer1_Features_CPU[i] < 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer8_Weights_GPU);
	cudaFree(Pool3_Layer_Features);
#endif
	//-------------------------------------------------------------------------------------------------------------------------------------
#if 1
	double *Layer9_Weights_GPU;
	err = cudaMalloc((void**) &Layer9_Weights_GPU, LAYER9_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer9_Weights_GPU,Layer9_Weights_CPU, LAYER9_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer9_Features;
	err = cudaMalloc((void**) &Layer9_Features, 512*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer9_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer9_Weights_Bias_GPU, LAYER9_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer9_Weights_Bias_GPU,Layer9_Weights_Bias_CPU, LAYER9_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	
	printf("Executing Ninth Layer\n");
	//Execute Ninth Layer
	ExecuteConvLayer<<<n3_blocks,n3_threads>>>(Layer9_Weights_GPU, Layer9_Features, Layer8_Features, 512, 512, IMAGE_WIDTH/8,Layer9_Weights_Bias_GPU);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "9th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer9_Features, 512*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<512*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM; i++){

		if(Layer1_Features_CPU[i] < 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer9_Weights_GPU);
	cudaFree(Layer8_Features);
#endif
	//------------------------------------------------------------------------------------------------------------------------
#if 1
	double *Layer10_Weights_GPU;
	err = cudaMalloc((void**) &Layer10_Weights_GPU, LAYER10_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer10_Weights_GPU,Layer10_Weights_CPU, LAYER10_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer10_Features;
	err = cudaMalloc((void**) &Layer10_Features, 512*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer10_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer10_Weights_Bias_GPU, LAYER10_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer10_Weights_Bias_GPU,Layer10_Weights_Bias_CPU, LAYER10_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	

	printf("Executing Tenth Layer\n");
	//Execute Tenth Layer
	ExecuteConvLayer<<<n3_blocks,n3_threads>>>(Layer10_Weights_GPU, Layer10_Features, Layer9_Features, 512, 512, IMAGE_WIDTH/8,Layer10_Weights_Bias_GPU);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "10th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer10_Features, 512*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<512*(IMAGE_WIDTH/8)*(IMAGE_WIDTH/8)* NUM; i++){

		if(Layer1_Features_CPU[i] < 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer10_Weights_GPU);
	cudaFree(Layer9_Features);
#endif
	//------------------------------------------------------------------------------------------------------------------------------
	double *Pool4_Layer_Features;
	err = cudaMalloc((void**) &Pool4_Layer_Features, 512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM * sizeof(double));
	printf("Executing Fourth Pool Layer\n");
	pooling<<<n3_blocks,n3_threads>>>(Layer10_Features, Pool4_Layer_Features, 512 /*IMAGE DEPTH*/, IMAGE_WIDTH/16, IMAGE_WIDTH/16, 2/*KERENEL*/, 2/*STRIDE*/, IMAGE_WIDTH/8, IMAGE_WIDTH/8);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "4th Pool Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	double *Pool4_Layer_Features_cpu = (double*) malloc(512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM * sizeof(double));
	cudaMemcpy(Pool4_Layer_Features_cpu, Pool4_Layer_Features, 512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	for(int i=0; i<512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM; i++){

		if(Pool4_Layer_Features_cpu[i] < 0)
			printf("\n%lf", Pool4_Layer_Features_cpu[i]);
	}
#endif
	cudaFree(Layer10_Features);
	//------------------------------------------------------------------------------------------------------------------------
#if 1
	double *Layer11_Weights_GPU;
	err = cudaMalloc((void**) &Layer11_Weights_GPU, LAYER11_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer11_Weights_GPU,Layer11_Weights_CPU, LAYER11_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer11_Features;
	err = cudaMalloc((void**) &Layer11_Features, 512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer11_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer11_Weights_Bias_GPU, LAYER11_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer11_Weights_Bias_GPU,Layer11_Weights_Bias_CPU, LAYER11_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);

	printf("Executing Eleventh Layer\n");
	//Execute 11th Layer
	dim3 n4_blocks(1,1,1);
	dim3 n4_threads(14,14,1);
	ExecuteConvLayer<<<n4_blocks,n4_threads>>>(Layer11_Weights_GPU, Layer11_Features, Pool4_Layer_Features, 512, 512, IMAGE_WIDTH/16,Layer11_Weights_Bias_GPU);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "11th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer11_Features, 512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM; i++){

		if(Layer1_Features_CPU[i] < 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer11_Weights_GPU);
	cudaFree(Pool4_Layer_Features);
#endif

	//------------------------------------------------------------------------------------------------------------------------------
#if 1
	double *Layer12_Weights_GPU;
	err = cudaMalloc((void**) &Layer12_Weights_GPU, LAYER12_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer12_Weights_GPU,Layer12_Weights_CPU, LAYER12_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer12_Features;
	err = cudaMalloc((void**) &Layer12_Features, 512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer12_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer12_Weights_Bias_GPU, LAYER12_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer12_Weights_Bias_GPU,Layer12_Weights_Bias_CPU, LAYER12_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);


	printf("Executing Twelvth Layer\n");
	//Execute 12th Layer
	ExecuteConvLayer<<<n4_blocks,n4_threads>>>(Layer12_Weights_GPU, Layer12_Features, Layer11_Features, 512, 512, IMAGE_WIDTH/16,Layer12_Weights_Bias_GPU);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "12th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer12_Features, 512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM; i++){

		if(Layer1_Features_CPU[i] < 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer12_Weights_GPU);
	cudaFree(Layer11_Features);
#endif
	//------------------------------------------------------------------------------------------------------------------------------
#if 1
	double *Layer13_Weights_GPU;
	err = cudaMalloc((void**) &Layer13_Weights_GPU, LAYER13_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer13_Weights_GPU,Layer13_Weights_CPU, LAYER13_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer13_Features;
	err = cudaMalloc((void**) &Layer13_Features, 512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer13_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer13_Weights_Bias_GPU, LAYER13_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer13_Weights_Bias_GPU,Layer13_Weights_Bias_CPU, LAYER13_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);

	printf("Executing Thirteenth Layer\n");
	//Execute 13th Layer
	ExecuteConvLayer<<<n4_blocks,n4_threads>>>(Layer13_Weights_GPU, Layer13_Features, Layer12_Features, 512, 512, IMAGE_WIDTH/16,Layer13_Weights_Bias_GPU);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "13th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer13_Features, 512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for(int i=0; i<512*(IMAGE_WIDTH/16)*(IMAGE_WIDTH/16)* NUM; i++){

	//	if(Layer1_Features_CPU[i] < 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer13_Weights_GPU);
	cudaFree(Layer12_Features);
#endif
	//------------------------------------------------------------------------------------------------------------------------------
	double *Pool5_Layer_Features;
	err = cudaMalloc((void**) &Pool5_Layer_Features, 512*(IMAGE_WIDTH/32)*(IMAGE_WIDTH/32)* NUM * sizeof(double));
	printf("Executing Fifth Pool Layer\n");
	pooling<<<n4_blocks,n4_threads>>>(Layer13_Features, Pool5_Layer_Features, 512 /*IMAGE DEPTH*/, IMAGE_WIDTH/32, IMAGE_WIDTH/32, 2/*KERENEL*/, 2/*STRIDE*/, IMAGE_WIDTH/16, IMAGE_WIDTH/16);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "4th Pool Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	double *Pool5_Layer_Features_cpu = (double*) malloc(512*(IMAGE_WIDTH/32)*(IMAGE_WIDTH/32)* NUM * sizeof(double));
	cudaMemcpy(Pool5_Layer_Features_cpu, Pool5_Layer_Features, 512*(IMAGE_WIDTH/32)*(IMAGE_WIDTH/32)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	for(int i=0; i<512*(IMAGE_WIDTH/32)*(IMAGE_WIDTH/32)* NUM; i++){

	//	if(Pool5_Layer_Features_cpu[i] < 0)
			printf("%.17f\n", Pool5_Layer_Features_cpu[i]);
	}
#endif
	cudaFree(Layer13_Features);
	//------------------------------------------------------------------------------------------------------------------------
	double *Layer14_Weights_GPU;
	err = cudaMalloc((void**) &Layer14_Weights_GPU, LAYER14_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer14_Weights_GPU,Layer14_Weights_CPU, LAYER14_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer14_Features;
	err = cudaMalloc((void**) &Layer14_Features, 4096 * NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer14_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer14_Weights_Bias_GPU, LAYER14_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer14_Weights_Bias_GPU,Layer14_Weights_Bias_CPU, LAYER14_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	printf("Executing First FC Layer\n");
	//Execute 14th Layer -- fully-connected1

	dim3 n5_blocks(4,1,1);
	dim3 n5_threads(1024,1,1);
	ExecuteFcLayer<<<n5_blocks,n5_threads>>>(Layer14_Weights_GPU,Layer14_Features,Pool5_Layer_Features, 7, 512, Layer14_Weights_Bias_GPU);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "14th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer14_Features, 4096*(IMAGE_WIDTH/224)*(IMAGE_WIDTH/224)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	for(int i=0; i<4096*(IMAGE_WIDTH/224)*(IMAGE_WIDTH/224)* NUM; i++){

	//	if(Layer1_Features_CPU[i] != 0)
			printf("%.17f\n", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Pool5_Layer_Features);
	cudaFree(Layer14_Weights_GPU);
	cudaFree(Layer14_Weights_Bias_GPU);
	//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#endif
#if 1
	//	bool use_relu = true;
	double *Layer15_Weights_GPU;
	err = cudaMalloc((void**) &Layer15_Weights_GPU, LAYER15_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer15_Weights_GPU, Layer15_Weights_CPU, LAYER15_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer15_Features;
	err = cudaMalloc((void**) &Layer15_Features, 4096 * NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer15_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer15_Weights_Bias_GPU, LAYER15_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer15_Weights_Bias_GPU,Layer15_Weights_Bias_CPU, LAYER15_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	printf("Executing Second FC Layer\n");
	//Execute 15th Layer -- fully-connected2

	ExecuteFc1Layer<<<n5_blocks,n5_threads>>>(Layer15_Weights_GPU,Layer15_Features,Layer14_Features, 4096, Layer15_Weights_Bias_GPU,true);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "15th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer15_Features, 4096*(IMAGE_WIDTH/224)*(IMAGE_WIDTH/224)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	for(int i=0; i<4096*(IMAGE_WIDTH/224)*(IMAGE_WIDTH/224)* NUM; i++){

		if(Layer1_Features_CPU[i] != 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer14_Features);
	cudaFree(Layer15_Weights_GPU);
	cudaFree(Layer15_Weights_Bias_GPU);
#endif
#if 1

	double *Layer16_Weights_GPU;
	err = cudaMalloc((void**) &Layer16_Weights_GPU, LAYER16_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer16_Weights_GPU, Layer16_Weights_CPU, LAYER16_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	double *Layer16_Features;
	err = cudaMalloc((void**) &Layer16_Features, 4096 * NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	double *Layer16_Weights_Bias_GPU;
	err = cudaMalloc((void**) &Layer16_Weights_Bias_GPU, LAYER16_BIAS_PARAMS* NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer16_Weights_Bias_GPU,Layer16_Weights_Bias_CPU, LAYER16_BIAS_PARAMS* NUM * sizeof(double), cudaMemcpyHostToDevice);
	//Execute 16th Layer -- fully-connected3
	printf("Executing Third FC Layer\n");
	//bool use_relu = false;
	ExecuteFc1Layer<<<1,1000>>>(Layer16_Weights_GPU,Layer16_Features,Layer15_Features, 4096, Layer16_Weights_Bias_GPU,false);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "16th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0
	cudaMemcpy(Layer1_Features_CPU, Layer16_Features, 1000*(IMAGE_WIDTH/224)*(IMAGE_WIDTH/224)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	for(int i=0; i<1000*(IMAGE_WIDTH/224)*(IMAGE_WIDTH/224)* NUM; i++){

		if(Layer1_Features_CPU[i] != 0)
			printf("\n%lf", Layer1_Features_CPU[i]);
	}
#endif
	cudaFree(Layer15_Features);
	cudaFree(Layer16_Weights_GPU);
	cudaFree(Layer16_Weights_Bias_GPU);
#endif
#if 1
	//---------------------------------------------------------------------------------------------
	//Softmax layer
	double *Layer17_Features;
	err = cudaMalloc((void**) &Layer17_Features, 1000 * NUM * sizeof(double));
	if (err != cudaSuccess){

		fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#if 0

	// finding max valus in cpu side
	double *layer16_c = (double*)malloc(1000*sizeof(double));
	cudaMemcpy(layer16_c,Layer16_Features,  1000*(IMAGE_WIDTH/224)*(IMAGE_WIDTH/224)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	
	printf("Running Softmax layer\n");
	double m = -INFINITY;
	for(int i = 0;i<1000;i++) {
		if (layer16_c[i] > m)
			m = layer16_c[i];
	}
	
	softmax<<<1,1000>>>(Layer17_Features, Layer16_Features,m);
#endif
	softmax<<<1,1000>>>(Layer17_Features, Layer16_Features);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "17th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(Layer1_Features_CPU, Layer17_Features, 1000*(IMAGE_WIDTH/224)*(IMAGE_WIDTH/224)* NUM * sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaPeekAtLastError();
	if (err != cudaSuccess){

		fprintf(stderr, "cudaMemCpy Failed (error code %s)\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    double maxf[5] = {0};
    int idxp[5] = {0};
    for (int j = 0; j < 5; j++) {
        double max = Layer1_Features_CPU[0];
        int index = 0;
        for(int i=0; i<1000*(IMAGE_WIDTH/224)*(IMAGE_WIDTH/224)* NUM; i++){
            if (max < Layer1_Features_CPU[i]) {
                max = Layer1_Features_CPU[i];
                index = i;
            }
        }
        maxf[j] = max;
        idxp[j] = index;
        Layer1_Features_CPU[index] = 0;

    }

    char* image_class[1000] = {0};

    for(int i = 0;i<1000;i++) {
        image_class[i] = (char*) malloc(125*sizeof(char));
    }

    LoadImageNetClass(image_class,(char*)"data/imagenet_classes.txt");

    for (int i = 0;i<5;i++)
        printf("\n%.17f at index %d - class %s\n",maxf[i],idxp[i],image_class[idxp[i]]);
#endif
}
