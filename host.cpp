#include<stdio.h>
#include<stdlib.h>
#include<CL/cl.h>
#include<string.h>
#include<iostream> 
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <cmath>
#include <time.h>
//#include "basic.h"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#define TOTAL 10000
using namespace std;
using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
cl_device_id device; // num_devices elements
cl_context context = NULL;
cl_command_queue queue; // num_devices elements
cl_program program = NULL;
cl_kernel kernel_1; //
cl_kernel kernel_2;
#if USE_SVM_API == 0
cl_mem input_buf; // num_devices elements
cl_mem weight1_traint_buf; // num_devices elements
cl_mem weight2_traint_buf; // num_devices elements
cl_mem b1_traint_buf; // num_devices elements
cl_mem b2_traint_buf; // num_devices elements
cl_mem output1_buf;
cl_mem output2_buf; // num_devices elements
#endif /* USE_SVM_API == 0 */

					// Problem data.
const int first = 784;
const int second = 100;
const int third = 10;
const float alpha = 0.35;

int input[first];
int target[third];
float weight1[first][second];
float weight2[second][third];
float output1[second];
float output2[third];
float delta1[second];
float delta2[third];
float b1[second];
float b2[third];

float test_num = 0.0;
float test_success_count = 0.0;

char *input_train; // num_devices elements
float *weight1_traint; // num_devices elements
float *weight2_traint; // num_devices elements
float *b1_train; // num_devices elements
float *b2_train; // num_devices elements
float *output1_train;
float *output2_train; // num_devices elements
char *target_train;

bool init_opencl() {
	cl_int status;

	printf("Initializing OpenCL\n");

	if (!setCwdToExeDir()) {
		return false;
	}

	// Get the OpenCL platform.
	platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
	if (platform == NULL) {
		printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
		return false;
	}
	cl_int errNum;
	// Query the available OpenCL device.
	errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, &num_devices);  //返回设备ID和设备数
	printf("Platform: %s\n", getPlatformName(platform).c_str());
	printf("Using %d device(s)\n", num_devices);
	for (unsigned i = 0; i < num_devices; ++i) {
		printf("  %s\n", getDeviceName(device).c_str());
	}

	// Create the context.
	context = clCreateContext(NULL, num_devices, &device, &oclContextCallback, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the program for all device. Use the first device as the
	// representative device (assuming all device are of the same type).
	std::string binary_file = getBoardBinaryFile("device_17", device);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, num_devices);

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");


	// Command queue.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Kernel.
	kernel_1 = clCreateKernel(program, "BP_Test_1", &status);
	kernel_2 = clCreateKernel(program, "BP_Test_2", &status);
	checkError(status, "Failed to create kernel");



	// Input buffers.
	// For matrix A, each device only needs the rows corresponding
	// to the rows of the output matrix. We specifically
	// assign this buffer to the first bank of global memory.
	input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		sizeof(cl_char) * TOTAL * 784, NULL, &status);
	checkError(status, "Failed to create buffer for input");

	// For matrix B, each device needs the whole matrix. We specifically
	// assign this buffer to the second bank of global memory.
	weight1_traint_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		784 * 100 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for weight1");

	weight2_traint_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		100 * 10 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for weight2");

	b1_traint_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		100 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for b1");

	b2_traint_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		10 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for b2");

	output1_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		TOTAL * 100 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for output_1");

	output2_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		TOTAL * 10 * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for output_2");

	return true;
}


float f_(float x) {
	return 1.0 / (1.0 + exp(-x));
}

void run() {
	cl_int status;

	status = clEnqueueWriteBuffer(queue, input_buf, CL_TRUE,
		0, TOTAL * 784 * sizeof(char), input_train, 0, NULL, NULL);
	checkError(status, "Failed to transfer input A");

	status = clEnqueueWriteBuffer(queue, weight1_traint_buf, CL_TRUE,
		0, 784 * 100 * sizeof(float), weight1_traint, 0, NULL, NULL);
	checkError(status, "Failed to transfer weight1");

	status = clEnqueueWriteBuffer(queue, weight2_traint_buf, CL_TRUE,
		0, 100 * 10 * sizeof(float), weight2_traint, 0, NULL, NULL);
	checkError(status, "Failed to transfer weight2");

	status = clEnqueueWriteBuffer(queue, b1_traint_buf, CL_TRUE,
		0, 100 * sizeof(float), b1_train, 0, NULL, NULL);
	checkError(status, "Failed to transfer b1");

	status = clEnqueueWriteBuffer(queue, b2_traint_buf, CL_TRUE,
		0, 10 * sizeof(float), b2_train, 0, NULL, NULL);
	checkError(status, "Failed to transfer b2");


	// Wait for all queues to finish.

	//clFinish(queue);




	cl_event kernel_event;

	// Set kernel arguments and execute kernel_1
	unsigned argi = 0;

	status = clSetKernelArg(kernel_1, argi++, sizeof(cl_mem), &input_buf);
	checkError(status, "Failed to set argument %d", argi - 1);

	status = clSetKernelArg(kernel_1, argi++, sizeof(cl_mem), &weight1_traint_buf);
	checkError(status, "Failed to set argument %d", argi - 1);

	status = clSetKernelArg(kernel_1, argi++, sizeof(cl_mem), &b1_traint_buf);
	checkError(status, "Failed to set argument %d", argi - 1);

	status = clSetKernelArg(kernel_1, argi++, sizeof(cl_mem), &output1_buf);
	checkError(status, "Failed to set argument %d", argi - 1);


	size_t global_work_size[] = { 10000, 100 };
	size_t local_work_size[] = { 112, 10 };
	printf("Launching for kernel %d (global size: %zd)\n", 1, global_work_size[0]);

	//float start_1 = getCurrentTimestamp();
	status = clEnqueueNDRangeKernel(queue, kernel_1, 2, NULL,
		global_work_size, NULL, 0, NULL, &kernel_event);
	checkError(status, "Failed to launch kernel_1");

	//float end_1 = getCurrentTimestamp();

	//compute the output1 and then put back
	/*status = clEnqueueReadBuffer(queue, output1_buf, CL_TRUE,
		0, TOTAL * 100 * sizeof(float), output1_train, 0, NULL, NULL);
	checkError(status, "Failed to read output1");

	for (int i = 0; i < 10000; i++) {
		for (int j = 0; j < 100; j++) {
			output1_train[i * 100 + j] = 1.0 / (1.0 + exp(-output1_train[i * 100 + j]));
		}
	}

	status = clEnqueueWriteBuffer(queue, output1_buf, CL_TRUE,
		0, TOTAL * 100 * sizeof(float), output1_train, 0, NULL, NULL);
	checkError(status, "Failed to input output1");*/

	//execute kernel_2
	unsigned argj = 0;
	status = clSetKernelArg(kernel_2, argj++, sizeof(cl_mem), &output1_buf);
	checkError(status, "Failed to set argument %d", argi - 1);

	status = clSetKernelArg(kernel_2, argj++, sizeof(cl_mem), &weight2_traint_buf);
	checkError(status, "Failed to set argument %d", argi - 1);

	status = clSetKernelArg(kernel_2, argj++, sizeof(cl_mem), &b2_traint_buf);
	checkError(status, "Failed to set argument %d", argi - 1);

	status = clSetKernelArg(kernel_2, argj++, sizeof(cl_mem), &output2_buf);
	checkError(status, "Failed to set argument %d", argi - 1);
	size_t global_work_size_1[] = { TOTAL,10 };
	//float start_2 = getCurrentTimestamp();
	printf("Launching for kernel %d (global size: %zd)\n", 2, global_work_size_1[0]);
	status = clEnqueueNDRangeKernel(queue, kernel_2, 2, NULL,
		global_work_size_1, NULL, 0, NULL, &kernel_event);
	checkError(status, "Failed to launch kernel_2");
	//float end_2 = getCurrentTimestamp();

	// Wait for all kernels to finish.
	//clWaitForEvents(1, &kernel_event);
	status = clFinish(queue);
	//float kernel_1_time = end_1 - start_1;
	//float kernel_2_time = end_2 - start_2;


	/*cl_ulong time_ns = getStartEndTime(kernel_event);
	printf("Kernel time (device %d): %0.3f ms\n", 0, float(time_ns) * 1e-6);*/

	// Wall-clock time taken.
	//printf("\nkernel_1_Time: %0.3f ms\n", kernel_1_time * 1e3);
	//printf("\nkeinel_2_Time: %0.3f ms\n", kernel_2_time * 1e3);



	// Release kernel events.
	clReleaseEvent(kernel_event);

	// Read the result.
	status = clEnqueueReadBuffer(queue, output2_buf, CL_TRUE,
		0, TOTAL * 10 * sizeof(float), output2_train, 0, NULL, NULL);
	checkError(status, "Failed to read output");

	//compute the result
	/*for (int i = 0; i < 10000; i++) {
		for (int j = 0; j < 10; j++) {
			output2_train[i * 10 + j] = 1.0 / (1.0 + exp(-output2_train[i * 10 + j]));
		}
	}*/
}

void cleanup() {
	if (kernel_1) {
		clReleaseKernel(kernel_1);
	}
	if (kernel_2) {
		clReleaseKernel(kernel_2);
	}
	if (queue) {
		clReleaseCommandQueue(queue);
	}
	if (input_buf) {
		clReleaseMemObject(input_buf);
	}
	if (weight1_traint_buf) {
		clReleaseMemObject(weight1_traint_buf);
	}
	if (weight2_traint_buf) {
		clReleaseMemObject(weight2_traint_buf);
	}
	if (b1_traint_buf) {
		clReleaseMemObject(b1_traint_buf);
	}
	if (b2_traint_buf) {
		clReleaseMemObject(b2_traint_buf);
	}
	if (output2_buf) {
		clReleaseMemObject(output2_buf);
	}
	if (program) {
		clReleaseProgram(program);
	}
	if (context) {
		clReleaseContext(context);
	}
}


void initialize() {
	srand((int)time(0) + rand());
	for (int i = 0; i < first; i++) {
		for (int j = 0; j < second; j++) {
			weight1[i][j] = rand() % 1000 * 0.001 - 0.5;
		}
	}
	for (int j = 0; j < second; j++) {
		for (int k = 0; k < third; k++) {
			weight2[j][k] = rand() % 1000 * 0.001 - 0.5;
		}
	}

	for (int j = 0; j < second; j++) {
		b1[j] = rand() % 1000 * 0.001 - 0.5;
	}
	for (int k = 0; k < third; k++) {
		b2[k] = rand() % 1000 * 0.001 - 0.5;
	}
}

void ReadFile(char *input, char *target) {
	//创建图像数据缓存并提取图像数据
	//open externel file
	FILE *image_train;
	FILE *image_label;
	image_train = fopen("E:/BP-Hand-Writing-master/tc/t10k-images.idx3-ubyte", "rb");
	image_label = fopen("E:/BP-Hand-Writing-master/tc/t10k-labels.idx1-ubyte", "rb");
	if (image_train == NULL || image_label == NULL) {
		cout << "can't open the file!" << endl;
		exit(0);
	}

	//create host buffer and kernel buffer
	unsigned char image_buf[784];
	unsigned char label_buf[10];

	/*data_input = clCreateBuffer(context, CL_MEM_READ_WRITE, TOTAL * 784 * sizeof(int), NULL, &ret);
	target_input = clCreateBuffer(context, CL_MEM_READ_WRITE, TOTAL * 10 * sizeof(int), NULL, &ret);*/

	//throw away useless data
	int useless1[1000];
	fread(useless1, 1, 16, image_train);
	fread(useless1, 1, 8, image_label);

	int cnt = 0;
	int num = 0;
	char *input_iptr = input;
	cout << "read test file..." << endl;
	//60000 times
	while (!feof(image_train) && !feof(image_label)) {
		memset(image_buf, 0, 784);
		memset(label_buf, 0, 10);
		fread(image_buf, 1, 784, image_train);
		fread(label_buf, 1, 1, image_label);

		//initialize the input by 28 x 28 (0,1)matrix of the images
		for (int i = 0; i < 784; i++) {
			if ((unsigned int)image_buf[i] < 128) {
				input_iptr[num * 784 + i] = 0;
			}
			else {
				input_iptr[num * 784 + i] = 1;
			}
		}

		//initialize the target output
		int target_value = (unsigned int)label_buf[0];
		for (int k = 0; k < third; k++) {
			target[num * 10 + k] = 0;
		}
		target[num * 10 + target_value] = 1;

		num++;

		//if (num % 100 == 0) {
		//	printf("testing %d\n", num);

		//	//printf("target = %d\n", target[num]);
		//}
		//printf( "input = %d\n", input[num+784*50]);
		if (num == TOTAL)
		{
			break;
		}
	}

}

void Test(float *weight1_test, float *weight2_test, float *b1_test, float *b2_test) {
	//测试训练模型
	//define parameter
	cl_char *input_test;
	cl_char *target_test;
	float output1_test[second];
	float output2_test[third];

	float test_num = 0.0;
	float test_success_count = 0.0;

	input_test = (cl_char*)malloc(TOTAL * 784 * sizeof(cl_char));
	target_test = (cl_char*)malloc(TOTAL * 10 * sizeof(cl_char));
	memset(input_test, 0, TOTAL * 784 * sizeof(cl_char));
	memset(target_test, 0, TOTAL * 10 * sizeof(cl_char));

	FILE *image_test;
	FILE *image_test_label;
	image_test = fopen("E:/BP-Hand-Writing-master/tc/t10k-images.idx3-ubyte", "rb");
	image_test_label = fopen("E:/BP-Hand-Writing-master/tc/t10k-labels.idx1-ubyte", "rb");
	if (image_test == NULL || image_test_label == NULL) {
		cout << "can't open the file!" << endl;
		exit(0);
	}

	unsigned char image_buf[784];
	unsigned char label_buf[10];

	int useless[1000];
	fread(useless, 1, 16, image_test);
	fread(useless, 1, 8, image_test_label);
	while (!feof(image_test) && !feof(image_test_label)) {
		memset(image_buf, 0, 784);
		memset(label_buf, 0, 10);
		fread(image_buf, 1, 784, image_test);
		fread(label_buf, 1, 1, image_test_label);

		//initialize the input by 28 x 28 (0,1)matrix of the images
		for (int i = 0; i < 784; i++) {
			if ((unsigned int)image_buf[i] < 128) {
				input_test[i] = 0;
			}
			else {
				input_test[i] = 1;
			}
		}

		//initialize the target output
		for (int k = 0; k < third; k++) {
			target_test[k] = 0;
		}
		int target_value = (unsigned int)label_buf[0];
		target_test[target_value] = 1;

		//get parameter from kernel buffer

		//get the ouput and compare with the targe

		for (int j = 0; j < 100; j++) {
			float sigma = 0;
			for (int i = 0; i < 784; i++) {
				sigma += input_test[i] * weight1_test[i * 100 + j];
			}
			float x = sigma + b1_test[j];
			output1_test[j] = f_(x);
		}

		for (int k = 0; k < 10; k++) {
			float sigma = 0;
			for (int j = 0; j < 100; j++) {
				sigma += output1_test[j] * weight2_test[j * 10 + k];
			}
			float x = sigma + b2_test[k];
			output2_test[k] = f_(x);
		}

		float max_value = -99999;
		int max_index = 0;
		for (int k = 0; k < third; k++) {
			if (output2_test[k] > max_value) {
				max_value = output2_test[k];
				max_index = k;
			}
		}

		//output == target
		if (target_test[max_index] == 1) {
			test_success_count++;
		}

		test_num++;

		if ((int)test_num % 1000 == 0) {
			cout << "test num: " << test_num << "  success: " << test_success_count << endl;
		}
	}
	cout << endl;
	cout << "The success rate: " << test_success_count / test_num << endl;


}

void op1_() {
	for (int j = 0; j < second; j++)
	{
		float sigma = 0;
		for (int i = 0; i < first; i++) {
			sigma += input[i] * weight1[i][j];
		}
		float x = sigma + b1[j];
		output1[j] = f_(x);
	}
}

void op2_() {
	for (int k = 0; k < third; k++) {
		float sigma = 0;
		for (int j = 0; j < second; j++) {
			sigma += output1[j] * weight2[j][k];
		}
		float x = sigma + b2[k];
		output2[k] = f_(x);
	}
}

void dt2_() {
	for (int k = 0; k < third; k++) {
		delta2[k] = (output2[k]) * (1.0 - output2[k]) * (output2[k] - target[k]);
	}
}

void dt1_() {
	for (int j = 0; j < second; j++) {
		float sigma = 0;
		for (int k = 0; k < third; k++) {
			sigma += weight2[j][k] * delta2[k];
		}
		delta1[j] = (output1[j]) * (1.0 - output1[j]) * sigma;
	}
}

void feedback_second() {
	for (int j = 0; j < second; j++) {
		b1[j] = b1[j] - alpha * delta1[j];
		for (int i = 0; i < first; i++) {
			weight1[i][j] = weight1[i][j] - alpha * input[i] * delta1[j];
		}
	}
}

void feedback_third() {
	for (int k = 0; k < third; k++) {
		b2[k] = b2[k] - alpha * delta2[k];
		for (int j = 0; j < second; j++) {
			weight2[j][k] = weight2[j][k] - alpha * output1[j] * delta2[k];
		}
	}
}

void training() {
	FILE *image_train;
	FILE *image_label;
	image_train = fopen("E:/BP-Hand-Writing-master/tc/train-images.idx3-ubyte", "rb");
	image_label = fopen("E:/BP-Hand-Writing-master/tc/train-labels.idx1-ubyte", "rb");
	if (image_train == NULL || image_label == NULL) {
		cout << "can't open the file!" << endl;
		exit(0);
	}

	unsigned char image_buf[784];
	unsigned char label_buf[10];

	int useless[1000];
	fread(useless, 1, 16, image_train);
	fread(useless, 1, 8, image_label);

	int cnt = 0;
	cout << "Start training..." << endl;
	//60000 times
	while (!feof(image_train) && !feof(image_label)) {
		memset(image_buf, 0, 784);
		memset(label_buf, 0, 10);
		fread(image_buf, 1, 784, image_train);
		fread(label_buf, 1, 1, image_label);

		//initialize the input by 28 x 28 (0,1)matrix of the images
		for (int i = 0; i < 784; i++) {
			if ((unsigned int)image_buf[i] < 128) {
				input[i] = 0;
			}
			else {
				input[i] = 1;
			}
		}

		//initialize the target output
		int target_value = (unsigned int)label_buf[0];
		for (int k = 0; k < third; k++) {
			target[k] = 0;
		}
		target[target_value] = 1;

		//get the output and start training
		op1_();
		op2_();
		dt2_();
		dt1_();
		feedback_second();
		feedback_third();

		cnt++;
		if (cnt % 1000 == 0) {
			cout << "training image: " << cnt << endl;
		}
	}
	cout << endl;
}



//主函数
int main(int argc, char **argv)
{
	Options options(argc, argv);

	if (!init_opencl()) {
		return -1;
	}

	initialize();

	training();

	//申请内存，初始化
	input_train = (char*)malloc(TOTAL * 784 * sizeof(char));
	output1_train = (float*)malloc(TOTAL * 100 * sizeof(float));
	output2_train = (float*)malloc(TOTAL * 10 * sizeof(float));
	weight1_traint = (float*)malloc(784 * 100 * sizeof(float));
	weight2_traint = (float*)malloc(100 * 10 * sizeof(float));
	b1_train = (float*)malloc(100 * sizeof(float));
	b2_train = (float*)malloc(10 * sizeof(float));
	target_train = (char*)malloc(TOTAL * 10 * sizeof(char));

	memset(input_train, 0, TOTAL * 784 * sizeof(char));
	memset(output2_train, 0, TOTAL * 10 * sizeof(float));
	memset(weight1_traint, 0, 784 * 100 * sizeof(float));
	memset(weight2_traint, 0, 100 * 10 * sizeof(float));
	memset(b1_train, 0, 100 * sizeof(float));
	memset(b2_train, 0, 10 * sizeof(float));
	memset(target_train, 0, TOTAL * 10 * sizeof(char));

	for (int i = 0; i < 784; i++) {
		for (int j = 0; j <100; j++) {
			weight1_traint[j * 784 + i] = weight1[i][j];
		}
	}
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 10; j++) {
			weight2_traint[j * 100 + i] = weight2[i][j];
		}
	}
	for (int i = 0; i < 100; i++) {
		b1_train[i] = b1[i];
	}
	for (int i = 0; i < 10; i++) {
		b2_train[i] = b2[i];
	}

	ReadFile(input_train, target_train);
	float start = getCurrentTimestamp();
	run();
	float end = getCurrentTimestamp();
	double total = (end - start);
	printf("total: %0.3f ms\r\n", total * 1e3);

	for (int i = 0; i < TOTAL; i++) {
		float max_value = -99999;
		int max_index = 0;
		for (int k = 0; k < third; k++) {
			if (output2_train[i * 10 + k] > max_value) {
				max_value = output2_train[i * 10 + k];
				max_index = k;
			}
		}
		//output == target
		if (target_train[i * 10 + max_index] == 1) {
			test_success_count++;
		}

		test_num++;

		if ((int)test_num % 1000 == 0) {
			cout << "test num: " << test_num << "  success: " << test_success_count << endl;
		}

	}
	cout << "The success rate: " << test_success_count / test_num << endl;
	cleanup();
	system("pause");
	return 0;
}// Add you host code
 // Add you host code