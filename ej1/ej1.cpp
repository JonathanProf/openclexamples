#define CL_TARGET_OPENCL_VERSION 200
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl.h>
#endif

char* readKernel( const char *filename ){
		
		FILE *fp;
		char *fileData;
		long fileSize;
		
		/* Open the file */
		fp = fopen(filename, "r");
		if( !fp ){
			printf( "Cpuld not open file: %s\n", filename );
			exit(-1);
		}
		
		/* Determine the file size */
		if( fseek(fp, 0, SEEK_END)){
			printf("Error reading the file\n");
			exit(-1);
		}
		fileSize = ftell(fp);
		if(fileSize < 0){
			printf("Error reading the file\n");
			exit(-1);
		}
		if( fseek(fp, 0, SEEK_SET)){
			printf("Error reading the file\n");
			exit(-1);
		}
		
		/* Read the contents */
		fileData = (char*)malloc(fileSize+1);
		if( !fileData ){
			exit(-1);
		}
		if( fread(fileData, fileSize, 1, fp) != 1 ){
			printf("Error reading the file\n");
			exit(-1);
		}
		
		/* Terminate the string */
		fileData[fileSize] = '\0';
		
		/* Close the file */
		if( fclose(fp) ){
			printf("Error closing the file\n");
			exit(-1);
		}
		return fileData;
}

#define CASE_CL_ERROR(NAME) case NAME: return #NAME;

const char* opencl_error_to_str (cl_int error) {
	switch(error) {
		CASE_CL_ERROR(CL_SUCCESS)
		CASE_CL_ERROR(CL_DEVICE_NOT_FOUND)
		CASE_CL_ERROR(CL_DEVICE_NOT_AVAILABLE)
		CASE_CL_ERROR(CL_INVALID_PROGRAM)
		CASE_CL_ERROR(CL_INVALID_DEVICE)
		CASE_CL_ERROR(CL_INVALID_BINARY)
		CASE_CL_ERROR(CL_INVALID_BUILD_OPTIONS)
		CASE_CL_ERROR(CL_INVALID_OPERATION)
		CASE_CL_ERROR(CL_COMPILER_NOT_AVAILABLE)
		CASE_CL_ERROR(CL_BUILD_PROGRAM_FAILURE)
		CASE_CL_ERROR(CL_OUT_OF_RESOURCES)
		CASE_CL_ERROR(CL_OUT_OF_HOST_MEMORY)
	default:
	return "UNKNOWN ERROR CODE";
	}
}

#define CHECK_STATUS(status) \
			if (status != CL_SUCCESS) {\
				fprintf(stderr,\
				"OpenCL error in file %s line %d, error code %s\n",\
				__FILE__,\
				__LINE__,\
				opencl_error_to_str(status));\
				exit(0);\
			}
	
#define VECTOR_SIZE 10


int main(void) {
	
	// This code executes on the OpenCL host
	
	const char *programSource = readKernel("vecadd.cl");
	printf("El programa es:\n%s",programSource);
	
	
	// Elements in each array
	const int elements = 10;
	
	// Compute the size of the data
	size_t datasize = sizeof(int)*elements;
	
	// Allocate space for input/output host data
	int *A = (int*)malloc(datasize);
	int *B = (int*)malloc(datasize);
	int *C = (int*)malloc(datasize);
  
	int i;
	for( i=0 ; i < elements ; i++){
		A[i]  = i;
		B[i]  = i;
	}
	
	cl_int status;

	// Get platform and device information
	cl_platform_id *platforms = NULL;
	cl_uint num_platforms;
	
	status = clGetPlatformIDs(0, NULL,&num_platforms);
	CHECK_STATUS(status);

	platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*num_platforms);
	status = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_STATUS(status);
		

	//Get the devices list and choose the device you want to run on
	cl_device_id     *device_list = NULL;
	cl_uint           num_devices;

	status = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
	CHECK_STATUS(status);
	
	device_list = (cl_device_id *)  malloc(sizeof(cl_device_id)*num_devices);
	
	status = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
	CHECK_STATUS(status);
  
	// Create one OpenCL context for each device in the platform
	cl_context context;
	context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &status);
	CHECK_STATUS(status);
	
	// Create a command queue
	cl_command_queue cmdQueue = clCreateCommandQueueWithProperties(context, device_list[0], 0, &status);
	CHECK_STATUS(status);

	// Create memory buffers on the device for each vector
	cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,		datasize, NULL, &status);
	CHECK_STATUS(status);
	cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,		datasize, NULL, &status);
	CHECK_STATUS(status);
	cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 	datasize, NULL, &status);
	CHECK_STATUS(status);
	
	// Copy the Buffer A and B to the device
	status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_TRUE, 0, datasize, A, 0, NULL, NULL);
	CHECK_STATUS(status);
	status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_TRUE, 0, datasize, B, 0, NULL, NULL);
	CHECK_STATUS(status);
	
	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,(const char **)&programSource, NULL, &status);
	CHECK_STATUS(status);

	// Build the program
	status = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
	CHECK_STATUS(status);


	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "vecadd", &status);
	CHECK_STATUS(status);
	
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufA);
	CHECK_STATUS(status);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufB);
	CHECK_STATUS(status);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufC);
	CHECK_STATUS(status);
	
	// Define an index space of work-items for execution.
	// A work-group size is not required, but can be used.
	size_t indexSpaceSize[1], workGroupSize[1];
	
	// There are 'elements' work-items
	indexSpaceSize[0] = elements;
	workGroupSize[0] = 2;
	
	// Execute the kernel
	status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);
	CHECK_STATUS(status);
	// Read thr device output buffer to the host output array
	status = clEnqueueReadBuffer( cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0 , NULL, NULL);
	CHECK_STATUS(status);
	
	
	for(int i = 0 ; i < elements ; i++)
		printf("C[%d]=%d\n",i,C[i]);
	
	// Free OpenCL resources
	clReleaseKernel( kernel );
	clReleaseProgram( program );
	clReleaseCommandQueue( cmdQueue );
	clReleaseMemObject( bufA );
	clReleaseMemObject( bufB );
	clReleaseMemObject( bufC );
	clReleaseContext( context );
	
	// Free host resources
	free(A);
	free(B);
	free(C);
	
	return 0;
}
