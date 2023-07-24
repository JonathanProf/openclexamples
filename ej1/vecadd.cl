__kernel void vecadd( __global int *A, __global int *B, __global int *C)
{                                          			
    	//Get the index of the work-item
    	int inx = get_global_id(0);
		C[inx] = A[inx] + B[inx];
}                                          
