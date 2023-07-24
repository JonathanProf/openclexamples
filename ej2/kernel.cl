__kernel void main_kernel(__global int *a, __global int *b, __global int *c)
{
	int indx = get_global_id(0);
	c[indx] = a[indx]+b[indx];
}
