//#include<cstdio>
//#include<cuda.h>
//#include<cuda_runtime.h>
//#include<device_launch_parameters.h>
//#include<vector>
//#include "Common.cuh"
//#include "Mesh.cuh"
//using namespace std;
//
////#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
////#else
////__device__ double atomicAdd(double* address, double val) { 
////	unsigned long long int* address_as_ull = (unsigned long long int*)address; 
////	unsigned long long int old = *address_as_ull, assumed; 
////	do { 
////		assumed = old; 
////		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); 
////		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
////	} while (assumed != old); 
////	return __longlong_as_double(old); 
////} 
////
////#endif
//
//
//
//__global__ void d_square(Vector3d *d_in, Vector3d *d_out) {
//	int idx = threadIdx.x;
//	Vector3d val = d_in[idx];
//	printf("%d\n", sizeof(d_in[idx]));
//	d_out[idx] = val + val;
//}
//
//
//__global__ void reduce(double *d_data, double *d_out) {
//	int tid = threadIdx.x;
//	printf("%lf, %d, %lf\n", d_out[0], tid, d_data[tid]);
//	atomicAdd(&d_out[0], d_data[tid]);
//}
//
//
//void main() {
//	unsigned int size = 5, bytes = size * sizeof(double);
//	double *test = new double[size];
//	for (int i = 0; i < size; i++) {
//		test[i] = i + 1;
//	}
//	double *res = new double[size];
//	
//	double *d_in, *d_out;
//
//	cudaMalloc((void**)&d_in, bytes);
//	cudaMalloc((void**)&d_out, bytes);
//
//	cudaMemcpy(d_in, test, bytes, cudaMemcpyHostToDevice);
//
//	reduce<<<1,size>>>(d_in, d_out);
//
//	cudaMemcpy(res, d_out, bytes, cudaMemcpyDeviceToHost);
//	printf("%lf\n", res[0]);
//	
//}
//
////// TIMER
/////*chrono::high_resolution_clock Clock;
////auto t1 = Clock.now();
////auto t2 = Clock.now();
////double t = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
////
////printf("time = %lf \n", t * 1e-9);*/
