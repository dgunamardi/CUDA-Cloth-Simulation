#include<cstdio>
#include"Model.cuh"


Model model;

int main(int argc, char *argv[]) {

	chrono::high_resolution_clock Clock;
	auto t1 = Clock.now();

	model.initModel(30, 30, 10, 10);

	auto t2 = Clock.now();
	double t = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();

	printf("time = %lf \n", t * 1e-9);

	for (int i = 0; i < 10; i++) {
		model.computeForces();
	}



	return 0;
}