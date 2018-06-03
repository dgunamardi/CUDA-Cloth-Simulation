
#include<cstdio>

// CUDA INCLUDES
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

// Libraries
#include "Common.cuh"
#include "Data.cuh"
#include "Mesh.cuh"
#include "Bending.cuh"
#include "Stiffness.cuh"

#include<chrono>

using namespace std;

#pragma region Global Funcs

__global__ void kernel_generateVertices(Vector3d *d_vertices, unsigned int rows, unsigned int cols, double dy, double dx) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < rows * cols) {
		int i = idx / cols;
		int j = idx % cols;
		double y = i * dy;
		double x = j * dx;
		d_vertices[i * cols + j] = Vector3d(x, -y, -100);
	}
	
}
__global__ void kernel_generateIndices(unsigned int *d_indices, unsigned int rows, unsigned int cols) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < (rows - 1) * (cols - 1)) {
		int i = idx / (cols - 1);
		int j = idx % (cols - 1);
		int helper = i % 2 == j % 2 ? 1 : 0;
		int index = 6 * (i * (cols - 1)) + 6 * j;
		d_indices[index] = i* cols + j;
		d_indices[index + 1] = i*cols + j + 1;
		d_indices[index + 2] = (i + 1)*cols + j + helper;
		index += 3;
		d_indices[index] = (i + 1)*cols + j + 1;
		d_indices[index + 1] = (i + 1)*cols + j;
		d_indices[index + 2] = i*cols + j + 1 - helper;
		index += 3;
	}


}

#pragma endregion

class Model {
private:
	unsigned int m_width, m_height;
	unsigned int m_rows, m_cols;

	Data m_data;	// particle data
	Mesh m_mesh;
	QuadraticBending m_bendingModel;
	CorotatedStiffness m_stiffnessModel;

	double m_timeStep; // simulation time step
	Vector3d m_gravity; // external forces

	vector<double> m_S;
	double m_eps;
	int iterCount; // step counter (for debugging)

	Matrix M, Q, fext;

public:
	Model(unsigned int width = 50, unsigned int height = 50, unsigned int rows = 10, unsigned int cols = 10) {
		
		iterCount = 0;
		m_timeStep = 0.1; //in seconds
		m_gravity.set(0, -9.80f, 0);
		m_eps = 0.5;
	}

	void initModel(unsigned int width = 50, unsigned int height = 50, unsigned int rows = 10, unsigned int cols = 10) {
		m_width = width; m_height = height; m_rows = rows; m_cols = cols;

		size_t bytes;
		int blockDim = 128;
		int gridDim;
		// LINEAR FASTER - CHECKED UP TO 50x50

		
#pragma region GENERATE VERTICES

		const double dy = m_width / (double)(m_cols - 1);
		const double dx = m_height / (double)(m_rows - 1);
		Vector3d *vertices = new Vector3d[m_rows * m_cols];
		Vector3d *d_vertices;

		gridDim = ((m_rows * m_cols) / blockDim) + 1;

		bytes = m_rows * m_cols * sizeof(Vector3d);
		cudaMalloc((void**)&d_vertices, bytes);

		cudaMemcpy(d_vertices, vertices, bytes, cudaMemcpyHostToDevice);
		kernel_generateVertices << < gridDim, blockDim >> > (d_vertices, m_rows, m_cols, dy, dx);
		cudaMemcpy(vertices, d_vertices, bytes, cudaMemcpyDeviceToHost);
	
		cudaFree(d_vertices);
		//for (int i = 0; i < m_rows; i++) {
		//	for (int j = 0; j < m_cols; j++) {
		//		vertices[i*m_cols + j].display();
		//	}
		//}

#pragma endregion
#pragma region GENERATE INDICES
		const int nIndices = 6 * (m_cols - 1) * (m_rows - 1);
		unsigned int *indices = new unsigned int[nIndices];
		unsigned int *d_indices;

		gridDim = (((m_rows-1) * (m_cols-1)) / blockDim) + 1;

		bytes = nIndices * sizeof(unsigned int);
		cudaMalloc((void**)&d_indices, bytes);

		cudaMemcpy(d_indices, indices, bytes, cudaMemcpyHostToDevice);
		kernel_generateIndices << <gridDim, blockDim >> > (d_indices, m_rows, m_cols);
		cudaMemcpy(indices, d_indices, bytes, cudaMemcpyDeviceToHost);

		cudaFree(d_indices);

		//for (int i = 0; i < nIndices; i+=3) {
		//	printf("%d %d %d\n", indices[i], indices[i + 1], indices[i + 2]);
		//}

#pragma endregion

#pragma region INITIALIZE DATAS
		m_data.initData(m_rows * m_cols, nIndices / 3, vertices);
		m_mesh.initMesh(m_rows * m_cols, nIndices * 2 / 3, nIndices / 3, indices);

		m_bendingModel.initData(m_rows * m_cols, m_mesh.verticesPerFace());
		m_stiffnessModel.initData(m_mesh.numFaces());

#pragma endregion

#pragma region PRECOMPUTATION
		
		m_data.computeBaseNormalAndAreas(m_mesh.numFaces(), indices);
		m_data.computeMass(m_mesh.numFaces(), indices);
		
		m_stiffnessModel.computeLinearStiffness(m_mesh.numFaces(), m_mesh, m_data);
		m_bendingModel.computeBending(m_mesh.getFaces(), m_mesh.getEdges(), m_data.getVertex0(), m_data.getArea());
		
#pragma endregion

#pragma region Constraints setup

		m_S.resize(m_mesh.numVertices());
		for (int i = 0; i < m_mesh.numVertices(); i++) {
			m_S[i] = m_data.isFixed(i) ? 0.0 : 1.0;
		}

		//m_data.setMass(0, (std::numeric_limits<double>::max)());
		//m_data.setMass(9, (std::numeric_limits<double>::max)());

#pragma endregion

		fext = assembleExternalForces();
		Q = m_bendingModel.getBending();
		M = m_data.assembleMass();

		iterCount = 0;

		delete[] vertices;
		delete[] indices;
		
	}

	void updateParticles(Matrix v_delta, Matrix x_delta) {
		m_data.updateVelocity(v_delta);
		m_data.updatePosition(x_delta);
	}

	void computeForces() {
		m_data.computeNormal(m_mesh.numFaces(), &m_mesh.getFaceIndices()[0]);

		m_stiffnessModel.computeCorotatedStiffness(m_mesh.numFaces(), m_mesh, m_data);
		m_stiffnessModel.assembleAll(m_mesh.numVertices(), m_mesh.verticesPerFace(), m_mesh.getFaces());

		Matrix K = m_stiffnessModel.getStiffness();
		K = K + Q;
		Matrix D = m_stiffnessModel.getDamping();

		Matrix X = m_data.assembleVertex();
		Matrix v = m_data.assembleVelocity();

		Matrix f0 = m_stiffnessModel.getInternalForce();

		Matrix A = M + D + (K * pow(m_timeStep, 2.0));
		//if (iterCount % 10 == 0) {
		//	A.display();
		//	printf("\n");
		//}

		Matrix B = ((K * X) + (f0 - fext) + (K * v * m_timeStep) + (D * v)) * -m_timeStep;
		//B.display();

		// Gaussian
		//Matrix fin = A.append(B);
		////fin.display();
		//Matrix v_delta = fin.gaussElimination();

		// PCG
		Matrix v_delta = solver_PCG(A, B);
		//v_delta.display();                                                                                                                                        

		v = v + v_delta;
		Matrix x_delta = v * m_timeStep;

		// Update Velocity and Position
		updateParticles(v_delta, x_delta);

		// Final Equation
		// Ax = B, solved using gaussian elim
		// (M + D + dt^2 K)dv = -dt(Kx + f0-fext + dtKv + Dv)
		// where x = dv
		iterCount++;
	}

	Matrix assembleExternalForces() {
		Matrix res(m_mesh.numVertices() * m_mesh.verticesPerFace(), 1);
		Matrix g = m_gravity.toMatrix();
		for (int i = 0; i < m_mesh.numVertices(); i++) {
			for (int j = 0; j < m_mesh.verticesPerFace(); j++) {
				if (m_data.isFixed(i)) {
					res.set(i * 3 + j, 0, 0);
				}
				else {
					res.set(i * 3 + j, 0, g.data(j, 0));
				}
			}
		}
		return res;
	}

#pragma region Linear System Solver using PCG

	Matrix filter(Matrix a) {
		Matrix res(a.m_rows, a.m_cols);
		for (int i = 0; i < a.m_rows / 3; i++) {
			for (int j = 0; j < 3; j++) {
				res.set(i * 3 + j, 0, a.data(i * 3 + j, 0) * m_S[i]);
			}
		}
		return res;
	}
	Matrix solver_PCG(Matrix A, Matrix b) {
		printf("%d: ", iterCount);
		Matrix P_inv = A.getDiagonal();
		Matrix P = A.getDiagonalInv();
		unsigned int nVertices = m_mesh.numVertices();

		Matrix v_delta(b.m_rows, b.m_cols);
		double g0 = (filter(b).transpose() * P * filter(b)).getValue();

		Matrix r = filter(b - A * v_delta); // difference
		Matrix c = filter(P_inv * r); // cg times dif
		double gNew = (r.transpose() * c).getValue();

		int count = 0;
		while (gNew > pow(m_eps, 2) * g0) {
			Matrix q = filter(A*c);
			double alpha = gNew / (c.transpose() * q).getValue();
			v_delta = v_delta + c * alpha;
			r = r - q * alpha;
			Matrix s = P_inv * r;
			double gOld = gNew;
			gNew = (r.transpose() * s).getValue();
			c = filter(s + (c * (gNew / gOld)));

			count++;
		}
		printf("%d\n", count);

		return v_delta;
	}
#pragma endregion Solver

};

