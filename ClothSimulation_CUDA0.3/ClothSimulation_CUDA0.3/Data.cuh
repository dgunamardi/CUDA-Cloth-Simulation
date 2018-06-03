#pragma once

// CUDA INCLUDES
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

// Libraries
#include "Common.cuh"
#include "Mesh.cuh"

#include<cstdio>
#include<vector>
#include<algorithm>

using namespace std;

#ifndef DATA_CUH
#define DATA_CUH

#pragma region KERNEL AREA

__global__ void kernel_computeNormal(unsigned int numFaces, unsigned int *d_vertexIndices, Vector3d *d_X, Vector3d *d_normal) {

	int idx = (blockIdx.x * blockDim.x)+ threadIdx.x;
	if (idx < numFaces) {
		unsigned int a, b, c;
		a = d_vertexIndices[idx * 3];
		b = d_vertexIndices[idx * 3 + 1];
		c = d_vertexIndices[idx * 3 + 2];
		Vector3d x0 = d_X[a];
		Vector3d x1 = d_X[b];
		Vector3d x2 = d_X[c];
		Vector3d res = (x0 - x1).normalize() * (x2-x1).normalize();
		d_normal[idx] = res.normalize();
	}
}

__global__ void kernel_computeNormalandArea(unsigned int numFaces, unsigned int *d_vertexIndices, Vector3d *d_X, Vector3d *d_normal, double *d_area) {

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < numFaces) {
		unsigned int a, b, c;
		a = d_vertexIndices[idx * 3];
		b = d_vertexIndices[idx * 3 + 1];
		c = d_vertexIndices[idx * 3 + 2];
		Vector3d x0 = d_X[a];
		Vector3d x1 = d_X[b];
		Vector3d x2 = d_X[c];
		
		Vector3d s1 = x0 - x1;
		Vector3d s2 = x2 - x1;
		Vector3d s3 = x2 - x0;

		d_normal[idx] = (s1.normalize() * s2.normalize()).normalize();

		double l1 = s1.magnitude(), l2 = s2.magnitude(), l3 = s3.magnitude();
		double s = (l1 + l2 + l3) / 2;
		d_area[idx] = sqrt(s * (s - l1) * (s - l2) * (s - l3));
	}
}

__global__ void kernel_computeMass(unsigned int numFaces, unsigned int *d_vertexIndices, Vector3d *d_X, double *d_area, double *d_mass, double d_density) {
	// atomicadd double on d_mass

	// where gidx = face index
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numFaces) {

		double *angles = new double[3];
		unsigned int *index = new unsigned int[3];

		bool isObtuse = false;

		Vector3d *X = new Vector3d[3];
		for (int i = 0; i < 3; i++) {
			X[i] = d_X[d_vertexIndices[idx * 3 + i]];
			index[i] = d_vertexIndices[idx * 3 + i];
		}

		unsigned int *ia = new unsigned int[3];
		unsigned int *ib = new unsigned int[3];
		Vector3d *ea = new Vector3d[3];
		Vector3d *eb = new Vector3d[3];

		for (int i = 0; i < 3; i++) {
			ia[i] = i - 1 < 0 ? 3 - 1 : i - 1;
			ib[i] = i + 1 > 3 - 1 ? 0 : i + 1;
			ea[i] = X[ia[i]] - X[i];
			eb[i] = X[ib[i]] - X[i];
			angles[i] = angle(ea[i], eb[i]);

			if (isRadObtuse(angles[i])) {
				isObtuse = true;
			}
		}

		if (!isObtuse) {	// VORONOI
			double voronoiArea = 0;
			for (int i = 0; i < 3; i++) {
				voronoiArea = (
					pow(ea[i].magnitude(), 2.0) / tan(angles[ib[i]]) +
					pow(eb[i].magnitude(), 2.0) / tan(angles[ia[i]])
					) / 8.0;
				//if (d_vertexIndices[idx * 3 + i] == 0) {
				//	printf("%d %lf %lf %d\n", idx, voronoiArea, d_mass[idx * 3 + i], d_vertexIndices[idx * 3 + i]);
				//}W
				atomicAdd(&(d_mass[index[i]]), voronoiArea * d_density);

			}
		}
		else {	// HYBRID
			double area = d_area[idx];
			double mass = 0;
			for (int i = 0; i < 3; i++) {
				mass = isRadObtuse(angles[i]) ? area / 2.0 : area / 4.0;
				atomicAdd(&d_mass[index[i]], (mass * d_density));
			}
		}

		
		delete[] angles;
		delete[] index;
		delete[] X;
		delete[] ia; delete[] ib;
		delete[] ea; delete[] eb;
	}
}

__global__ void kernel_updateData(unsigned int numVertices, double *d_in, double *d_out) {

}

#pragma endregion


class Data {
private:
	vector<Vector3d> m_X;	// current vertices
	vector<Vector3d> m_X0;	// model vertices (base posistons)
	vector<Vector3d> m_velocity;
	vector<bool> m_fixed;

	vector<Vector3d> m_fn;	// current face normals
	vector<Vector3d> m_fn0;	// model face normals
	vector<double> m_mass;	// masses
	vector<double> m_area;	// initial areas

	const double m_density = 0.1;	// mass density

	size_t m_xBytes;
	size_t m_inBytes;
	size_t m_fBytes;
	size_t m_mBytes;
	size_t m_aBytes;

public:
	Data() {}

	void initData(int nVertices, int nFaces, const Vector3d *vertices) {
		m_X.reserve(nVertices);
		m_X0.reserve(nVertices);
		m_velocity.reserve(nVertices);
		m_fixed.reserve(nVertices);
		m_fn.reserve(nFaces);
		m_fn0.reserve(nFaces);
		m_mass.reserve(nVertices);
		m_area.reserve(nFaces);

		for (int i = 0; i < nVertices; i++) {
			m_X.push_back(vertices[i]);
			m_X0.push_back(vertices[i]);
			m_fixed.push_back(false);
			m_mass.push_back(0);
			m_velocity.push_back(Vector3d(0, 0, 0));
		}

		for (int i = 0; i < nFaces; i++) {
			m_area.push_back(0);
			m_fn.push_back(Vector3d(0, 0, 0));
			m_fn0.push_back(Vector3d(0, 0, 0));
		}
		m_fixed[0] = true; //m_fixed[9] = true;

		m_xBytes = nVertices * sizeof(Vector3d);
		m_inBytes = nFaces * 3 * sizeof(unsigned int);
		m_fBytes = nFaces * sizeof(Vector3d);
		m_mBytes = nVertices * sizeof(double);
		m_aBytes = nFaces * sizeof(double);
	}
	void computeBaseNormalAndAreas(unsigned int numFaces, unsigned int *indices) {
		int blockDim = 128;
		int gridDim = (numFaces / blockDim) + 1;

		Vector3d *d_X, *d_normals;
		unsigned int *d_indices;
		double *d_areas;

		cudaMalloc((void**)&d_X, m_xBytes);
		cudaMalloc((void**)&d_normals, m_fBytes);
		cudaMalloc((void**)&d_indices, m_inBytes);
		cudaMalloc((void**)&d_areas, m_aBytes);

		cudaMemcpy(d_X, &m_X0[0], m_xBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_indices, indices, m_inBytes, cudaMemcpyHostToDevice);

		kernel_computeNormalandArea<<<gridDim, blockDim>>>(numFaces, d_indices, d_X, d_normals, d_areas);

		cudaMemcpy(&m_fn0[0], d_normals, m_fBytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(&m_area[0], d_areas, m_aBytes, cudaMemcpyDeviceToHost);
		
		cudaFree(d_X);
		cudaFree(d_normals);
		cudaFree(d_indices);
		cudaFree(d_areas);
	}
	void computeNormal(unsigned int numFaces, unsigned int *indices) {
		int blockDim = 128;
		int gridDim = (numFaces / blockDim) + 1;

		Vector3d *d_X, *d_normals;
		unsigned int *d_indices;

		cudaMalloc((void**)&d_X, m_xBytes);
		cudaMalloc((void**)&d_normals, m_fBytes);
		cudaMalloc((void**)&d_indices, m_inBytes);

		cudaMemcpy(d_X, &m_X[0], m_xBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_indices, indices, m_inBytes, cudaMemcpyHostToDevice);

		kernel_computeNormal<<<gridDim, blockDim>>>(numFaces, d_indices, d_X, d_normals);

		cudaMemcpy(&m_fn[0], d_normals, m_fBytes, cudaMemcpyDeviceToHost);

		cudaFree(d_X);
		cudaFree(d_normals);
		cudaFree(d_indices);
	}
	void computeMass(unsigned int numFaces, unsigned int *indices) {
		int blockDim = 128;
		int gridDim = (numFaces / blockDim) + 1;

		Vector3d *d_X;
		unsigned int *d_indices;
		double *d_area, *d_mass;	// area = face, mass = vert

		cudaMalloc((void**)&d_X, m_xBytes);
		cudaMalloc((void**)&d_indices, m_inBytes);
		cudaMalloc((void**)&d_area, m_aBytes);
		cudaMalloc((void**)&d_mass, m_mBytes);

		cudaMemcpy(d_X, &m_X0[0], m_xBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_indices, indices, m_inBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_area, &m_area[0], m_aBytes, cudaMemcpyHostToDevice);

		kernel_computeMass << < gridDim, blockDim >> > (numFaces, d_indices, d_X, d_area, d_mass, m_density);

		cudaMemcpy(&m_mass[0], d_mass, m_mBytes, cudaMemcpyDeviceToHost);

		cudaFree(d_X);
		cudaFree(d_indices);
		cudaFree(d_area);
		cudaFree(d_mass);

		printf("%lf %lf %lf\n", m_mass[0], m_mass[m_X0.size() / 2], m_mass[m_X0.size() - 1]);

		//for (int i = 0; i < m_mass.size(); i++) {
		//	printf("%lf\n",m_mass[i]);
		//}

	}

	// Updates Velocity and positions
	void updateVelocity(Matrix &v_delta) {
		for (int i = 0; i < m_velocity.size(); i++) {
			if (m_fixed[i])
				continue;
			Vector3d v;
			v.x(v_delta.data(i * 3, 0));
			v.y(v_delta.data(i * 3 + 1, 0));
			v.z(v_delta.data(i * 3 + 2, 0));
			m_velocity[i] = m_velocity[i] + v;
		}

	}
	void updatePosition(Matrix &x_delta) {
		for (int i = 0; i < m_X.size(); i++) {
			if (m_fixed[i])
				continue;
			Vector3d p;
			p.x(x_delta.data(i * 3, 0));
			p.y(x_delta.data(i * 3 + 1, 0));
			p.z(x_delta.data(i * 3 + 2, 0));
			m_X[i] = m_X[i] + p;
		}
	}


	// Project points to 2D, and grabs the projection matrix
	Matrix *getProjected(const unsigned int indices[3], unsigned int faceIndex, Matrix &projection) {
		Vector3d vx = (m_X[indices[1]] - m_X[indices[0]]);
		Vector3d vpx = (vx / vx.magnitude());
		Vector3d vy = m_fn[faceIndex] * vpx;
		Vector3d vpy = (vy / vy.magnitude());

		projection.initMatrix(2, 3);
		Matrix px = vpx.toMatrix().transpose();
		Matrix py = vpy.toMatrix().transpose();
		for (int i = 0; i < 3; i++) {
			projection.set(0, i, px.data(0, i));
			projection.set(1, i, py.data(0, i));
		}

		Matrix *m = new Matrix[3];
		for (int j = 0; j < 3; j++) {
			m[j] = projection * m_X[indices[j]].toMatrix();
		}
		return m;
	}
	Matrix *getProjected0(const unsigned int indices[3], unsigned int faceIndex) {
		//for (int i = 0; i < 3; i++) {
		//	printf("%d ", indices[i]);
		//	m_X0[indices[i]].display();
		//}printf("\n");

		Vector3d vx = (m_X0[indices[1]] - m_X0[indices[0]]);
		Vector3d vpx = (vx / vx.magnitude());
		Vector3d vy = m_fn0[faceIndex] * vpx;
		Vector3d vpy = (vy / vy.magnitude());

		Matrix projection(2, 3);
		Matrix px = vpx.toMatrix().transpose();
		Matrix py = vpy.toMatrix().transpose();
		for (int i = 0; i < 3; i++) {
			projection.set(0, i, px.data(0, i));
			projection.set(1, i, py.data(0, i));
		}
		Matrix *m = new Matrix[3];
		for (int j = 0; j < 3; j++) {
			m[j] = projection * m_X0[indices[j]].toMatrix();
		}
		return m;
	}

#pragma region Assembly

	Matrix assembleVertex() {
		Matrix m(m_X.size() * 3, 1);
		for (int i = 0; i < m_X.size(); i++) {
			m.set(i * 3, 0, m_X[i].x());
			m.set(i * 3 + 1, 0, m_X[i].y());
			m.set(i * 3 + 2, 0, m_X[i].z());
		}
		return m;
	}
	Matrix assembleVelocity() {
		Matrix m(m_X.size() * 3, 1);
		for (int i = 0; i < m_X.size(); i++) {
			m.set(i * 3, 0, m_velocity[i].x());
			m.set(i * 3 + 1, 0, m_velocity[i].y());
			m.set(i * 3 + 2, 0, m_velocity[i].z());
		}
		return m;
	}
	Matrix assembleMass() {
		unsigned int n = m_X.size();
		Matrix m(n * 3, n * 3);
		for (int i = 0; i < n; i++) {
			m.set(i * 3, i * 3, m_mass[i]);
			m.set(i * 3 + 1, i * 3 + 1, m_mass[i]);
			m.set(i * 3 + 2, i * 3 + 2, m_mass[i]);
		}
		return m;
	}

#pragma endregion Assembly

#pragma region PAM

	vector<Vector3d>& getVertex0() { return m_X0; }
	vector<Vector3d>& getVertex() { return m_X; }
	vector<double>& getArea() { return m_area; }
	double getArea(unsigned int index) { return m_area[index]; }
	vector<double>& getMass() { return m_mass; }
	Matrix getMass(unsigned int *indices) {
		Matrix m(9, 9);
		for (int j = 0; j < 3; j++) {
			m.set(j * 3, j * 3, m_mass[indices[j]]);
			m.set(j * 3 + 1, j * 3 + 1, m_mass[indices[j]]);
			m.set(j * 3 + 2, j * 3 + 2, m_mass[indices[j]]);
		}
		return m;
	}

	vector<Vector3d>& getNormal0() { return m_fn0; }
	vector<Vector3d>& getNormal() { return m_fn; }

	bool isFixed(unsigned int vIndex) { return m_fixed[vIndex]; }
	

#pragma endregion
};

#endif //! DATA_CUH