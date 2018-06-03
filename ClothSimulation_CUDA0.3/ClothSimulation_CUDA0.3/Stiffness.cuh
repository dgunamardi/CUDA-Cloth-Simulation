#pragma once

#include "Common.cuh"
#include<chrono>

using namespace std;

#ifndef STIFFNESS_H
#define STIFFNESS_H

class CorotatedStiffness {
private:

	// Young Moduli | Poission Ratios
	double m_weft, m_warp, m_shear;	// Ex, Ey, Es
	double m_Vxy; double m_Vyx;

	double m_damping_mass, m_damping_stiffness; // damping coeff

	Matrix m_C; // Tensor

	vector<Matrix> m_Ke;
	vector<Matrix> m_KeR;
	vector<Matrix> m_fe0;
	vector<Matrix> m_De;
	Matrix m_K;
	Matrix m_f0;
	Matrix m_D;

public:
	CorotatedStiffness() {
		m_weft = m_warp = m_shear = 500.0;
		m_Vxy = m_Vyx = 0.33;
		m_damping_mass = 0.1;
		m_damping_stiffness = 0.1;
	}

	void initData(unsigned int numFaces) {
		m_Ke.reserve(numFaces);
		m_KeR.reserve(numFaces);
		m_fe0.reserve(numFaces);
		m_De.reserve(numFaces);
		Matrix p0(9, 6);
		Matrix p1(9, 9);
		Matrix p2(9, 1);
		for (int i = 0; i < numFaces; i++) {
			m_Ke.push_back(p0);
			m_KeR.push_back(p1);
			m_fe0.push_back(p2);
			m_De.push_back(p1);
		}
		computeElasticityTensor();
	}

	// Calculate C
	void computeElasticityTensor() {
		m_C.initMatrix(3, 3);
		double poi = 1.0 - (m_Vxy * m_Vyx);

		m_C.set(0, 0, m_weft / poi);
		m_C.set(0, 1, m_weft * m_Vyx / poi);
		m_C.set(1, 0, m_warp * m_Vxy / poi);
		m_C.set(1, 1, m_warp / poi);
		m_C.set(2, 2, m_shear);

		//m_C.display();
	}

	// Calculate Corotated Stiffness and damping
	void computeCorotatedStiffness(unsigned int nFaces, Mesh &mesh, Data &data) {
		Matrix projectionMatrix_x;

		/*chrono::high_resolution_clock Clock;
		auto t1 = Clock.now();*/

		for (int i = 0; i < nFaces; i++) {
			unsigned int *indices = mesh.getIndicesofFace(i);
			Matrix *x0p = data.getProjected0(indices, i);
			Matrix *xp = data.getProjected(indices, i, projectionMatrix_x);

			computeLocalCorotatedStiffness(i, xp, x0p, projectionMatrix_x);
			computeDamping(i, data.getMass(indices));
			delete[] indices;
			delete[] x0p;
			delete[] xp;
		}
		//m_KeR[0].display();

		/*auto t2 = Clock.now();
		double t = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();*/

		//printf("time = %lf \n", t * 1e-9);


		//m_K.display();
		//m_f0.display();
		//m_D.display();
	}

	// Linear Stiffness Ke
	void computeLinearStiffness(unsigned int numFaces, Mesh &mesh, Data &data) {
		for (int i = 0; i < numFaces; i++) {
			unsigned int *indices = mesh.getIndicesofFace(i);
			Matrix *x0p = data.getProjected0(indices, i);

			computeLocalLinearStiffness(i, x0p, data.getArea(i));

			delete[] indices;
			delete[] x0p;
		}
	}
	void computeLocalLinearStiffness(unsigned int faceIndex, Matrix* x0p, double faceArea) {
		// Shape Function -> Ni/m
		Matrix Nm(3, 2);
		int a = 1, b = 2;
		double area = 2.0 * faceArea;
		for (int j = 0; j < 3; j++) {
			Nm.set(j, 0, ((x0p[a].data(1, 0) - x0p[b].data(1, 0)) / area));
			Nm.set(j, 1, ((x0p[b].data(0, 0) - x0p[a].data(0, 0)) / area));
			a = a + 1 > 2 ? 0 : a + 1;
			b = b + 1 > 2 ? 0 : b + 1;
		}

		//Nm.display();

		// Stress -> B
		Matrix B(3, 6);

		for (int j = 0; j < 3; j++) {
			B.set(0, j * 2, Nm.data(j, 0));
			B.set(1, (j * 2) + 1, Nm.data(j, 1));
			B.set(2, j * 2, Nm.data(j, 1) / 2.0);
			B.set(2, (j * 2) + 1, Nm.data(j, 0) / 2.0);

		}

		// Linear Stiffness | K = A * Bt * C * B
		m_Ke[faceIndex] = B.transpose() * m_C * B * faceArea;

	}

	// Corotated Stiffness KeR
	void computeLocalCorotatedStiffness(unsigned int faceIndex, Matrix* xp, Matrix* x0p, Matrix p) {
		// Get Deformation Matrix F
		Matrix S(2, 2), T(2, 2);
		Matrix s1 = x0p[1] - x0p[0];
		Matrix s2 = x0p[2] - x0p[0];
		S.set(0, 0, s1.data(0, 0));
		S.set(1, 0, s1.data(1, 0));
		S.set(0, 1, s2.data(0, 0));
		S.set(1, 1, s2.data(1, 0));

		Matrix t1 = xp[1] - xp[0];
		Matrix t2 = xp[2] - xp[0];
		T.set(0, 0, t1.data(0, 0));
		T.set(1, 0, t1.data(1, 0));
		T.set(0, 1, t2.data(0, 0));
		T.set(1, 1, t2.data(1, 0));

		Matrix F = T * S.inverse();

		// Polar Decomposition
		Matrix R = F.FindR();

		// Find P^T R
		R = p.transpose() * R;

		// Rp,e Matrix
		Matrix Rp(9, 6);
		for (int j = 0; j < 3; j++) {
			unsigned int r = j * 3, c = j * 2;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 2; l++) {
					Rp.set(r + k, c + l, R.data(k, l));
				}
			}
		}

		// Corotated Stiffness -> KeR
		m_KeR[faceIndex] = Rp * m_Ke[faceIndex] * Rp.transpose();

		Matrix x0p_asmb(6, 1);
		for (int i = 0; i < 3; i++) {
			x0p_asmb.set(i * 2, 0, x0p[i].data(0, 0));
			x0p_asmb.set(i * 2 + 1, 0, x0p[i].data(1, 0));
		}

		// Initial Force (of undeformed points) -> fe0
		Matrix KeR_bar = Rp * m_Ke[faceIndex];
		m_fe0[faceIndex] = KeR_bar * x0p_asmb * -1.0;
	}

	void computeDamping(unsigned int faceIndex, Matrix mass) {
		m_De[faceIndex] = (mass * m_damping_mass) + (m_KeR[faceIndex] * m_damping_stiffness);
	}

	void assembleAll(unsigned int nVertices, unsigned int vpf, vector<Face> &faces) {
		assembleStiffness(nVertices, vpf, faces);
		assembleInternalForces(nVertices, vpf, faces);
		assembleDamping(nVertices, vpf, faces);
	}

#pragma region Display

	void displayK(unsigned int p1, unsigned int p2) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				printf("%lf ", m_K.data(p1 * 3 + i, p2 * 3 + j));
			}printf("\n");
		}
	}

	void displayD(unsigned int p1, unsigned int p2) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				printf("%lf ", m_D.data(p1 * 3 + i, p2 * 3 + j));
			}printf("\n");
		}
	}

#pragma endregion Display

#pragma region Assembly

	Matrix assembleStiffness(unsigned int nVertices, unsigned int vpf, vector<Face> &faces) {
		unsigned int size = nVertices * vpf;
		m_K.clearData();
		m_K = Matrix(size, size, false);
		// loop of all elemen stiffnesses
		for (int i = 0; i < m_Ke.size(); i++) {
			// loop for each particle
			unsigned int *vIndex = faces[i].m_vertexIndices;
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					// fill matrix
					for (int l = 0; l < 3; l++) {
						for (int m = 0; m < 3; m++) {
							//printf("%d %d\n", vIndex[j] * 3 + l, vIndex[k] * 3 + m);
							double t = m_KeR[i].data(j * 3 + l, k * 3 + m);
							m_K.insert(vIndex[j] * 3 + l, vIndex[k] * 3 + m, t);
						}
					}
				}
			}
		}
		//m_K.display();
		return m_K;
	}

	Matrix assembleInternalForces(unsigned int nVertices, unsigned int vpf, vector<Face> &faces) {
		m_f0.clearData();
		m_f0 = Matrix(nVertices * vpf, 1);
		for (int i = 0; i < m_fe0.size(); i++) {
			unsigned int* index = faces[i].m_vertexIndices;
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					m_f0.insert(index[j] * 3 + k, 0, m_fe0[i].data(j * 3 + k, 0));
				}
			}
		}
		return m_f0;
	}

	Matrix assembleDamping(unsigned int nVertices, unsigned int vpf, vector<Face> &faces) {
		unsigned int size = nVertices * vpf;
		m_D.clearData();
		m_D = Matrix(size, size, false);
		for (int i = 0; i < m_Ke.size(); i++) {
			unsigned int *vIndex = faces[i].m_vertexIndices;
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						for (int m = 0; m < 3; m++) {
							double t = m_De[i].data(j * 3 + l, k * 3 + m);
							m_D.insert(vIndex[j] * 3 + l, vIndex[k] * 3 + m, t);
						}
					}
				}
			}
		}

		return m_D;
	}

#pragma endregion Assembly

#pragma region PAM
	Matrix getStiffness() {
		return m_K;
	}
	Matrix getInternalForce() {
		return m_f0;
	}
	Matrix getDamping() { return m_D; }

#pragma endregion PAM

};

#endif // !STIFFNESS_H