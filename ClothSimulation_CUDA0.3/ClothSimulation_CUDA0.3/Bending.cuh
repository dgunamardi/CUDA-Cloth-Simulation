#pragma once

#include"Common.cuh"
#include"Mesh.cuh"
#include<cassert>

#ifndef BENDING_H
#define BENDING_H

class QuadraticBending {
private:
	Matrix m_Q;
public:

	QuadraticBending() {}
	void initData(unsigned int nVertices, unsigned int vpf) {
		m_Q.initMatrix(nVertices * vpf, nVertices * vpf, false);
	}
	double cotTheta(Vector3d v, Vector3d w) {
		const double cosTheta = v ^ w;
		const double sinTheta = (v * w).magnitude();
		return (cosTheta / sinTheta);
	}
	void computeBending(const vector<Face> &faces, vector<Edge> edges, vector<Vector3d> vertices, const vector<double> &area) {

		//printf("%d %d %d %d\n", faces.size(), edges.size(), vertices.size(), area.size());
		for (int i = 0; i < edges.size(); i++) {
			Edge e = edges[i];
			if (e.faceCount == 2) {
				unsigned int indices[4];

				indices[0] = e.m_vertexIndices[0];
				indices[1] = e.m_vertexIndices[1];

				unsigned int index;
				for (int j = 0; j < 3; j++) {
					index = faces[e.m_faceIndices[0]].m_vertexIndices[j];
					if (index != indices[0] && index != indices[1]) {
						indices[2] = index;
						break;
					}
				}
				for (int j = 0; j < 3; j++) {
					index = faces[e.m_faceIndices[1]].m_vertexIndices[j];
					if (index != indices[0] && index != indices[1]) {
						indices[3] = index;
						break;
					}
				}

				Matrix Q_local;
				computeLocalBending(vertices[indices[0]], vertices[indices[1]], vertices[indices[2]], vertices[indices[3]], Q_local, area[e.m_faceIndices[0]], area[e.m_faceIndices[1]]);

				for (int j = 0; j < 4; j++) {
					for (int k = 0; k < 4; k++) {
						for (int l = 0; l < 3; l++) {
							m_Q.insert(indices[j] * 3 + l, indices[k] * 3 + l, Q_local.data(j, k));
						}
					}
				}
			}
		}
	}
	void computeLocalBending(Vector3d v0, Vector3d v1, Vector3d v2, Vector3d v3, Matrix &Q_local, double A0, double A1) {
		Matrix k(1, 4);

		Vector3d e0 = v1 - v0;
		Vector3d e1 = v2 - v0;
		Vector3d e2 = v3 - v0;
		Vector3d e3 = v2 - v1;
		Vector3d e4 = v3 - v1;

		double c01 = cotTheta(e0, e1);
		double c02 = cotTheta(e0, e2);
		double c03 = cotTheta(-e0, e3);
		double c04 = cotTheta(-e0, e4);

		k.set(0, 0, c03 + c04);
		k.set(0, 1, c01 + c02);
		k.set(0, 2, -(c01 + c03));
		k.set(0, 3, -(c02 + c04));


		Q_local = k.transpose() * k  *(3.0 / (A0 * A1));
	}
	void displayQ(unsigned int p1, unsigned int p2) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				printf("%lf ", m_Q.data(p1 * 3 + i, p2 * 3 + j));
			}printf("\n");
		}
	}
	Matrix getBending() {
		return m_Q;
	}
};


#endif // !BENDING_H