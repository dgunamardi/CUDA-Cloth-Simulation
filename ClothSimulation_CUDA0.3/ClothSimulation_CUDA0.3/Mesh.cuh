#pragma once

#include "Common.cuh"
#include <vector>
#include <iterator>

using namespace std;
using namespace stdext;

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif 

#ifndef MESH_CUH
#define MESH_CUH

class Edge {
public:
	int faceCount;
	unsigned int m_faceIndices[2];
	unsigned int m_vertexIndices[2];	// wrong order
	bool hasBending;
	Edge() {
		faceCount = 0;
		hasBending = false;
	}
};
class Face {
public:
	unsigned int m_edgeIndices[3];
	unsigned int m_vertexIndices[3];	// accurate order

	CUDA_CALLABLE Face() {}
	CUDA_CALLABLE Face(const Face& other) {
		for (int i = 0; i < 3; i++) {
			m_edgeIndices[i] = other.m_edgeIndices[i];
			m_vertexIndices[i] = other.m_vertexIndices[i];
		}
	}
	CUDA_CALLABLE Face& operator =(const Face &other) {
		for (int i = 0; i < 3; i++) {
			m_edgeIndices[i] = other.m_edgeIndices[i];
			m_vertexIndices[i] = other.m_vertexIndices[i];
		}
		return *this;
	}
	CUDA_CALLABLE bool hasVertex(unsigned int vIndex) {
		for (int i = 0; i < 3; i++) {
			if (vIndex == m_vertexIndices[i]) {
				return true;
			}
		}
		return false;
	}
	CUDA_CALLABLE unsigned int *otherVertex(unsigned int vIndex) {
		unsigned int*other = new unsigned int[2];
		for (int i = 0, j = 0; i < 3; i++) {
			if (m_vertexIndices[i] != vIndex) {
				other[j++] = m_vertexIndices[i];
			}
		}
		return other;
	}
	CUDA_CALLABLE unsigned int *getVertexIndices() { return m_vertexIndices; }
};

class VertexFaces {
public:
	unsigned int m_numFaces;
	unsigned int *m_fIndices;
	VertexFaces() { m_fIndices = 0; m_numFaces = 0; }
	VertexFaces(VertexFaces const &other) { *this = other; }
	VertexFaces& operator= (VertexFaces const& other) {
		m_numFaces = other.m_numFaces;
		m_fIndices = new unsigned int[m_numFaces];
		copy(other.m_fIndices, other.m_fIndices + m_numFaces, unchecked_array_iterator<unsigned int*>(m_fIndices));
		return *this;
	}
	~VertexFaces() { delete[] m_fIndices; }
};
class VertexEdges {
public:
	unsigned int m_numEdges;
	unsigned int *m_eIndices;
	VertexEdges() { m_eIndices = 0; m_numEdges = 0; }
	VertexEdges(VertexEdges const &other) { *this = other; }
	VertexEdges& operator= (VertexEdges const& other) {
		m_numEdges = other.m_numEdges;
		m_eIndices = new unsigned int[m_numEdges];
		copy(other.m_eIndices, other.m_eIndices + m_numEdges, unchecked_array_iterator<unsigned int*>(m_eIndices));
		return *this;
	}
	~VertexEdges() { delete[] m_eIndices; }
};


// ----- CONTAINS ALL VISUAL DATA -----

class Mesh {
private:
	int m_numVertices;

	vector<unsigned int> m_fIndices;
	vector<unsigned int> m_eIndices;

	vector<int> m_eFaceCount;
	vector<unsigned int> m_eFaceIndices;

	vector<Edge> m_edges;
	vector<Face> m_faces;

	vector<VertexFaces> m_verticesFaces;
	vector<VertexEdges> m_verticesEdges;

	const unsigned int m_verticesPerFace = 3;

public:
	Mesh() {}
	void initMesh(const unsigned int nVertices, const unsigned int nEdges, const unsigned int nFaces, unsigned int* indices) {
		m_numVertices = nVertices;
		m_fIndices.reserve(nFaces* m_verticesPerFace);
		m_eIndices.reserve(nEdges * 2);
		m_eFaceCount.reserve(nEdges);
		m_eFaceIndices.reserve(nEdges * 2);
		m_edges.reserve(nEdges);
		m_faces.reserve(nFaces);
		for (unsigned int i = 0; i < nFaces; i++) {
			addFace(&indices[3 * i]);
		}
		buildNeighbors();
	}

	void addFace(const unsigned int* indices) {
		for (unsigned int i = 0u; i < m_verticesPerFace; i++) {
			m_fIndices.push_back(indices[i]);
		}
	}

	void buildNeighbors() {
		vector<unsigned int> *pEdges = new vector<unsigned int>[numVertices()];
		vector<unsigned int> *vFaces = new vector<unsigned int>[numVertices()];
		vector<unsigned int> *vEdges = new vector<unsigned int>[numVertices()];

		m_edges.clear();
		m_faces.resize(numFaces());

		unsigned int *edges = new unsigned int[m_verticesPerFace * 2];
		for (unsigned int i = 0; i < numFaces(); i++) {

			for (unsigned int j = 0u; j < m_verticesPerFace; j++)
				m_faces[i].m_vertexIndices[j] = m_fIndices[m_verticesPerFace*i + j];

			for (unsigned int j = 0u; j < m_verticesPerFace - 1u; j++) {
				edges[2 * j] = m_faces[i].m_vertexIndices[j];
				edges[2 * j + 1] = m_faces[i].m_vertexIndices[j + 1];
			}
			edges[2 * (m_verticesPerFace - 1)] = m_faces[i].m_vertexIndices[m_verticesPerFace - 1];
			edges[2 * (m_verticesPerFace - 1) + 1] = m_faces[i].m_vertexIndices[0];

			for (unsigned int j = 0u; j < m_verticesPerFace; j++) {
				// add vertex-face connection
				const unsigned int vIndex = m_faces[i].m_vertexIndices[j];
				bool found = false;
				for (unsigned int k = 0; k < vFaces[vIndex].size(); k++) {
					if (vFaces[vIndex][k] == i) {
						found = true; break;
					}
				}
				if (!found) {
					vFaces[vIndex].push_back(i);
				}

				// add edge information
				const unsigned int a = edges[j * 2 + 0];
				const unsigned int b = edges[j * 2 + 1];
				unsigned int edge = 0xffffffff;
				// find edge
				for (unsigned int k = 0; k < pEdges[a].size(); k++) {
					const Edge& e = m_edges[pEdges[a][k]];
					if (((e.m_vertexIndices[0] == a) || (e.m_vertexIndices[0] == b)) && ((e.m_vertexIndices[1] == a) || (e.m_vertexIndices[1] == b))) {
						edge = pEdges[a][k];
						break;
					}
				}
				if (edge == 0xffffffff) {
					// create new
					Edge e;
					e.m_vertexIndices[0] = a;
					e.m_vertexIndices[1] = b;

					e.m_faceIndices[0] = i;
					e.faceCount++;
					e.m_faceIndices[1] = 0xffffffff;

					m_edges.push_back(e);
					m_eIndices.push_back(a);
					m_eIndices.push_back(b);

					m_eFaceIndices.push_back(i);
					m_eFaceCount.push_back(1);
					m_eFaceIndices.push_back(0xffffffff);

					edge = (unsigned int)m_edges.size() - 1u;

					// add vertex-edge connection				
					vEdges[a].push_back(edge);
					vEdges[b].push_back(edge);
				}
				else {
					Edge& e = m_edges[edge];
					e.m_faceIndices[1] = i;
					e.faceCount++;

					m_eFaceIndices[edge * 2 + 1] = i;
					m_eFaceCount[edge]++;
				}
				// append to points
				pEdges[a].push_back(edge);
				pEdges[b].push_back(edge);
				// append face
				m_faces[i].m_edgeIndices[j] = edge;
			}
		}
		delete[] edges;

		// build vertex-face structure
		m_verticesFaces.clear(); // to delete old pointers
		m_verticesFaces.resize(numVertices());
		m_verticesEdges.clear(); // to delete old pointers
		m_verticesEdges.resize(numVertices());
		for (unsigned int i = 0; i < numVertices(); i++) {
			m_verticesFaces[i].m_numFaces = (unsigned int)vFaces[i].size();
			m_verticesFaces[i].m_fIndices = new unsigned int[m_verticesFaces[i].m_numFaces];
			memcpy(m_verticesFaces[i].m_fIndices, vFaces[i].data(), sizeof(unsigned int)*m_verticesFaces[i].m_numFaces);

			m_verticesEdges[i].m_numEdges = (unsigned int)vEdges[i].size();
			m_verticesEdges[i].m_eIndices = new unsigned int[m_verticesEdges[i].m_numEdges];
			memcpy(m_verticesEdges[i].m_eIndices, vEdges[i].data(), sizeof(unsigned int)*m_verticesEdges[i].m_numEdges);
		}

		// check for boundary
		bool isClosed = true;
		for (unsigned int i = 0; i < (unsigned int)m_edges.size(); i++) {
			Edge& e = m_edges[i];
			if (e.m_faceIndices[1] == 0xffffffff) {
				isClosed = false;
				break;
			}
		}

		delete[] pEdges;
		delete[] vFaces;
		delete[] vEdges;
	}

#pragma region PAM
	vector<unsigned int>& getFaceIndices() { return m_fIndices; }
	vector<unsigned int>& getEdgeIndices() { return m_eIndices; }
	vector<int>& getEdgeFaceCount() { return m_eFaceCount; }
	vector<unsigned int>& getEdgeFaceIndices() { return m_eFaceIndices; }

	unsigned int *getIndicesofFace(unsigned int faceIndex) {
		unsigned int *indices = new unsigned int[m_verticesPerFace];
		for (int i = 0; i < m_verticesPerFace; i++) {
			indices[i] = m_faces[faceIndex].m_vertexIndices[i];
		}
		return indices;
	}
	vector<Face>& getFaces() { return m_faces; }
	Face getFace(unsigned int index) { return m_faces[index]; }
	const vector<Edge>& getEdges() const { return m_edges; }
	Edge getEdge(unsigned int index) { return m_edges[index]; }

	unsigned int numVertices() { return m_numVertices; }
	unsigned int numEdges() { return numFaces() * 2; }
	unsigned int numFaces() { return m_fIndices.size() / m_verticesPerFace; }
	unsigned int numIndices() { return m_fIndices.size(); }
	unsigned int verticesPerFace() { return m_verticesPerFace; }


#pragma endregion PAM
};
#endif //! MESH_CUH