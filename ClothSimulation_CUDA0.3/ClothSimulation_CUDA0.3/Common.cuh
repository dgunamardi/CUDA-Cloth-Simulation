#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include<cstdio>
#include"math_functions.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif 

#ifndef COMMON_CUH
#define COMMON_CUH

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) { 
	unsigned long long int* address_as_ull = (unsigned long long int*)address; 
	unsigned long long int old = *address_as_ull, assumed; 
	do { 
		assumed = old; 
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); 
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
	} while (assumed != old); 
	return __longlong_as_double(old); 
} 

#endif

__device__ __constant__ double kernel_PI = 3.14159265;
const double PI = 3.14159265;

double toRad(double deg) {
	return deg * (PI / 180.0);
}
double toDeg(double rad) {
	return rad * (180.0 / PI);
}
class Matrix {
public:
	double *m_data;
	int m_rows, m_cols;

	CUDA_CALLABLE Matrix(int r = 1, int c = 1, bool id = false) {
		initMatrix(r, c, id);
	}
	CUDA_CALLABLE Matrix(const Matrix &other) {
		m_rows = other.m_rows;
		m_cols = other.m_cols;

		m_data = new double[m_rows * m_cols];
		for (int i = 0; i < m_rows * m_cols; i++) {
			m_data[i] = other.m_data[i];
		}
	}
	CUDA_CALLABLE ~Matrix() {
		m_rows = 0; m_cols = 0;
		delete[] m_data;
	}
	CUDA_CALLABLE void initMatrix(int r, int c, bool id = false) {
		if (m_rows > 0 && m_cols > 0) {
			delete[] m_data;
		}

		m_rows = r; m_cols = c;
		m_data = new double[m_rows * m_cols];
		if (m_rows == m_cols && id)
			loadIdentity();
		else
			fill(0);
	}
	CUDA_CALLABLE Matrix operator *(const Matrix &other) {
		if (m_cols == other.m_rows) {
			Matrix result(m_rows, other.m_cols);
			for (int i = 0; i < m_rows; i++) {
				for (int j = 0; j < other.m_cols; j++) {
					float sum = 0;
					for (int k = 0; k < other.m_rows; k++) {
						sum = sum + (m_data[i * m_cols + k] * other.m_data[k * other.m_cols + j]);
					}
					result.m_data[i * result.m_cols + j] = sum;
				}
			}
			return result;
		}
		else {
			printf("invalid matrix size");
			return Matrix(1, 1);
		}
	}
	CUDA_CALLABLE Matrix operator +(const Matrix &other) {
		Matrix result(m_rows, m_cols);
		for (int i = 0; i < m_rows * m_cols; i++) {
			result.m_data[i] = m_data[i] + other.m_data[i];
		}
		return result;
	}
	CUDA_CALLABLE Matrix operator -(const Matrix &other) {
		Matrix result(m_rows, m_cols);
		for (int i = 0; i < m_rows * m_cols; i++) {
			result.m_data[i] = m_data[i] - other.m_data[i];
		}
		return result;
	}
	CUDA_CALLABLE Matrix operator *(double val) {
		Matrix result(m_rows, m_cols);
		for (int i = 0; i < m_rows * m_cols; i++) {
			result.m_data[i] = m_data[i] * val;
		}
		return result;
	}
	CUDA_CALLABLE Matrix operator /(double val) {
		Matrix result(m_rows, m_cols);
		for (int i = 0; i < m_rows * m_cols; i++) {
			result.m_data[i] = m_data[i] / val;
		}
		return result;
	}
	CUDA_CALLABLE Matrix & operator =(const Matrix &other) {
		if (m_rows > 0 && m_cols > 0) {
			delete[] m_data;
		}

		m_rows = other.m_rows;
		m_cols = other.m_cols;
		m_data = new double[m_rows * m_cols];
		for (int i = 0; i < m_rows * m_cols; i++) {
			m_data[i] = other.m_data[i];
		}

		return *this;
	}
	CUDA_CALLABLE void loadIdentity() {
		if (m_rows == m_cols) {
			for (int i = 0; i < m_rows; i++) {
				for (int j = 0; j < m_cols; j++) {
					m_data[i * m_cols + j] = i == j ? 1 : 0;
				}
			}
		}
	}
	CUDA_CALLABLE void fill(double val) {
		for (int i = 0; i < m_rows * m_cols; i++) {
			m_data[i] = val;
		}
	}
	CUDA_CALLABLE void clearData() {
		delete[] m_data;
		m_cols = m_rows = 0;
	}
	CUDA_CALLABLE void display() {
		for (int i = 0; i < m_rows; i++) {
			for (int j = 0; j < m_cols; j++) {
				printf("%lf ", m_data[i * m_cols + j]);
			}
			printf("\n");
		}
	}
	CUDA_CALLABLE void shape() {
		printf("%d %d\n", m_rows, m_cols);
	}
	CUDA_CALLABLE Matrix transpose() {
		Matrix res(m_cols, m_rows);
		for (int i = 0; i < res.m_rows; i++) {
			for (int j = 0; j < res.m_cols; j++) {
				res.m_data[i * res.m_cols + j] = m_data[j * m_cols + i];
			}
		}
		return res;
	}
	CUDA_CALLABLE Matrix getDiagonal() {
		Matrix res(m_rows, m_cols);
		if (m_rows == m_cols) {
			for (int i = 0; i < m_rows; i++) {
				res.m_data[i * m_cols + i] = m_data[i * m_cols + i];
			}
		}
		else {
			printf("invalid matrix size");
		}
		return res;
	}
	CUDA_CALLABLE Matrix getDiagonalInv() {
		Matrix res(m_rows, m_cols);
		if (m_rows == m_cols) {
			for (int i = 0; i < m_rows; i++) {
				res.m_data[i * m_cols + i] = 1.0 / m_data[i * m_cols + i];
			}
		}
		else {
			printf("invalid matrix size");
		}
		return res;
	}
	CUDA_CALLABLE double getValue() {
		if (m_rows == m_cols == 1) {
			return m_data[0];
		}
		else {
			printf("not 1");
			return 0;
		}
	}

#pragma region Linear Equation Solver
	// Join the matrix first!
	Matrix gaussElimination() {
		int n = m_rows;

		for (int i = 0; i < n; i++) {
			// Search for maximum in this column
			double maxEl = fabs(data(i, i));
			int maxRow = i;
			for (int k = i + 1; k<n; k++) {
				if (fabs(data(k, i)) > maxEl) {
					maxEl = fabs(data(k, i));
					maxRow = k;
				}
			}

			// Swap maximum row with current row (column by column)
			for (int k = i; k<n + 1; k++) {
				double tmp = data(maxRow, k);
				set(maxRow, k, data(i, k));
				set(i, k, tmp);
			}

			// Make all rows below this one 0 in current column
			for (int k = i + 1; k<n; k++) {
				double c = -data(k, i) / data(i, i);
				for (int j = i; j < n + 1; j++) {
					if (i == j) {
						set(k, j, 0);
					}
					else {
						insert(k, j, c * data(i, j));
					}
				}
			}
		}

		// Solve equation Ax=b for an upper triangular matrix A
		Matrix x(n, 1);
		for (int i = n - 1; i >= 0; i--) {
			x.set(i, 0, data(i, n) / data(i, i));
			for (int k = i - 1; k >= 0; k--) {
				double res = (data(k, i) * data(i, 0));
				x.insert(k, n, -res);
			}
		}
		return x;
	}
	Matrix append(Matrix &b, int c = 0) {
		Matrix fin(m_rows, m_cols + b.m_cols);
		if (c != 0) {
			printf("%d %d\n", fin.m_rows, fin.m_cols);
		}
		for (int i = 0; i < m_rows; i++) {
			for (int j = 0; j < m_cols; j++) {
				fin.set(i, j, data(i, j));
			}
		}
		for (int i = 0; i < m_rows; i++) {
			fin.set(i, fin.m_cols - 1, b.data(i, 0));
		}
		return fin;
	}
#pragma endregion // Linear Equation Solver (Seidel and Gaussian)

#pragma region Inverse (UNDER CONSIDERATION)
	CUDA_CALLABLE Matrix inverse() {
		Matrix res(m_cols, m_rows);

		double det = 1.0 / (data(0, 0) * data(1, 1) - data(0, 1) * data(1, 0));
		res.set(0, 0, data(1, 1));
		res.set(1, 1, data(0, 0));
		res.set(0, 1, data(0, 1) * -1.0);
		res.set(1, 0, data(1, 0) * -1.0);
		res = res * det;
		return res;
	}
	bool LU_Decomposition(Matrix &L, Matrix &U) {
		unsigned int n = m_rows;
		if (m_rows != m_cols) {
			return false;
		}

		L.initMatrix(n, n, true); U.initMatrix(n, n, true);

		for (int i = 0; i < n; i++) {
			L.set(i, 0, data(i, 0));
		}
		for (int j = 1; j < n; j++) {
			U.set(0, j, data(0, j) / L.data(0, 0));
		}

		for (int j = 1; j < n - 1; j++) {
			for (int i = j; i < n; i++) {
				double sum = 0;
				for (int k = 0; k < j - 1; k++) {
					sum = sum + L.data(i, k) * U.data(k, j);
				}
				L.set(i, j, data(i, j) - sum);
			}
			for (int k = j + 1; k < n; k++) {
				double sum = 0;
				for (int i = 0; i < j - 1; i++) {
					sum = sum + L.data(j, i) * U.data(i, k);
				}
				U.set(j, k, (data(j, k) - sum) / L.data(j, j));
			}
		}

		double sum = 0;
		for (int k = 0; k < n - 1; k++) {
			sum = sum + L.data(n - 1, k) * U.data(k, n - 1);
		}
		L.set(n - 1, n - 1, data(n - 1, n - 1) - sum);

		return true;
	}
#pragma endregion // Inverse (UNDER CONSIDERATION)

	CUDA_CALLABLE bool isDiagonal() {
		if (data(0, 1) == 0 && data(1, 0) == 0) {
			return true;
		}
		return false;
	}

#pragma region SquareRoot and Polar Decomposition
	CUDA_CALLABLE Matrix sqrtSimple() {
		Matrix res(m_cols, m_rows);
		for (int i = 0; i < m_rows; i++) {
			for (int j = 0; j < m_cols; j++) {
				if (i == j) {
					res.set(i, j, sqrt(data(i, j)));
				}
			}
		}
		for (int i = 0; i < m_rows; i++) {
			res.set(i, i, sqrt(data(i, i)));
		}
		return res;
	}
	CUDA_CALLABLE Matrix FindR() {
		Matrix Fs = (*this).transpose() * (*this);
		Matrix R(m_cols, m_rows);
		//Fs.display();

		Matrix U;
		if (Fs.isDiagonal()) {
			U = Fs.sqrtSimple();
		}
		else {
			double rad = atan((2.0 * Fs.data(0, 1)) / (Fs.data(0, 0) - Fs.data(1, 1))) / 2.0;
			//printf("%lf\n", toDeg(rad));

			//printf("%lf\n\n", cos(rad));
			Matrix rot(m_cols, m_rows);

			rot.set(0, 0, cos(rad)); rot.set(0, 1, sin(rad));
			rot.set(1, 0, -sin(rad)); rot.set(1, 1, cos(rad));


			U = rot * Fs * rot.transpose();
			U = U.sqrtSimple();

			Matrix rrot(m_cols, m_rows);

			rrot.set(0, 0, cos(-rad)); rrot.set(0, 1, sin(-rad));
			rrot.set(1, 0, -sin(-rad)); rrot.set(1, 1, cos(-rad));

			U = rrot * U * rrot.transpose();
			//U.display();
		}
		R = (*this) * U.inverse();

		//R.display();

		return R;

	}
#pragma endregion //SquareRoot

	CUDA_CALLABLE bool hasZeroDiagonal() {
		for (int i = 0; i < m_rows; i++) {
			if (fabs(m_data[i * m_cols + i]) < 0.000000f) {
				return true;
			}
		}
		return false;
	}
	CUDA_CALLABLE double data(unsigned int i, unsigned int j) {
		return m_data[i*m_cols + j];
	}
	CUDA_CALLABLE void insert(unsigned int i, unsigned int j, double val) {
		m_data[i*m_cols + j] = m_data[i*m_cols + j] + val;
	}
	CUDA_CALLABLE void set(unsigned int i, unsigned int j, double val) {
		m_data[i*m_cols + j] = val;
	}
	CUDA_CALLABLE void getRaw(double *data, int &rows, int &cols) {
		rows = m_rows;
		cols = m_cols;
		data = new double[rows * cols];
		for (int i = 0; i < rows * cols; i++) {
			data[i] = m_data[i];
		}
	}
	CUDA_CALLABLE double *getRawData(double *data) {
		return m_data; 
	}
	CUDA_CALLABLE void setRaw(double *data, int rows, int cols) {
		if (m_cols > 0 && m_rows > 0) {
			delete[] m_data;
		}
		m_rows = rows; m_cols = cols;
		m_data = new double[m_rows * m_cols];
		for (int i = 0; i < m_rows * m_cols; i++) {
			m_data[i] = data[i];
		}
	}
	CUDA_CALLABLE void setRaw(double *data, int offset, int rows, int cols) {
		if (m_cols > 0 && m_rows > 0) {
			delete[] m_data;
		}
		m_rows = rows; m_cols = cols;
		m_data = new double[m_rows * m_cols];
		for (int i = 0; i < m_rows * m_cols; i++) {
			m_data[i] = data[offset + i];
		}
	}
};

class Vector3d {
private:
	double m_x, m_y, m_z, m_w;
public:
	CUDA_CALLABLE Vector3d(double x = 0, double y = 0, double z = 0, double w = 1) {
		m_x = x; m_y = y; m_z = z; m_w = w;
	}
	CUDA_CALLABLE Vector3d(const Vector3d &other) {
		m_x = other.m_x;
		m_y = other.m_y;
		m_z = other.m_z;
		m_w = other.m_w;
	}
	CUDA_CALLABLE void set(double x, double y, double z, double w = 1) {
		m_x = x; m_y = y; m_x = z;
	}
	CUDA_CALLABLE Vector3d operator = (const Vector3d &other) {
		m_x = other.m_x; m_y = other.m_y; m_z = other.m_z;
		return *this;
	}
	CUDA_CALLABLE Vector3d operator - (const Vector3d &other) {
		return Vector3d(m_x - other.m_x, m_y - other.m_y, m_z - other.m_z);
	}
	CUDA_CALLABLE Vector3d operator * (const Vector3d &other) {	//cross
		return Vector3d(m_y * other.m_z - other.m_y * m_z, m_z * other.m_x - other.m_z * m_x, m_x * other.m_y - other.m_x * m_y);
	}
	CUDA_CALLABLE Vector3d normalize() {
		double div = magnitude();
		return Vector3d(m_x / div, m_y / div, m_z / div);
	}

	CUDA_CALLABLE Vector3d operator * (float value) {
		return Vector3d(m_x * value, m_y * value, m_z * value);
	}
	CUDA_CALLABLE Vector3d operator / (float value) {
		value > 0 ? value = value : value = 1;
		return Vector3d(m_x / value, m_y / value, m_z / value);
	}
	CUDA_CALLABLE Vector3d operator + (Vector3d other) {
		return Vector3d(m_x + other.m_x, m_y + other.m_y, m_z + other.m_z);
	}
	CUDA_CALLABLE double operator ^ (Vector3d other) {	//dot
		return (m_x * other.m_x + m_y * other.m_y + m_z * other.m_z);
	}
	CUDA_CALLABLE double magnitude() { return sqrt(m_x * m_x + m_y * m_y + m_z * m_z); }
	CUDA_CALLABLE Vector3d operator -() const {
		Vector3d v;
		v.m_x = -m_x;
		v.m_y = -m_y;
		v.m_z = -m_z;
		return v;
	}

	CUDA_CALLABLE Matrix toMatrix() {
		Matrix res(3, 1);
		res.set(0, 0, m_x);
		res.set(1, 0, m_y);
		res.set(2, 0, m_z);
		return res;
	}
	CUDA_CALLABLE void display() {
		printf("%lf %lf %lf \n", m_x, m_y, m_z);
	}
#pragma region PAM
	void x(double val) { m_x = val; }
	void y(double val) { m_y = val; }
	void z(double val) { m_z = val; }
	void w(double val) { m_w = val; }
	double x() { return m_x; }
	double y() { return m_y; }
	double z() { return m_z; }
	double w() { return m_w; }
#pragma endregion PAM

};

CUDA_CALLABLE bool isRadObtuse(double rad) {
	if (rad * 180 / kernel_PI > 90.0005) {
		return true;
	}
	return false;
}

CUDA_CALLABLE double angle(Vector3d a, Vector3d b) {
	return acos(a.normalize() ^ b.normalize());
}


#endif