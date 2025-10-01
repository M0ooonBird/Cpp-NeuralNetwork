#pragma once
#include <iostream>
#include <vector>
#include "FloatType.h"
#include "Vector.h"

template <typename T>
class MatrixT
{
	using iterator = T*;
	using const_iterator = const T*;
public:
	MatrixT() : _row(0), _col(0) {
		_data = nullptr;
	}

	MatrixT(int n) : MatrixT(n, n, T()) {
	}

	MatrixT(int row, int col) : MatrixT(row, col, T()) {
	}

	MatrixT(int row, int col, T val) : _row(row), _col(col)
	{
		_data = new T[row * col];

		for (int i = 0; i < row * col; i++)
			_data[i] = val;
	}
	MatrixT(int row, int col, T* data) : _row(row), _col(col)
	{
		_data = data;
		_isOwner = false; // 不是数据的拥有者
	}

	MatrixT(const MatrixT<T>& mat) : _row(mat._row), _col(mat._col)
	{
		_data = new T[_row * _col];

		for (int i = 0; i < _row * _col; i++)
			_data[i] = mat._data[i];
	}

	MatrixT(MatrixT<T>&& mat) noexcept
		: _row(mat._row), _col(mat._col)
	{
		_data = mat._data;
		mat._data = nullptr;
		mat._row = 0;
		mat._col = 0;
	}

	~MatrixT() {
		if (!_isOwner)
		{
			_data = nullptr;
			_row = 0;
			_col = 0;
			return;
		}
		if (_data != nullptr)
		{
			delete[] _data;
			_data = nullptr;
			_row = 0;
			_col = 0;
		}
	}

	const T* data() const { return _data; }
	T* data() { return _data; }

	void Clear()
	{
		if (_data != nullptr) {
			delete[] _data;
			_data = nullptr;
		}
		_row = 0;
		_col = 0;
	}

	void Resize(int n)
	{
		this->Clear();
		_row = n;
		_col = n;
		_data = new T[n * n];

		for (int i = 0; i < n * n; i++)
			_data[i] = T();
	}
	void Resize(int n, int m)
	{
		this->Clear();
		_row = n;
		_col = m;
		_data = new T[n * m];

		for (int i = 0; i < n * m; i++)
			_data[i] = T();
	}

	int RowSize() { return _row; }
	int ColSize() { return _col; }

	void Print()
	{
		if (RowSize() * ColSize() == 0)
		{
			std::cout << "empty " << std::endl;
			return;
		}
		for (int i = 0; i < RowSize(); i++)
		{
			for (int j = 0; j < ColSize(); j++)
			{
				std::cout << _data[j + i * _col] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	// 迭代器
	iterator begin() noexcept {
		return _data;
	}
	const_iterator begin() const noexcept {
		return _data;
	}
	iterator end() noexcept {
		return _data + _row * _col;
	}
	const_iterator end() const noexcept {
		return _data + _row * _col;
	}

	T& operator () (int i, int j)
	{
		return _data[j + i * _col];
	}
	const T& operator () (int i, int j) const
	{
		return _data[j + i * _col];
	}

	MatrixT<T>& operator = (const MatrixT<T>& mat)
	{
		if (this == &mat)
		{
			return *this;
		}
		if (this->_row != mat._row || this->_col != mat._col)
		{
			this->Resize(mat._row, mat._col);
		}
		memcpy(_data, mat._data, _col * _row * sizeof(T));

		return *this;
	}

	MatrixT<T>& operator = (MatrixT<T>&& mat) noexcept
	{
		if (this != &mat)
		{
			if (_row * _col > 0)
				delete[] _data;

			_row = mat._row;
			_col = mat._col;
			_data = mat._data;

			mat._data = nullptr;
			mat._row = 0;
			mat._col = 0;
		}
		return *this;
	}

	MatrixT<T>& operator += (const MatrixT<T>& mat)
	{
		if (_row != mat._row || _col != mat._col) {
			throw std::invalid_argument("Matrix size mismatch for += operation");
		}
	#pragma omp parallel for num_threads(16)
		for (int i = 0; i < _col * _row; i++)
		{
			_data[i] += mat._data[i];
		}
		return *this;
	}
	MatrixT<T>& operator -= (const MatrixT<T>& mat)
	{
		if (_row != mat._row || _col != mat._col) {
			throw std::invalid_argument("Matrix size mismatch for -= operation");
		}
	#pragma omp parallel for num_threads(16)
		for (int i = 0; i < _col * _row; i++)
		{
			_data[i] -= mat._data[i];
		}
		return *this;
	}
	MatrixT<T>& operator *= (const T& val)
	{
		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < _col * _row; i++)
		{
			_data[i] *= val;
		}
		return *this;
	}

	friend MatrixT<T> operator + (const MatrixT<T>& mat1, const MatrixT<T>& mat2)
	{
		if (mat1._row != mat2._row || mat1._col != mat2._col)
		{
			throw std::invalid_argument("Matrix size mismatch for addition");
		}
		MatrixT<T> Msum(mat1._row, mat1._col);
		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < Msum._row * Msum._col; i++)
		{
			Msum._data[i] = mat1._data[i] + mat2._data[i];
		}
		return Msum;
	}
	friend MatrixT<T> operator - (const MatrixT<T>& mat1, const MatrixT<T>& mat2)
	{
		if (mat1._row != mat2._row || mat1._col != mat2._col)
		{
			throw std::invalid_argument("Matrix size mismatch for addition");
		}
		MatrixT<T> Msum(mat1._row, mat1._col);

		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < Msum._row * Msum._col; i++)
		{
			Msum._data[i] = mat1._data[i] - mat2._data[i];
		}
		return Msum;
	}

	friend MatrixT<T> operator * (const T& val, const MatrixT<T>& mat)
	{
		MatrixT<T> Msum(mat._row, mat._col);
		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < Msum._row * Msum._col; i++)
		{
			Msum._data[i] = mat._data[i] * val;
		}
		return Msum;
	}
	friend MatrixT<T> operator * (const MatrixT<T>& mat, const T& val)
	{
		MatrixT<T> Msum(mat._row, mat._col);
		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < Msum._row * Msum._col; i++)
		{
			Msum._data[i] = mat._data[i] * val;
		}
		return Msum;
	}

	friend VectorT<T> operator * (const MatrixT<T>& mat, const VectorT<T>& vec)
	{
		if (vec.Size() != static_cast<size_t>(mat._col)) 
		{
			throw std::invalid_argument("Vector size mismatch with matrix columns");
		}
		VectorT<T> V(mat._row);

		for (int i = 0; i < mat._row; i++)
		{
			V[i] = 0;
			for (int j = 0; j < mat._col; j++)
			{
				V[i] += mat._data[j + i * mat._col] * vec[j];
			}
		}
		return V;
	}

	friend VectorT<T> operator * (const VectorT<T>& vec, const MatrixT<T>& mat)
	{
		if (vec.Size() != static_cast<size_t>(mat._row))
		{
			throw std::invalid_argument("Vector size mismatch with matrix rows");
		}
		VectorT<T> V(mat._col);

		for (int i = 0; i < mat._col; i++)
		{
			for (int j = 0; j < mat._row; j++)
			{
				V[i] += mat._data[i + j * mat._col] * vec[j];
			}
		}
		return V;
	}

	friend MatrixT<T> operator * (const MatrixT<T>& mat1, const MatrixT<T>& mat2)
	{
		if (mat2._row != mat1._col)
		{
			throw std::invalid_argument("Matrix size mismatchs");
		}
		MatrixT<T> MP(mat1._row, mat2._col);

		for (int i = 0; i < MP._row; i++)
		{
			for (int j = 0; j < MP._col; j++)
			{
				for (int k = 0; k < mat1._col; k++)
				{
					MP._data[i * MP._col + j] += mat1._data[i * mat1._col + k] * mat2._data[k * mat2._col + j];
				}
			}
		}
		return MP;
	}
	friend MatrixT<T> operator / (const MatrixT<T>& mat1, const MatrixT<T>& mat2)
	{
		if (mat1._row != mat2._row || mat1._col != mat2._col)
		{
			throw std::invalid_argument("Matrix size mismatch for addition");
		}
		MatrixT<T> Msum(mat1._row, mat1._col);

		for (int i = 0; i < Msum._row * Msum._col; i++)
		{
			Msum._data[i] = mat1._data[i] / mat2._data[i];
		}
		return Msum;
	}
	friend MatrixT<T> operator / (const MatrixT<T>& mat, const T& val)
	{
		MatrixT<T> Msum(mat._row, mat._col);

		for (int i = 0; i < Msum._row * Msum._col; i++)
		{
			Msum._data[i] = mat._data[i] / val;
		}
		return Msum;
	}

	MatrixT<T> GetTransPose()
	{
		MatrixT<T> MT(_col, _row);
		for (int i = 0; i < MT._row; i++)
		{
			for (int j = 0; j < MT._col; j++)
			{
				MT(i,j) = _data[i + j * _col];
			}
		}
		return MT;
	}

	int GetRank()
	{
		int n = _row;
		int m = _col;
		if (n * m == 0)
		{
			return 0;
		}
		MatrixT<T> temp = *this;
		double EPS = 1.0E-9;

		int rank = 0;
		std::vector<bool> row_selected(n, false);
		for (int i = 0; i < m; ++i) {
			int j;
			for (j = 0; j < n; ++j) {
				if (!row_selected[j] && abs(temp(j, i)) > EPS)
					break;
			}

			if (j != n) {
				++rank;
				row_selected[j] = true;
				for (int p = i + 1; p < m; ++p)
				{
					temp(j, p) = temp(j, p) / temp(j, i);
				}

				for (int k = 0; k < n; ++k) {
					if (k != j && abs(temp(k, i)) > EPS) {
						for (int p = i + 1; p < m; ++p)
							temp(k, p) -= temp(j, p) * temp(k, i);
					}
				}
			}
		}
		return rank;
	}


private:
	int _row;//
	int _col;//
	bool _isOwner = true;
	T* _data = nullptr;
};

typedef MatrixT<unsigned char> iMat;
typedef MatrixT<scalar> Matrix;

