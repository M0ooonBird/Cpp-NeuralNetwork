#pragma once
#include <iostream>
#include <vector>
#include "FloatType.h"

template <typename T>
class VectorT
{
	using iterator = T*;
	using const_iterator = const T*;
public:
	VectorT() : _len(0){
		_data = nullptr;
	}

	VectorT(int n) : VectorT(n, T()) {
	}

	VectorT(int n, T val) : _len(n)
	{
		_data = new T[n];

		for (int i = 0; i < n; i++)
			_data[i] = val;
	}
	VectorT(T* data, int n) : _len(n)
	{
		_data = data;
		_isOwner = false; // 不是数据的拥有者
	}

	VectorT(const VectorT<T>& vec) : _len(vec._len)
	{
		_data = new T[_len];

		for (int i = 0; i < _len; i++)
			_data[i] = vec._data[i];
	}

	VectorT(VectorT<T>&& vec) noexcept
		: _len(vec._len)
	{
		_data = vec._data;
		vec._data = nullptr;
		vec._len = 0;
	}

	~VectorT() {
		if (!_isOwner)
		{
			_data = nullptr;
			_len = 0;
			return;
		}
		if (_data != nullptr)
		{
			delete[] _data;
			_data = nullptr;
			_len = 0;
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
		_len = 0;
	}

	void Resize(int n)
	{
		this->Clear();
		_len = n;
		_data = new T[n];

		for (int i = 0; i < n; i++)
			_data[i] = T();
	}

	int Size() const { return _len; }

	void Print()
	{
	}

	// 迭代器
	iterator begin() noexcept {
		return _data;
	}
	const_iterator begin() const noexcept {
		return _data;
	}
	iterator end() noexcept {
		return _data + _len;
	}
	const_iterator end() const noexcept {
		return _data + _len;
	}


	T& operator () (int i)
	{
		return _data[i];
	}
	T& operator [] (int i)
	{
		return _data[i];
	}
	const T& operator () (int i) const 
	{
		return _data[i];
	}
	const T& operator [] (int i) const 
	{
		return _data[i];
	}

	VectorT<T>& operator = (const VectorT<T>& mat)
	{
		if (this == &mat)
		{
			return *this;
		}
		if (this->_len != mat._len)
		{
			this->Resize(mat._len);
		}
		memcpy(_data, mat._data, _len * sizeof(T));
		/*for (int i = 0; i < _len; i++)
		{
			_data[i] = mat._data[i];
		}*/
		return *this;
	}

	VectorT<T>& operator = (VectorT<T>&& mat) noexcept
	{
		if (this != &mat)
		{
			if (_len > 0)
				delete[] _data;

			_len = mat._len;
			_data = mat._data;

			mat._data = nullptr;
			mat._len = 0;
		}
		return *this;
	}

	VectorT<T>& operator += (const VectorT<T>& vec)
	{
		if (_len != vec._len) {
			throw std::invalid_argument("Vector size mismatch for += operation");
		}
		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < _len; i++)
		{
			_data[i] += vec._data[i];
		}
		return *this;
	}
	VectorT<T>& operator -= (const VectorT<T>& vec)
	{
		if (_len != vec._len) {
			throw std::invalid_argument("Vector size mismatch for -= operation");
		}
		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < _len; i++)
		{
			_data[i] -= vec._data[i];
		}
		return *this;
	}
	VectorT<T>& operator *= (const T& val)
	{
		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < _len; i++)
		{
			_data[i] *= val;
		}
		return *this;
	}
	VectorT<T>& operator /= (const T& val)
	{
		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < _len; i++)
		{
			_data[i] /= val;
		}
		return *this;
	}




	friend VectorT<T> operator + (const VectorT<T>& vec1, const VectorT<T>& vec2)
	{
		if (vec1._len != vec2._len)
		{
			throw std::invalid_argument("Vector size mismatch for addition");
		}
		VectorT<T> Vsum(vec1._len);

		for (int i = 0; i < Vsum._len; i++)
		{
			Vsum._data[i] = vec1._data[i] + vec2._data[i];
		}
		return Vsum;
	}
	friend VectorT<T> operator - (const VectorT<T>& vec1, const VectorT<T>& vec2)
	{
		if (vec1._len != vec2._len)
		{
			throw std::invalid_argument("Vector size mismatch for addition");
		}
		VectorT<T> Vsum(vec1._len);

		for (int i = 0; i < Vsum._len; i++)
		{
			Vsum._data[i] = vec1._data[i] - vec2._data[i];
		}
		return Vsum;
	}

	friend VectorT<T> operator * (const T& val, const VectorT<T>& mat)
	{
		VectorT<T> Vec(mat._len);

		for (int i = 0; i < Vec._len; i++)
		{
			Vec._data[i] = mat._data[i] * val;
		}
		return Vec;
	}
	friend VectorT<T> operator * (const VectorT<T>& vec, const T& val)
	{
		VectorT<T> Vec(vec._len);

		for (int i = 0; i < Vec._len; i++)
		{
			Vec._data[i] = vec._data[i] * val;
		}
		return Vec;
	}
	friend VectorT<T> operator / (const VectorT<T>& vec, const T& val)
	{
		VectorT<T> Vec(vec._len);

		for (int i = 0; i < Vec._len; i++)
		{
			Vec._data[i] = vec._data[i] / val;
		}
		return Vec;
	}
	friend VectorT<T> operator / (const VectorT<T>& vec1, const VectorT<T>& vec2)
	{
		if (vec1._len != vec2._len)
		{
			throw std::invalid_argument("Vector size mismatch for addition");
		}
		VectorT<T> Vd(vec1._len);

		for (int i = 0; i < Vd._len; i++)
		{
			Vd._data[i] = vec1._data[i] / vec2._data[i];
		}
		return Vd;
	}
	

private:
	int _len;
	T* _data = nullptr;
	bool _isOwner = true;
};

typedef VectorT<unsigned char> iVec;
typedef VectorT<scalar> Vector;