#include "MathFunction.h"
#include <cmath>

constexpr scalar eps = 1.0E-7;

int delta(int i, int j)
{
	return (i == j) ? 1 : 0;
}

int ReverseInt(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 0xFF;         // 最低位字节
	ch2 = (i >> 8) & 0xFF;  // 次低位字节
	ch3 = (i >> 16) & 0xFF; // 次高位字节
	ch4 = (i >> 24) & 0xFF; // 最高位字节
	return (ch1 << 24) + (ch2 << 16) + (ch3 << 8) + ch4;
}

template <typename T>
T Dot(const VectorT<T>& vec1, const VectorT<T>& vec2)
{
	if (vec1.Size() != vec2.Size())
	{
		throw std::invalid_argument("Vector size mismatchs");
	}
	T sum = 0;

	for (int i = 0; i < vec1.Size(); i++)
	{
		sum += vec1[i] * vec2[2];
	}
	return sum;
}

template <typename T>
MatrixT<T> Cross(const VectorT<T>& vec1, const VectorT<T>& vec2)
{
	MatrixT<T> MP(vec1.Size(), vec2.Size());
	for (int i = 0; i < vec1.Size(); i++)
	{
		for (int j = 0; j < vec2.Size(); j++)
		{
			MP(i, j) = vec1(i) * vec2(j);
		}
	}
	return MP;
}

template <typename T>
VectorT<T> Square(const VectorT<T>& vec)
{
	VectorT<T> R(vec);
	for (auto& v : R)
	{
		v *= v;
	}
	return R;
}

template <typename T>
MatrixT<T> Square(const MatrixT<T>& mat)
{
	MatrixT<T> R(mat);
	for (auto& v : R)
	{
		v *= v;
	}
	return R;
}

template <typename T>
VectorT<T> Sqrt(const VectorT<T>& vec)
{
	VectorT<T> R(vec);
	for (auto& v : R)
	{
		v = std::sqrt(v) + eps;
	}
	return R;
}

template <typename T>
MatrixT<T> Sqrt(const MatrixT<T>& mat)
{
	MatrixT<T> R(mat);
	for (auto& v : R)
	{
		v = std::sqrt(v) + eps;
	}
	return R;
}

template
VectorT<scalar> Square<scalar>(const VectorT<scalar>& vec1);
template
MatrixT<scalar> Square<scalar>(const MatrixT<scalar>& mat);
template
VectorT<scalar> Sqrt<scalar>(const VectorT<scalar>& vec1);
template
MatrixT<scalar> Sqrt<scalar>(const MatrixT<scalar>& mat);


template
scalar Dot<scalar>(const VectorT<scalar>& vec1, const VectorT<scalar>& vec2);
template
MatrixT<scalar> Cross<scalar>(const VectorT<scalar>& vec1, const VectorT<scalar>& vec2);