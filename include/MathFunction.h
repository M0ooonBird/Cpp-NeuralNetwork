#pragma once
#include "Matrix.h"
#include "Vector.h"

// 向量内积
template <typename T>
T Dot(const VectorT<T>& vec1, const VectorT<T>& vec2);

// 向量外积
template <typename T>
MatrixT<T> Cross(const VectorT<T>& vec1, const VectorT<T>& vec2);

// 向量逐元素平方
template <typename T>
VectorT<T> Square(const VectorT<T>& vec1);

// 矩阵逐元素平方
template <typename T>
MatrixT<T> Square(const MatrixT<T>& mat);

// 向量逐元素开方
template <typename T>
VectorT<T> Sqrt(const VectorT<T>& vec1);

// 矩阵逐元素开方
template <typename T>
MatrixT<T> Sqrt(const MatrixT<T>& mat);

// 
int delta(int i, int j);

// 反转整数字节序（大端转小端）
int ReverseInt(int i);

int GetMaxvalueIdx(const scalar* v, int size);