#include "NeuralNet.h"
#include <cmath>

void NeuralNet::Forward()
{
	memset(_hidden.data(), 0, _hidden.size() * sizeof(scalar));
	// omp
	for (int i = 0; i < _hSize; i++)
	{
		_hidden[i] = _offset1[i];
		for (int j = 0; j < _iSize; j++)
		{
			_hidden[i] += _weight1[_iSize * i + j] * _input[j];
		}
	}
	// omp 激活函数
	for (int i = 0; i < _hSize; i++)
	{
		_hidden[i] = activate(_hidden[i]);
	}

	memset(_hidden.data(), 0, _hidden.size() * sizeof(scalar));
	// omp
	for (int i = 0; i < _oSize; i++)
	{
		_output[i] = _offset2[i];
		for (int j = 0; j < _hSize; j++)
		{
			_output[i] += _weight2[_hSize * i + j] * _hidden[j];
		}
	}
}
void NeuralNet::Train() 
{

}

void NeuralNet::Loss()
{
	//setinput;
	scalar Li = 0;

	// 取部分训练样本
	for (int i = 0; i < 10; i++)
	{
		SetInput();
		this->Forward();
		Softmax(_output); // output 所有元素约化到0-1之间

		int yi = 0; // 数据输出 \in {0,1,2,...K-1}
		Li += -std::log(_output[yi]);

	}


}

void NeuralNet::Softmax(std::vector<scalar>& vec)
{
	scalar D = 0;
	for (auto val : vec)
	{
		D += std::exp(val);
	}
	for (auto& val : vec)
	{
		val = std::exp(val) / D;
	}
}