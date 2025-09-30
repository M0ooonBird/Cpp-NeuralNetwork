#pragma once 
#include <vector>
#include "FloatType.h"
#include "Activation.h"

class NeuralNet
{
public:
	NeuralNet(int isize, int osize, int hsize) :_iSize(isize), _oSize(osize), _hSize(hsize) 
	{
		// 默认使用RELU
		_atype = ActivationType::RELU;

		_input.resize(_iSize);
		_output.resize(_oSize);
		_hidden.resize(_hSize);

		_weight1.resize(_iSize * _hSize);
		_weight2.resize(_oSize * _hSize);

		_offset1.resize(_hSize);
		_offset2.resize(_oSize);

	}
	~NeuralNet() {}

	void SetActivation(ActivationType type) { _atype = type; }

	void SetInput() {}
	void Forward(); // 前向传播，根据输入给出输出
	void Train();
	void Loss();
	static void Softmax(std::vector<scalar>& vec);

private:
	scalar activate(scalar z) {
		switch (_atype)
		{
		case ActivationType::RELU:
			return Activation::ReLU(z);
			break;
		case ActivationType::SIGMOID:
			return Activation::Sigmoid(z);
			break;
		default:
			break;
		}
	}


private:
	int _iSize;	//input
	int _oSize; //output

	int _hSize; // hidden layer

	ActivationType _atype;

	std::vector<scalar> _input;
	std::vector<scalar> _hidden;
	std::vector<scalar> _output;

	std::vector<scalar> _weight1;
	std::vector<scalar> _offset1;

	std::vector<scalar> _weight2;
	std::vector<scalar> _offset2;

};
