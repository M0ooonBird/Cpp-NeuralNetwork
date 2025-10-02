#pragma once 
#include <vector>
#include "FloatType.h"
#include "Activation.h"
#include "Matrix.h"
#include "Vector.h"

using iType = unsigned char;
class NeuralNet
{
	
public:
	NeuralNet(int isize, int osize, int hsize) :_iSize(isize), _oSize(osize), _hSize(hsize) 
	{
		// 默认使用RELU
		_atype = ActivationType::RELU;

		_input.Resize(_iSize);

		_output.Resize(_oSize);
		_hidden.Resize(_hSize);
		_hidden_a.Resize(_hSize);

		_weight1.Resize(_hSize, _iSize);
		_weight2.Resize(_oSize, _hSize);

		_offset1.Resize(_hSize);
		_offset2.Resize(_oSize);

	}
	~NeuralNet() {}

	// 设置激活函数类型
	void SetActivation(ActivationType type) { _atype = type; }

	// 导入所有训练数据
	void LoadTrainData(std::vector<iMat>&& train_data,
		std::vector<iType>&& train_label);
	void SetTrainParameter(int train_num, int batch, int epoch) 
	{
		_trainNum = train_num;
		_batchSize = batch;
		_epochNum = epoch;
	}
	void SetTestParameter(int t)
	{
		_testNum = t;
	}

	// 设置输入Vec
	void SetInput(const iType* data);
	// 前向传播，根据输入给出输出
	void Forward(bool isTrain = false); 

	// 随机初始化权重
	void InitWeights();
	void Train();
	void TrainEpoch();

	void Test();
	void Test(const iMat& image);
	void Loss();
	static void Softmax(Vector& vec);

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
	scalar _alpha = 0.0002;
	scalar _beta = 0.9;
	scalar _gamma = 0.999;

	int _iSize;	//input
	int _oSize; //output
	int _hSize; // hidden layer

	ActivationType _atype;

	Vector _input;
	Vector _hidden;
	Vector _hidden_a;
	Vector _output;

	Matrix _weight1;
	Vector _offset1;

	Matrix _weight2;
	Vector _offset2;

	std::vector<int> _shuffledIdx;
	std::vector<iMat> _train_data;
	std::vector<iType> _train_label;
	std::vector<iMat> _test_data;
	std::vector<iType> _test_label;

	int _trainNum;
	int _batchSize;
	int _epochNum;

	int _testNum;
};
