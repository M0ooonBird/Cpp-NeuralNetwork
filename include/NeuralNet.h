#pragma once 
#include <vector>
#include "FloatType.h"
#include "Activation.h"
#include "Matrix.h"
#include "Vector.h"

enum class NN_Mode
{
	TRAIN,
	TEST
};


class NeuralNet
{
	
public:
	NeuralNet(int isize, int osize, int hsize) :_iSize(isize), _oSize(osize), _hSize(hsize) 
	{
		// 默认使用RELU
		_atype = ActivationType::RELU;

		_H0.Resize(_iSize);
		_F1.Resize(_hSize);
		_H1.Resize(_hSize);
		_H2.Resize(_oSize);

		idx_b0 = 0;
		idx_w0 = idx_b0 + _hSize;
		idx_b1 = idx_w0 + _hSize * _iSize;
		idx_w1 = idx_b1 + _oSize;
		_Parameters.Resize(_hSize + _hSize * _iSize + _oSize + _oSize * _hSize);
		_para_size = _Parameters.Size();
	}
	~NeuralNet() {}

	// 设置激活函数类型
	void SetActivation(ActivationType type) { _atype = type; }

	// 导入所有训练数据
	void LoadData(std::vector<iMat>&& data,
		std::vector<iType>&& label, NN_Mode mode);
	void SetTrainParameter(int train_num, int batch, int epoch) 
	{
		_trainNum = train_num;
		_batchSize = batch;
		_epochNum = epoch;
	}
	void SetTestNum(int t)
	{
		_testNum = t;
	}
	void SetNNParameter(const std::vector<scalar>& data);

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
	int GetParaSize() { return _para_size; }

	static void Softmax(Vector& vec);

	Vector _Parameters;


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

	scalar _alpha = 0.0002;
	scalar _beta = 0.9;
	scalar _gamma = 0.999;

	int _iSize;	//input
	int _oSize; //output
	int _hSize; // hidden layer

	ActivationType _atype;

	Vector _H0;
	Vector _F1;
	Vector _H1;
	Vector _H2;

	int idx_b0;
	int idx_w0;
	int idx_b1;
	int idx_w1;
	int _para_size;

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
