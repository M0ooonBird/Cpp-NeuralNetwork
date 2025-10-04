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
	NeuralNet(int L, const std::vector<int>& LSizes) :_L(L),_LSizes(LSizes)
	{
		// 默认使用RELU
		_atype = ActivationType::RELU;
 
		_H.resize(L + 2);
		_F.resize(L + 2);

		for (int i = 0; i < L + 2; i++)
		{
			_H[i].Resize(_LSizes[i]);
			_F[i].Resize(_LSizes[i]);
		}

		_idx_b.resize(L + 1);
		_idx_w.resize(L + 1);
		_idx_b[0] = 0;
		_idx_w[0] = _LSizes[1];
		for (int i = 1; i <= L; i++)
		{
			_idx_b[i] = _idx_w[i - 1] + LSizes[i] * LSizes[i - 1];
			_idx_w[i] = _idx_b[i] + _LSizes[i + 1];
		}
		
		_para_size = _idx_w[L] + _LSizes[L + 1] * _LSizes[L];
		_Parameters.Resize(_para_size);

	}
	~NeuralNet() {}

	// 设置激活函数类型
	void SetActivation(ActivationType type) { _atype = type; }

	// 导入所有训练数据
	void LoadData(std::vector<iMat>&& data,
		std::vector<iType>&& label, NN_Mode mode);

	void SetHyperParameter(scalar alpha, scalar beta = 0.9, scalar gamma = 0.999)
	{
		_alpha = alpha;
		_beta = beta;
		_gamma = gamma;
	}
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

	scalar _alpha = 0.001; // leraning rate
	scalar _beta = 0.9;	
	scalar _gamma = 0.999;

	int _L;
	std::vector<int> _LSizes;

	ActivationType _atype;

	std::vector<Vector> _F;
	std::vector<Vector> _H;

	std::vector<int> _idx_b;
	std::vector<int> _idx_w;
	size_t _para_size;

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
