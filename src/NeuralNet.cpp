#include "NeuralNet.h"
#include <cmath>
#include <random>
#include "MathFunction.h"

constexpr int threads_num = 16;

void NeuralNet::SetInput(const iType* data)
{
	#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < _iSize; i++)
	{
		_input[i] = data[i]; // 0-255
		// 归一化
		_input[i] /= 255.0;
		_input[i] = _input[i] * 2 - 1;
	}
}

void NeuralNet::Forward(bool isTrain)
{
	_hidden = _offset1 + _weight1 * _input;

	// 激活函数
	#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < _hSize; i++)
	{
		_hidden_a[i] = activate(_hidden[i]);
	}

	_output = _offset2 + _weight2 * _hidden_a;
}

void NeuralNet::InitWeights()
{
	std::random_device rd;
	std::mt19937 gen(rd());

	double std_dev1 = std::sqrt(2.0 / _iSize);
	std::normal_distribution<double> normal1(0.0, std_dev1);
	// 初始化权重
	for (auto& wi : _weight1) {
		wi = normal1(gen);
	}

	double std_dev2 = std::sqrt(2.0 / _hSize);
	std::normal_distribution<double> normal2(0.0, std_dev2);
	for (auto& wi : _weight2) {
		wi = normal2(gen);
	}
}
void NeuralNet::Train() 
{
	// initial parameters
	InitWeights();

	for (int i = 0; i < _epochNum; i++)
	{
		printf("Epoch: %d >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n", i);
		TrainEpoch();
	}
}

void NeuralNet::TrainEpoch()
{
	// 一轮训练有 n 块batch
	int n = (_trainNum - 1) / _batchSize + 1;

	// 1阶矩
	Vector Gradm_b2(_oSize);
	Vector Gradm_b1(_hSize);
	Matrix Gradm_w2(_oSize, _hSize);
	Matrix Gradm_w1(_hSize, _iSize);
	// 2阶矩
	Vector Gradv_b2(_oSize);
	Vector Gradv_b1(_hSize);
	Matrix Gradv_w2(_oSize, _hSize);
	Matrix Gradv_w1(_hSize, _iSize);

	//每一轮训练n次，每次取一块batch
	for (int t = 0; t < n; t++)
	{
		printf("batch: %d,\t", t);
		scalar Loss = 0;
		// batch中依次取样本
		Vector Grad_b2(_oSize); 
		Vector Grad_b1(_hSize);
		Matrix Grad_w2(_oSize, _hSize);
		Matrix Grad_w1(_hSize, _iSize);

		for (int i = 0; i < _batchSize; i++)
		{
			int idx = t * _batchSize + i;

			SetInput(_train_data[idx].data());
			Forward(true);//向前传播，记录中间值
			Softmax(_output); // output 所有元素约化到0-1之间
			int y = _train_label[idx]; // 样本对应的实际数字
			Loss += -std::log(_output[y]);

			Vector grad_b2i = _output;
			grad_b2i(y) -= 1;

			Grad_b2 += grad_b2i;
			Grad_w2 += Cross(grad_b2i , _hidden_a);

			Vector grad_b1i = _weight2.GetTransPose() * grad_b2i;
			for (int e = 0; e < _hSize; e++)
			{
				if(_hidden[e] < 0) // ReLU 特性
					grad_b1i[e] = 0;
			}
			Grad_b1 += grad_b1i;
			Grad_w1 += Cross(grad_b1i, _input);
		}

		Gradm_b2 = _beta * Gradm_b2 + (1 - _beta) * Grad_b2;
		Gradm_b1 = _beta * Gradm_b1 + (1 - _beta) * Grad_b1;
		Gradm_w2 = _beta * Gradm_w2 + (1 - _beta) * Grad_w2;
		Gradm_w1 = _beta * Gradm_w1 + (1 - _beta) * Grad_w1;
		
		Gradv_b2 = _gamma * Gradv_b2 + (1 - _gamma) * Square(Grad_b2);
		Gradv_b1 = _gamma * Gradv_b1 + (1 - _gamma) * Square(Grad_b1);
		Gradv_w2 = _gamma * Gradv_w2 + (1 - _gamma) * Square(Grad_w2);
		Gradv_w1 = _gamma * Gradv_w1 + (1 - _gamma) * Square(Grad_w1);
		
		auto r_Gradm_w2 = Gradm_w2 / (1 - std::pow(_beta, t + 1));
		auto r_Gradv_w2 = Gradv_w2 / (1 - std::pow(_gamma, t + 1));

		auto r_Gradm_w1 = Gradm_w1 / (1 - std::pow(_beta, t + 1));
		auto r_Gradv_w1 = Gradv_w1 / (1 - std::pow(_gamma, t + 1));

		auto r_Gradm_b2 = Gradm_b2 / (1 - std::pow(_beta, t + 1));
		auto r_Gradv_b2 = Gradv_b2 / (1 - std::pow(_gamma, t + 1));

		auto r_Gradm_b1 = Gradm_b1 / (1 - std::pow(_beta, t + 1));
		auto r_Gradv_b1 = Gradv_b1 / (1 - std::pow(_gamma, t + 1));
		// 随机梯度下降 SGD，更新参数
		_weight2 -= _alpha * r_Gradm_w2 / (Sqrt(r_Gradv_w2));
		_weight1 -= _alpha * r_Gradm_w1 / (Sqrt(r_Gradv_w1));
		_offset2 -= _alpha * r_Gradm_b2 / (Sqrt(r_Gradv_b2));
		_offset1 -= _alpha * r_Gradm_b1 / (Sqrt(r_Gradv_b1));

		printf("Loss = %f\n", Loss/_batchSize);

	}
}

void NeuralNet::Loss()
{
	//setinput;
	scalar Li = 0;

	// 取部分训练样本
	for (int i = 0; i < 10; i++)
	{
		//SetInput();
		this->Forward();
		Softmax(_output); // output 所有元素约化到0-1之间

		int yi = 0; // 数据输出 \in {0,1,2,...K-1}
		Li += -std::log(_output[yi]);
	}
}

void NeuralNet::Test()
{
	int correct = 0;
	for (int i = 0; i < _testNum; i++)
	{
		int idx = _trainNum + 1000 + i;

		SetInput(_train_data[idx].data()); 
		Forward(false);//向前传播，记录中间值
		Softmax(_output); // output 所有元素约化到0-1之间
		
		int yi = _train_label[idx]; // 样本对应的实际数字

		int y = 0;
		// 从第二个元素开始遍历
		for (size_t e = 1; e < _oSize; ++e) {
			if (_output[e] > _output[y]) {
				y = e; // 如果找到更大的元素，更新下标
			}
		}
		if (y == yi)
		{
			correct++;
		}
	}
	std::cout << "正确率：" << (double)correct / _testNum << std::endl;
}

void NeuralNet::Softmax(Vector& vec)
{
	scalar maxnum = vec[0];
	for (auto val : vec)
	{
		maxnum = std::max(maxnum, val);
	}
	scalar D = 0;
	for (auto val : vec)
	{
		D += std::exp(val - maxnum);
	}
	for (auto& val : vec)
	{
		val = std::exp(val - maxnum) / D;
	}
}

void NeuralNet::LoadTrainData(std::vector<iMat>&& train_data, 
	std::vector<iType>&& train_label)
{
	_train_data = std::move(train_data);
	_train_label = std::move(train_label);
}