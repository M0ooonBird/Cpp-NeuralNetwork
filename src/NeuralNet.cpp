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
		_H0[i] = data[i]; // 0-255
		// 归一化
		_H0[i] /= 255.0;
		_H0[i] = _H0[i] * 2 - 1;
	}
}

void NeuralNet::Forward(bool isTrain)
{
	Vector b1(_Parameters.data()+ idx_b0, _hSize);
	Matrix w1(_Parameters.data()+ idx_w0, _hSize, _iSize);
	_F1 = b1 + w1 * _H0;

	// 激活函数
	#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < _hSize; i++)
	{
		_H1[i] = activate(_F1[i]);
	}
	Vector b2(_Parameters.data() + idx_b1, _oSize);
	Matrix w2(_Parameters.data() + idx_w1, _oSize, _hSize);
	_H2 = b2 + w2 * _H1;

	Softmax(_H2); // output 所有元素约化到0-1之间
}

void NeuralNet::InitWeights()
{
	std::random_device rd;
	std::mt19937 gen(rd());

	// He初始化
	double std_dev1 = std::sqrt(2.0 / _iSize);
	std::normal_distribution<double> normal1(0.0, std_dev1);
	for (int i = _hSize; i < _hSize + _hSize * _iSize; i++)
	{
		_Parameters[i] = normal1(gen);  // w1
	}

	double std_dev2 = std::sqrt(2.0 / _hSize);
	std::normal_distribution<double> normal2(0.0, std_dev2);
	for (int i = _hSize + _hSize * _iSize + _oSize; i < _Parameters.Size(); i++)
	{
		_Parameters[i] = normal2(gen);  // w2
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
	Vector Gradm_p(_Parameters.Size());
	// 2阶矩
	Vector Gradv_p(_Parameters.Size());

	//每一轮训练n次，每次取一块batch
	for (int t = 0; t < n; t++)
	{
		printf("batch: %d,\t", t);
		scalar Loss = 0;
		// batch中依次取样本
		Vector Grad_p(_Parameters.Size());
		Vector Grad_b2(Grad_p.data() + idx_b1, _oSize);
		Vector Grad_b1(Grad_p.data() + idx_b0, _hSize);
		Matrix Grad_w2(Grad_p.data() + idx_w1, _oSize, _hSize);
		Matrix Grad_w1(Grad_p.data() + idx_w0, _hSize, _iSize);

		for (int i = 0; i < _batchSize; i++)
		{
			int idx = _shuffledIdx[t * _batchSize + i];

			SetInput(_train_data[idx].data());
			Forward(true);//向前传播，记录中间值
			
			int y = _train_label[idx]; // 样本对应的实际数字
			Loss += -std::log(_H2[y]);

			Vector grad_b2i = _H2;
			grad_b2i(y) -= 1;

			Grad_b2 += grad_b2i;
			Grad_w2 += Cross(grad_b2i , _H1);

			Matrix w2(_Parameters.data() + idx_w1, _oSize, _hSize);
			Vector grad_b1i = w2.GetTransPose() * grad_b2i;
			for (int e = 0; e < _hSize; e++)
			{
				if(_F1[e] < 0) // ReLU 特性
					grad_b1i[e] = 0;
			}
			Grad_b1 += grad_b1i;
			Grad_w1 += Cross(grad_b1i, _H0);
		}
		Gradm_p = _beta * Gradm_p + (1 - _beta) * Grad_p;
		Gradv_p = _gamma * Gradv_p + (1 - _gamma) * Square(Grad_p);

		auto r_Gradm = Gradm_p / (1 - std::pow(_beta, t + 1));
		auto r_Gradv = Gradv_p / (1 - std::pow(_gamma, t + 1));

		// 随机梯度下降 SGD，更新参数
		_Parameters -= _alpha * r_Gradm / Sqrt(r_Gradv);

		printf("Loss = %f\n", Loss/_batchSize);
	}
}

void NeuralNet::SetNNParameter(const std::vector<scalar>& para_data)
{
	memcpy(_Parameters.data(), para_data.data(), _Parameters.Size() * sizeof(scalar));
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

		int yi = 0; // 数据输出 \in {0,1,2,...K-1}
		Li += -std::log(_H2[yi]);
	}
}

void NeuralNet::Test()
{
	int correct = 0;
	for (int i = 0; i < _testNum; i++)
	{
		int idx = _shuffledIdx[i];

		SetInput(_test_data[idx].data()); 
		Forward(false);//向前传播，记录中间值
		
		int yi = _test_label[idx]; // 样本对应的实际数字
		int y = GetMaxvalueIdx(_H2.data(), _oSize);
		if (y == yi)
		{
			correct++;
		}
	}
	std::cout << "正确率：" << (double)correct / _testNum << std::endl;
}

void NeuralNet::Test(const iMat& image)
{
	auto data = image.data();
	#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < _iSize; i++)
	{
		_H0[i] = data[i]; // 0-255
		// 归一化
		_H0[i] /= 255.0;
		_H0[i] = _H0[i] * 2 - 1;
	}

	Forward(false);

	int y = GetMaxvalueIdx(_H2.data(), _oSize);

	printf("预测结果为：%d\n", y);
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

void NeuralNet::LoadData(
	std::vector<iMat>&& data, 
	std::vector<iType>&& label, 
	NN_Mode mode)
{
	if (mode == NN_Mode::TRAIN)
	{
		_train_data = std::move(data);
		_train_label = std::move(label);

		std::random_device rd;
		std::mt19937 gen(rd());
		_shuffledIdx.resize(_train_data.size());
		for (int i = 0; i < _shuffledIdx.size(); ++i) {
			_shuffledIdx[i] = i;
		}
		std::shuffle(_shuffledIdx.begin(), _shuffledIdx.end(), gen);
	}
	else
	{
		_test_data = std::move(data);
		_test_label = std::move(label);

		std::random_device rd;
		std::mt19937 gen(rd());
		_shuffledIdx.resize(_test_data.size());
		for (int i = 0; i < _shuffledIdx.size(); ++i) {
			_shuffledIdx[i] = i;
		}
		std::shuffle(_shuffledIdx.begin(), _shuffledIdx.end(), gen);
	}
}