#include "NeuralNet.h"
#include <cmath>
#include <random>
#include "MathFunction.h"

constexpr int threads_num = 16;

void NeuralNet::SetInput(const iType* data)
{
	#pragma omp parallel for num_threads(threads_num)
	for (int e = 0; e < _LSizes[0]; e++)
	{
		_H[0][e] = data[e]; // 0-255
		// 归一化
		_H[0][e] /= 255.0;
		_H[0][e] = _H[0][e] * 2 - 1;
	}
}

void NeuralNet::Forward(bool isTrain)
{
	for (int i = 0; i < _L; i++)
	{
		Vector bi(_Parameters.data() + _idx_b[i], _LSizes[i+1]);
		Matrix wi(_Parameters.data() + _idx_w[i], _LSizes[i+1], _LSizes[i]);
		_F[i+1] = bi + wi * _H[i];

		// 激活函数
		#pragma omp parallel for num_threads(threads_num)
		for (int e = 0; e < _LSizes[i+1]; e++)
		{
			_H[i+1][e] = activate(_F[i+1][e]);
		}
	}
	int i = _L;
	Vector bi(_Parameters.data() + _idx_b[i], _LSizes[i + 1]);
	Matrix wi(_Parameters.data() + _idx_w[i], _LSizes[i + 1], _LSizes[i]);
	_F[i + 1] = bi + wi * _H[i];

	Softmax(_F[i+1]); // output 所有元素约化到0-1之间
	_H[i+1] = _F[i+1];
}

void NeuralNet::InitWeights()
{
	std::random_device rd;
	std::mt19937 gen(rd());

	// He初始化
	for (int i = 0; i < _L + 1; i++)
	{
		double std_dev = std::sqrt(2.0 / _LSizes[i]);
		std::normal_distribution<double> normal(0.0, std_dev);

		Matrix wi(_Parameters.data() + _idx_w[i], _LSizes[i + 1], _LSizes[i]);
		for (auto& val : wi)
		{
			val = normal(gen);
		}
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
	Vector Gradm_p(_para_size);
	// 2阶矩
	Vector Gradv_p(_para_size);

	//每一轮训练n次，每次取一块batch
	for (int t = 0; t < n; t++)
	{
		printf("batch: %d,\t", t);
		scalar Loss = 0;
		// 新的梯度 清零
		Vector Grad_p(_para_size);
		std::vector<Vector> Grad_b;
		std::vector<Matrix> Grad_w;
		for (int i = 0; i < _L + 1; i++)
		{
			Grad_b.emplace_back(Grad_p.data() + _idx_b[i], _LSizes[i+1]);
			Grad_w.emplace_back(Grad_p.data() + _idx_w[i], _LSizes[i+1], _LSizes[i]);
		}
		// batch中依次取样本
		for (int b = 0; b < _batchSize; b++)
		{
			int idx = _shuffledIdx[t * _batchSize + b];

			SetInput(_train_data[idx].data());
			Forward(true);//向前传播，记录中间值
			
			int y = _train_label[idx]; // 样本对应的实际数字
			Loss += -std::log(_H[_L + 1][y]);

			// Backward propagation
			Vector grad_bL = _H[_L + 1]; grad_bL(y) -= 1;
			Grad_b[_L] += grad_bL;
			Grad_w[_L] += Cross(grad_bL, _H[_L]);

			for (int i = _L - 1; i >= 0; i--)
			{
				Matrix wi(_Parameters.data() + _idx_w[i+1], _LSizes[i+2], _LSizes[i+1]);
				Vector grad_bl = wi.GetTransPose() * grad_bL;
				for (int e = 0; e < _LSizes[i+1]; e++)
				{
					if (_F[i+1][e] < 0) // ReLU 特性
						grad_bl[e] = 0;
				}
				Grad_b[i] += grad_bl;
				Grad_w[i] += Cross(grad_bl, _H[i]);
				grad_bL = grad_bl;
			}
			// Backward propagation
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
		int y = GetMaxvalueIdx(_H[_L+1].data(), _LSizes[_L+1]);
		if (y == yi)
		{
			correct++;
		}
	}
	std::cout << "Accuracy rate: " << (double)correct / _testNum << std::endl;
}

void NeuralNet::Test(const iMat& image)
{
	auto data = image.data();
	#pragma omp parallel for num_threads(threads_num)
	for (int i = 0; i < _LSizes[0]; i++)
	{
		_H[0][i] = data[i]; // 0-255
		// 归一化
		_H[0][i] /= 255.0;
		_H[0][i] = _H[0][i] * 2 - 1;
	}

	Forward(false);

	int y = GetMaxvalueIdx(_H[_L + 1].data(), _LSizes[_L + 1]);

	printf("number = %d\n", y);
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