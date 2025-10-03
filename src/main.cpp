#include <iostream>
#include <fstream>
#include <random>
#include "Activation.h"
#include "NeuralNet.h"
#include "MathFunction.h"
#include "Matrix.h"
#include "FileIO.h"


int main(int argc, const char* argv[])
{
    
    const int input = 28 * 28;    // 图片大小
    const int output = 10;        // 预测 0,1,...,9
    NeuralNet* nn = new NeuralNet(3, { input, 256,128,64,output });

#if 1   // 训练模式，自动训练NN参数
    int train_num = 40000;  // 训练样本数量
    int batch = 128;        // 批次大小
    int epoch = 5;         // 训练轮数
    nn->SetTrainParameter(train_num, batch, epoch);

    // 导入训练集 一共60000个
    std::vector<iMat> train_data = read_mnist_images("train-images.idx3-ubyte");
    std::vector<iType> train_label = read_mnist_labels("train-labels.idx1-ubyte");
    nn->LoadData(std::move(train_data), std::move(train_label), NN_Mode::TRAIN);

    nn->Train();
    save_parameters_binary(nn->_Parameters.data(), nn->GetParaSize(), "parameter.dat");

    printf("训练完毕！\n");

    // 导入测试集
    std::vector<iMat> test_data = read_mnist_images("t10k-images.idx3-ubyte");
    std::vector<iType> test_label = read_mnist_labels("t10k-labels.idx1-ubyte");
    nn->LoadData(std::move(test_data), std::move(test_label), NN_Mode::TEST);
    int test_num = 5000;
    nn->SetTestNum(test_num);
    nn->Test();

#else   // 直接导入参数 (NN尺寸须匹配)
    auto para =  load_parameters_binary("parameter.dat");
    nn->SetNNParameter(para);

    // 导入测试集
    std::vector<iMat> test_data = read_mnist_images("t10k-images.idx3-ubyte");
    std::vector<iType> test_label = read_mnist_labels("t10k-labels.idx1-ubyte");
    nn->LoadData(std::move(test_data), std::move(test_label), NN_Mode::TEST);
    int test_num = 5000;
    nn->SetTestNum(test_num);
    nn->Test();

    auto image = preprocess_image("test2.png");
    nn->Test(image);
#endif

    delete nn;
	return 0;
}