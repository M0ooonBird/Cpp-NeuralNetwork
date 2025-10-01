#include <iostream>
#include <fstream>
#include "Activation.h"
#include "NeuralNet.h"
#include "MathFunction.h"
#include "Matrix.h"


std::vector<iMat> read_mnist_images(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("无法打开文件");

    // 读取文件头并转换字节序
    int magic_number, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    magic_number = ReverseInt(magic_number);
    num_images = ReverseInt(num_images);
    rows = ReverseInt(rows);
    cols = ReverseInt(cols);

    // 验证魔数
    if (magic_number != 2051) throw std::runtime_error("无效的MNIST图像文件");

    // 读取像素数据到矩阵（每张图是28x28的vector）
    std::vector<iMat> images(num_images, iMat(rows, cols));
    for (int i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
    }
    return images;
}

std::vector<iType> read_mnist_labels(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    int magic_number, num_labels;
    // 读取魔数和标签数量（各4字节）
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

    // 反转字节序
    magic_number = ReverseInt(magic_number);
    num_labels = ReverseInt(num_labels);

    // 验证魔数（必须是2049）
    if (magic_number != 2049) {
        throw std::runtime_error("无效的MNIST标签文件，魔数应为2049");
    }

    // 读取标签数据到vector
    std::vector<iType> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return labels;
}


int main(int argc, const char* argv[])
{
    // 一共60000个训练/测试数据
    std::vector<iMat> train_data = read_mnist_images("train-images.idx3-ubyte");
    std::vector<iType> train_label = read_mnist_labels("train-labels.idx1-ubyte");
        
    int train_num = 30000; // 训练样本数量
    int batch = 128;
    int epoch = 15; // 训练轮数

    int input = 28 * 28; // 图片大小
    int output = 10;    // 预测 0,1,...,9
    int hsize = 256;
    NeuralNet* nn = new NeuralNet(input, output, hsize);

    nn->LoadTrainData(std::move(train_data),std::move(train_label));
    nn->SetTrainParameter(train_num, batch, epoch);

    nn->Train();

    int test_num = 5000;
    nn->SetTestParameter(test_num);
    nn->Test();

    delete nn;
	return 0;
}