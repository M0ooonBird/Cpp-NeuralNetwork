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
    const int hsize = 256;        // 隐藏层大小
    NeuralNet* nn = new NeuralNet(input, output, hsize);

    auto para = load_parameters_binary("parameter.dat");
    nn->SetNNParameter(para);

    if (argc < 2) {
        std::cerr << "Error: No image file provided." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        std::cerr << "Example: " << argv[0] << " my_handwritten_digit.png" << std::endl;
        return 1; 
    }

    // 从 argv 获取图片文件名
    std::string image_path = argv[1];

    auto image = preprocess_image(image_path);
    nn->Test(image);

    delete nn;
	return 0;
}