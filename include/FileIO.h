#pragma once
#include <iostream>
#include <vector>
#include "Matrix.h"

// 保存参数到二进制文件
void save_parameters_binary(const scalar* params, int size, const std::string& filename);

// 从二进制文件加载参数
std::vector<scalar> load_parameters_binary(const std::string& filename);

// 将一张真实世界的手写数字图片预处理成 MNIST 格式
// input_path: 你的图片路径 (e.g., "my_digit.png")
iMat preprocess_image(const std::string& input_path);


std::vector<iMat> read_mnist_images(const std::string& file_path);

std::vector<iType> read_mnist_labels(const std::string& file_path);