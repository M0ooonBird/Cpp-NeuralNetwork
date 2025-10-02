#include "FileIO.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include "MathFunction.h"

// 保存参数到二进制文件
void save_parameters_binary(const scalar* params, int size, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    // 同样，先写入参数的数量
    size_t num_params = size;
    outFile.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    // 然后写入所有参数数据
    outFile.write(reinterpret_cast<const char*>(params), size * sizeof(scalar));

    outFile.close();
    std::cout << "Parameters saved to " << filename << " successfully." << std::endl;
}

// 从二进制文件加载参数
std::vector<scalar> load_parameters_binary(const std::string& filename) {
    std::vector<scalar> params;
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return params;
    }

    // 先读取参数的数量
    size_t num_params;
    inFile.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    params.resize(num_params);

    // 然后读取所有参数数据
    inFile.read(reinterpret_cast<char*>(params.data()), num_params * sizeof(scalar));

    inFile.close();
    std::cout << "Parameters loaded from " << filename << " successfully." << std::endl;
    return params;
}




// 将一张真实世界的手写数字图片预处理成 MNIST 格式
// input_path: 你的图片路径 (e.g., "my_digit.png")
iMat preprocess_image(const std::string& input_path) {
    // 1. 读取图像 (以灰度模式)
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not read the image: " << input_path << std::endl;
        return {};
    }

    // 2. 调整大小为 28x28
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(28, 28));

    // 3. 颜色反转  如果你的照片已经是黑底白字，则注释掉这一行。
    cv::bitwise_not(resized_img, resized_img);

    // 4. 二值化处理，得到纯黑白图像
    cv::Mat binary_img;
    // THRESH_BINARY: 像素值 > 127 的变为 255 (白), 否则变为 0 (黑)
    // THRESH_OTSU: 自动寻找最佳阈值，效果更好
    //cv::threshold(resized_img, binary_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // (可选) 显示处理后的图像，用于调试
     //cv::imshow("Processed Image", resized_img);
     //cv::waitKey(0);

    iMat input_vector(28);

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            // binary_img 的数据类型是 uchar (0-255)
            uchar pixel_value = static_cast<uchar>(resized_img.at<uchar>(i, j));
            input_vector(i, j) = pixel_value;
        }
    }

    return input_vector;
}



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
