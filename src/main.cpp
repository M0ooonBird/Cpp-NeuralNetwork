#include <iostream>
#include "Activation.h"
#include "NeuralNet.h"

int main(int argc, const char* argv[])
{
	std::cout << "Hello\n" << std::endl;
	std::cout << Activation::ReLU(1.0);

	int inputSize = 100;
	int outputSize = 10;
	int hsize = 200;
	NeuralNet nn(inputSize, outputSize, hsize);



	return 0;
}