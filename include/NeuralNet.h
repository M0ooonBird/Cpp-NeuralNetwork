#pragma once 
#include <vector>
#include "FloatType.h"
#include "Activation.h"

class NeuralNet
{
public:
	NeuralNet() {}
	~NeuralNet() {}


	scalar activate(scalar z) { return 0; }


private:
	int _iSize;	//input
	int _oSize; //output

	int _hSize; // hidden layer

	ActivationType _atype;

	std::vector<scalar> _input;
	std::vector<scalar> _hidden;
	std::vector<scalar> _output;

	std::vector<scalar> _weight1;
	std::vector<scalar> _offset1;

	std::vector<scalar> _weight2;
	std::vector<scalar> _offset2;

};
