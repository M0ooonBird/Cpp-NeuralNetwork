#pragma once

#include "FloatType.h"
#include <cmath>

enum class ActivationType
{
	RELU = 1000,
	SIGMOID
};


namespace Activation 
{
	inline scalar ReLU(scalar z)
	{
		return (z > 0) ? z : 0;
	}

	inline scalar Sigmoid(scalar z)
	{
		return 1.0 / (1.0 + std::exp(-z));
	}

}
