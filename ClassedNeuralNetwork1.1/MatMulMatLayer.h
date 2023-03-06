#pragma once
#include "Layer.h"

class MatMulMatLayer : public Layer
{
public:
	uint32_t inputMatrixSize;
	uint32_t outputMatrixSize;
	uint32_t weightMatrixSize;

	float* inputMatrix;
	float* weightMatrix;
	float* outputMatrix;
	float* outputDerivativeMatrix;
	float* weightDerivativeMatrix;
	float* inputDerivativeMatrix;
};