#pragma once
#include "Random.h"
#include <iostream>
#include <vector>
#include <fstream>

void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

void cpuSaxpy(int N, const float* alpha, const float* X, int incX, float* Y, int incY)
{
	for (int i = N; i--;)
		Y[i * incY] += *alpha * X[i * incX];
}

/*void cpuSoftmax(float* input, float* output, uint32_t size)
{
	float sum = 0;
	for (uint32_t counter = size; counter--;)
	{
		output[counter] = std::exp(input[counter]);
		sum += output[counter];
	}
	sum = 1.0f / sum;
	for (uint32_t counter = size; counter--;)
		output[counter] *= sum;
}

void cpuSoftmaxDerivative(float* input, float* output, bool endState, uint32_t action, uint32_t size)
{
	float sampledProbability = input[action];
	float gradient = (endState - sampledProbability);
	for (uint32_t counter = size; counter--;)
		output[counter] = gradient * input[counter] * ((counter == action) - sampledProbability);
}*/

void PrintMatrix(float* arr, uint32_t rows, uint32_t cols, const char* label) {
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

/*float invSqrt(float number)
{
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}*/

namespace GLOBAL
{
	Random RANDOM(Random::MakeSeed());
	constexpr float ZEROF = 0.0f;
	constexpr float ONEF = 1.0f;

	constexpr uint32_t BATCH_SIZE = 1;
	constexpr uint32_t ITERATIONS = 1620 * BATCH_SIZE;
	constexpr float LEARNING_RATE = 0.1f;
}

void cpuGenerateUniform(float* matrix, uint32_t size, float min = 0, float max = 1)
{
	for (uint32_t counter = size; counter--;)
		matrix[counter] = GLOBAL::RANDOM.Rfloat(min, max);
}