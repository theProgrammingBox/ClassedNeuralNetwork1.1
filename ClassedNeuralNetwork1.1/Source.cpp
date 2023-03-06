#include "NeuralNetwork.h"

int main()
{
	constexpr uint32_t INPUT_MATRIX_ROWS = 1;
	constexpr uint32_t INPUT_MATRIX_COLS = 2;
	
	NeuralNetwork neuralNetwork(GLOBAL::LEARNING_RATE, GLOBAL::BATCH_SIZE);
	float* inputMatrix = neuralNetwork.AddInput(INPUT_MATRIX_ROWS, INPUT_MATRIX_COLS);
	neuralNetwork.AddLayer(new LeakyReluLayer(3));
	
	return 0;
}