#pragma once
#include "Header.h"

class Layer
{
public:
	Layer() {};
	virtual ~Layer() {};
	
	virtual void ForwardPropagate() = 0;
	virtual void BackPropagate() = 0;
};
