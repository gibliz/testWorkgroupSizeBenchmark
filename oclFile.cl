#include "math.cpp"

float calcMath(float a, float b);

__kernel void TestKernel(
    __global const float* pInputVector1,
    __global const float* pInputVector2,
    __global float* pOutputVector,
    int nItems)
{
    // get index into global data array
    size_t iJob = get_global_id(0);

    // check boundary conditions
    if (iJob >= nItems) return;

    // perform calculations
    pOutputVector[iJob] = calcMath(pInputVector1[iJob], pInputVector2[iJob]);
} //> TestKernel()
