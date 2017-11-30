#ifdef __CUDACC__

/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <host_defines.h>

#define PI 3.1415926536f


/*
* Paint a 2D texture with a moving red/green hatch pattern on a
* strobing blue background.  Note that this kernel reads to and
* writes from the texture, hence why this texture was not mapped
* as WriteDiscard.
*/
__global__ void cuda_kernel_texture_2d(unsigned char *surface, int width, int height, size_t pitch, float t)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned char *pixel;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	pixel = (unsigned char *)(surface + y*pitch) + 4 * x;

	// populate it
	pixel[0] = 100 * x * t; // red
	pixel[1] = 100 * y * t; // green
	pixel[2] = 60 * cos(t); // blue
	pixel[3] = 255; // alpha
}

extern "C"
void cuda_texture_2d(void *surface, int width, int height, size_t pitch, float t)
{
	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	cuda_kernel_texture_2d<<<Dg, Db>>>((unsigned char *)surface, width, height, pitch, t);
}


#endif //__CUDACC__
