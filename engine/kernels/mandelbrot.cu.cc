#ifdef __CUDACC__

#include <host_defines.h>
#include <math.h>
#include <vector_types.h>
#include "utilities/helper_math.h"

__global__ void mandelbrot(unsigned char *surface, int width, int height, size_t pitch, float t)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned char *pixel;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the pixel at (x,y)
	pixel = (unsigned char *)(surface + y*pitch) + 4 * x;

	float zoo = 0.62f + 0.38f*cosf(.07f * t);
	float coa = cosf(0.15f*(1.0f - zoo) * t);
	float sia = sinf(0.15f*(1.0f - zoo) * t);
	zoo = pow(zoo, 8.0f);

	float2 size = make_float2((float) width, (float) height);
	float2 param = make_float2(param.x = -0.745f, param.y = 0.186f);
	float2 pos = make_float2(pos.x = (float) x,	pos.y = (float) y);
	float2 p = (-size + 2.f * pos) / size;
	float2 xy = make_float2(p.x*coa - p.y*sia, p.x*sia + p.y*coa);
	float2 c = param + xy * zoo;
	float2 z = make_float2(0.f);
	float l = 0;
	for (int i = 0; i != 128; ++i) {
		z = make_float2(z.x*z.x - z.y*z.y, 2.f*z.x*z.y) + c;
		if (dot(z, z) > (128 * 128)) break;
		++l;
	}

	float r = 0.5f + 0.5f*cosf(3.0f + l*0.3f);
	float g = 0.5f + 0.5f*cosf(3.0f + l*0.3f + 0.6f);
	float b = 0.5f + 0.5f*cosf(3.0f + l*0.3f + 1.f);

	// populate it
	pixel[0] = (unsigned char)(r * 255.f); // red
	pixel[1] = (unsigned char)(g * 255.f); // green
	pixel[2] = (unsigned char)(b * 255.f); // blue
	pixel[3] = 255; // alpha
}

extern "C"
void cuda_mandelbrot(void *surface, int width, int height, size_t pitch, float t)
{
	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	mandelbrot<<<Dg, Db>>>( (unsigned char*) surface, width, height, pitch, t);
}

#endif //__CUDACC__
