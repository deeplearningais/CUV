#include <iostream>
#include <cuda.h>
#include <tools/texture.h>
#include <stdexcept>
#include <cuv_general.hpp>

#include <basics/dev_dense_matrix.hpp>
#include <basics/host_dense_matrix.hpp>

#include "move.hpp"
using namespace std;


/** 
 * @brief convert four rgb pixels to gray simultaenously
 */
uchar4 __host__ __device__ rgb_to_y(uchar4 pixel1, uchar4 pixel2, uchar4 pixel3, uchar4 pixel4)
{
	return make_uchar4(
			0.299f * pixel1.x + 0.587f * pixel1.y + 0.114f * pixel1.z,
			0.299f * pixel2.x + 0.587f * pixel2.y + 0.114f * pixel2.z,
			0.299f * pixel3.x + 0.587f * pixel3.y + 0.114f * pixel3.z,
			0.299f * pixel4.x + 0.587f * pixel4.y + 0.114f * pixel4.z);
}

/** 
 * @brief convert an rgb pixel to gray
 */
uchar4 __host__ __device__ rgb_to_y(uchar1 pixel1, uchar1 pixel2, uchar1 pixel3, uchar1 pixel4)
{
	return make_uchar4(pixel1.x, pixel2.x, pixel3.x, pixel4.x);
}


/** 
 * @brief bilinear interpolation of for 4 pixels simultaenously
 */
uchar4 __host__ __device__ interpolate(uchar4 pixel1, uchar4 pixel2, uchar4 pixel3, uchar4 pixel4, float xfrac, float yfrac)
{
	return make_uchar4(
			(1.0f-yfrac) * ((1.0f-xfrac)*pixel1.x + xfrac*pixel2.x) + yfrac * ((1.0f-xfrac)*pixel3.x + xfrac*pixel4.x),
			(1.0f-yfrac) * ((1.0f-xfrac)*pixel1.y + xfrac*pixel2.y) + yfrac * ((1.0f-xfrac)*pixel3.y + xfrac*pixel4.y),
			(1.0f-yfrac) * ((1.0f-xfrac)*pixel1.z + xfrac*pixel2.z) + yfrac * ((1.0f-xfrac)*pixel3.z + xfrac*pixel4.z),
			0);
}


/** 
 * @brief bilinear interpolation for single pixel
 */
unsigned char __host__ __device__ interpolate(unsigned char pixel1, unsigned char pixel2, unsigned char pixel3, unsigned char pixel4, float xfrac, float yfrac)
{
	return ((1.0f-yfrac) * ((1.0f-xfrac)*pixel1 + xfrac*pixel2) + yfrac * ((1.0f-xfrac)*pixel3 + xfrac*pixel4));
}

/** 
 * @brief bilinear interpolation for single pixel
 */
uchar1 __host__ __device__ interpolate(uchar1 pixel1, uchar1 pixel2, uchar1 pixel3, uchar1 pixel4, float xfrac, float yfrac)
{
	return make_uchar1(
			(1.0f-yfrac) * ((1.0f-xfrac)*pixel1.x + xfrac*pixel2.x) + yfrac * ((1.0f-xfrac)*pixel3.x + xfrac*pixel4.x));
}

/**
 * @brief fetch a pixel using texture memory/global memory depending on template param
 */
template<bool UseCache>
__device__ uchar4 get_pixel(bool inrange, unsigned int index, uchar4 oorcolor, const uchar4* base)
{
		return (inrange ? fetch_x<UseCache>(base, index) : oorcolor);
}

/**
 * @brief fetch a pixel using texture memory/global memory depending on template param
 */
template<bool UseCache>
__device__ uchar1 get_pixel(bool inrange, unsigned int index, uchar1 oorcolor, const uchar1* base)
{
		return (inrange ? fetch_x<UseCache>(base, index) : oorcolor);
}

/**
 * @brief fetch a pixel using texture memory/global memory depending on template param
 */
template<bool UseCache>
__device__ uchar4 get_pixel(unsigned int index, const uchar4* base, uchar4 orgcolor, float dst)
{
		uchar4 p = fetch_x<UseCache>(base, index);
		p.x  = dst*orgcolor.x + (1.f-dst)*p.x;
		p.y  = dst*orgcolor.y + (1.f-dst)*p.y;
		p.z  = dst*orgcolor.z + (1.f-dst)*p.z;
		return p;
}

/**
 * @brief fetch a pixel using texture memory/global memory depending on template param
 */
template<bool UseCache>
__device__ uchar1 get_pixel(unsigned int index, const uchar1* base, uchar1 orgcolor, float dst)
{
		/*return dst*fetch_x<UseCache>(base, index) + (1.f-dst)*orgcolor.x;*/
		uchar1 p = fetch_x<UseCache>(base, index);
		p.x  = dst*orgcolor.x + (1.f-dst)*p.x;
		return p;
}


/**
 * @brief set a pixel to gray
 */
__device__ void set_default_color(uchar4& p){ p = make_uchar4(128,128,128,0); }
/**
 * @brief set a pixel to gray
 */
__device__ void set_default_color(uchar1& p){ p = make_uchar1(128); }

/**
 * set the three input maps to the decorrelated color values.
 */
template <class T>
void __host__ __device__ set_pca_maps (T* map1, T* map2, T* map3, unsigned int index, uchar4 pixel)
{
	map1[index] = (T)((-0.5525f*pixel.x - 0.5719f*pixel.y - 0.6063f*pixel.z + 441.3285f) * 0.004531772f - 1.0f);
	map2[index] = (T)(( 0.7152f*pixel.x + 0.0483f*pixel.y - 0.6973f*pixel.z + 177.8115f) * 0.005369070f - 1.0f);
	map3[index] = (T)((-0.4281f*pixel.x + 0.8189f*pixel.y - 0.3823f*pixel.z + 206.6520f) * 0.004813808f - 1.0f);
}

/**
 * set the first input map to the gray value of the pixel
 */
template <class T>
void __host__ __device__ set_pca_maps(T* map1, T* map2, T* map3, unsigned int index, uchar1 pixel)
{
	map1[index] = (T)(pixel.x * 0.007843137f - 1.0f);
}


/** 
 * @brief kernel for moving images up and down
 * @note  mostly shamelessly stolen from rafael uetz
 * 
 * @param mapsize            number of pixels in an output map
 * @param src                source pixels 
 * @param xshift             how much to shift left/right
 * @param yshift             how much to shift up/down
 * @param patwidth           width and height of the target image
 * @param enlarge            whether to scale up the image
 * 
 */
template<bool UseCache, class dst_pixel, class pixel>
__global__ void 
move_image_kernel(dst_pixel* dst, const pixel* src, char xshift, char yshift, unsigned int patwidth, unsigned char dst_num_maps, bool enlarge){
	
	const int iw = blockDim.x;                  // Determine input map width

	
	// Get x- and y-position of the input maps represented by the current thread
	const int mapx = threadIdx.x;
	const int mapy = blockIdx.x;

	// Get x- and y-position of the input pattern for the current position on the input maps.
	// Set inrange to false if the calculated position is out of range.
	int patx, paty;
	float patxf = 0.0f, patyf = 0.0f;
	bool inrange;
	pixel default_color = fetch_x<UseCache>(src, patwidth*patwidth*blockIdx.y);

	if (enlarge)
	{
		// Calculate x- and y-position on the input pattern for this thread
		patxf = (float(mapx - xshift) / iw) * patwidth;
		patyf = (float(mapy - yshift) / iw) * patwidth;

		// Store rounded position
		patx = int(patxf);
		paty = int(patyf);

		// Calculate remainder (required for interpolation)
		patxf -= patx;
		patyf -= paty;

		// Determine if the map pixel represented by the current thread shows a pixel of the pattern
		// (inrange=true) or is filled with the default color (inrange=false)
		inrange = (mapx >= xshift) && (mapy >= yshift) && (mapx < iw+xshift) && (mapy < iw+yshift);
		if(!inrange){
			char xn = max(1,min(patx,(int)patwidth-2));
			char yn = max(1,min(paty,(int)patwidth-2));
			default_color = get_pixel<UseCache>(patwidth*patwidth*blockIdx.y + patwidth*yn + xn,src, default_color,min(1.f,max(0.f,0.13f*(float)(abs(patx-xn)+abs(paty-yn))) ));
		}
	}
	else
	{
		// Determines at which x- and y-position of the map the pattern starts
		/*const int offset = iw/2 - patwidth/2;*/
		const int offset = 0;

		// Calculate x- and y-position on the input pattern for this thread
		patx = mapx - offset - xshift;
		paty = mapy - offset - yshift;

		// Determine if the map pixel represented by the current thread shows a pixel of the pattern
		// (inrange=true) or is filled with the default color (inrange=false)
		inrange = (patx >= 0) && (patx < patwidth) && (paty >= 0) && (paty < patwidth);
		if(!inrange){
			char xn = max(1,min(patx,(int)patwidth-2));
			char yn = max(1,min(paty,(int)patwidth-2));
			default_color = get_pixel<UseCache>(patwidth*patwidth*blockIdx.y + patwidth*yn + xn,src, default_color,min(1.f,max(0.f,0.13f*(float)(abs(patx-xn)+abs(paty-yn)) )));
		}
	}

	// Get index of processed pattern in the current mini batch
	const unsigned int patidx  = blockIdx.y;

	pixel pixel1, pixel2, pixel3, pixel4;
	pixel ipx;
	uchar4 graypx,grayipx;

	// Fetch colors of four adjacent pixels from texture.
	// If out of range, use default color defined above.
	// 1 2
	// 3 4
	pixel1 = get_pixel<UseCache>(inrange, patwidth*patwidth*patidx + patwidth*(paty+0) + (patx+0), default_color,src);
	pixel2 = get_pixel<UseCache>(inrange, patwidth*patwidth*patidx + patwidth*(paty+0) + (patx+1), default_color,src);
	pixel3 = get_pixel<UseCache>(inrange, patwidth*patwidth*patidx + patwidth*(paty+1) + (patx+0), default_color,src);
	pixel4 = get_pixel<UseCache>(inrange, patwidth*patwidth*patidx + patwidth*(paty+1) + (patx+1), default_color,src);

	// Calculate gray values of each of the four pixels.
	// x y
	// z w
	graypx = rgb_to_y(pixel1, pixel2, pixel3, pixel4);

	// Interpolate color and edges for current position from the four source pixels if enlargement is enabled
	if (enlarge)
	{
		const float gap = float(patwidth) / iw;
		ipx       = interpolate(pixel1, pixel2, pixel3, pixel4, patxf, patyf);
		grayipx.x = interpolate(graypx.x, graypx.y, graypx.z, graypx.w, max(patxf-gap/2, 0.0f), max(patyf-gap/2, 0.0f));
		grayipx.y = interpolate(graypx.x, graypx.y, graypx.z, graypx.w, min(patxf+gap/2, 1.0f), max(patyf-gap/2, 0.0f));
		grayipx.z = interpolate(graypx.x, graypx.y, graypx.z, graypx.w, max(patxf-gap/2, 0.0f), min(patyf+gap/2, 1.0f));
		grayipx.w = interpolate(graypx.x, graypx.y, graypx.z, graypx.w, min(patxf+gap/2, 1.0f), min(patyf+gap/2, 1.0f));
	}
	else
	{
		ipx = pixel1;
		grayipx = graypx;
	}

	const unsigned int wholeimgsize = dst_num_maps*iw*iw;
	set_pca_maps(dst +    wholeimgsize*patidx + 0*iw*iw
			,    dst +    wholeimgsize*patidx + 1*iw*iw 
			,    dst +    wholeimgsize*patidx + 2*iw*iw, iw*mapy + mapx, ipx);
}


#define V(X) #X << "=" <<(X) << ", "
namespace cuv
{
	namespace image_move_impl
	{
		template<class __value_typeA, class __value_typeB>
		void image_move(dev_dense_matrix<__value_typeA,column_major>& dst, const dev_dense_matrix<__value_typeB,column_major>& src, 
			const unsigned int& src_image_size, 
			const unsigned int& dst_image_size,
			const unsigned int& src_num_maps,
			const char& xshift, 
			const char& yshift){


			const unsigned char dst_num_maps = src_num_maps == 4 ? 3 : 1;

			cuvAssert(src.w() == dst.w());
			cuvAssert(src.h() % (src_image_size*src_num_maps) == 0);
			cuvAssert(dst.h() % (dst_image_size*dst_num_maps) == 0);

			dim3 blockDim(dst_image_size);
			dim3 gridDim (dst_image_size,src.w());
			static const bool UseCache = true;
			const bool enlarge = dst_image_size != src_image_size;
			if(src_num_maps == 4){
				typedef uchar4 T;
				const T* src_ptr = reinterpret_cast<const T*>(src.ptr());
				if(UseCache)
					bind_x(src_ptr, src.n()/src_num_maps);
				move_image_kernel<UseCache><<<gridDim,blockDim>>>(dst.ptr(),src_ptr,xshift,yshift,src_image_size,dst_num_maps,enlarge); 
				if(UseCache)
					unbind_x(src_ptr);
			}else if(src_num_maps == 1){
				typedef uchar1 T;
				const T* src_ptr = reinterpret_cast<const T*>(src.ptr());
				if(UseCache)
					bind_x(src_ptr, src.n());
				move_image_kernel<UseCache><<<gridDim,blockDim>>>(dst.ptr(),src_ptr,xshift,yshift,src_image_size,dst_num_maps,enlarge);
				if(UseCache)
					unbind_x(src_ptr);
			}else{
				throw std::runtime_error("wrong image format: Need RGBA interleaved _or_ grayscale");
			}
			cuvSafeCall(cudaThreadSynchronize());
		}
		template<class __value_typeA, class __value_typeB>
		void image_move(host_dense_matrix<__value_typeA,column_major>& dst, const host_dense_matrix<__value_typeB,column_major>& src, const unsigned int& image_width, const unsigned int& image_height, const unsigned int& num_maps, const char& xshift, const char& yshift){
			throw std::runtime_error("not implemented");
		}
		
	};
	template<class __matrix_typeA, class __matrix_typeB>
	void image_move(__matrix_typeA& dst, const __matrix_typeB& src, const unsigned int& image_width, const unsigned int& image_height, const unsigned int& num_maps, const int& xshift, const int& yshift){
		image_move_impl::image_move(dst,src,image_width,image_height,num_maps,(char)xshift,(char)yshift);
	}

#define INST(A,B) \
	template      \
	void image_move(dev_dense_matrix<A,column_major>&,const dev_dense_matrix<B,column_major>&, const unsigned int&, const unsigned int&, const unsigned int&, const int&, const int&); \
	template      \
	void image_move(host_dense_matrix<A,column_major>&,const host_dense_matrix<B,column_major>&, const unsigned int&, const unsigned int&, const unsigned int&, const int&, const int&); \

	INST(float,unsigned char);
	INST(unsigned char,unsigned char);
	
};


