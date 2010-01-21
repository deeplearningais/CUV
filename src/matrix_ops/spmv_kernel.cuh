


/*
 * this is slightly messy.
 * the code in this header can be included more than once,
 * with different pre-defined macros.
 * Make sure these macros ARE set before including this file.
 *   BLOCK_SIZE -- how many values to process at once
 *   NUM_IMG    -- how many images to process at once
 *   ROW_FACT   -- steepness of the diagonal (1: 45-deg, 2: steep, 4: steeeeeep)
 */

// set this to one if you have less than BLOCK_SIZE diagonals to deal with. 
// Makes program faster.
#define BLOCK_SIZE_LIMITS_NUM_DIAG 1

/****************************************************************
 *   Device Code
 ****************************************************************/

template <typename value_type, typename index_type, bool UseCache, bool wantFactAv, bool wantFactC>
	__global__ void
spmm_dia_kernel_trans_shared_[%bs%]_[%ni%]_[%rf%]
(
 const index_type A_h, 
 const index_type A_w, 
 const index_type A_nd,
 const index_type A_stride,
 const int        * A_diaoff,
 const value_type * A_data,
 const value_type * v, 
 value_type       * dst,
 const value_type factAv,
 const value_type factC,
 const unsigned int  toff)
{
#if BLOCK_SIZE_LIMITS_NUM_DIAG
	__shared__ int        offsets[[%bs%]];
#endif
	__shared__ value_type    sums[[%bs%] * [%ni%]];
	const index_type thread_id = large_grid_thread_id();
	const index_type grid_size = large_grid_thread_num();

	// load diagonal offsets into shared memory
#if BLOCK_SIZE_LIMITS_NUM_DIAG
	if(threadIdx.x < A_nd)
	   offsets[threadIdx.x] = A_diaoff[threadIdx.x];
	__syncthreads();
#endif

	for(index_type col = thread_id; col < A_w; col += grid_size)
	{
		for(unsigned int i=threadIdx.x;i<[%bs%]*[%ni%];  i += [%bs%])
			sums[i] = 0;

		for(index_type n = 0, offset=0; n < A_nd; n++, offset+=A_stride)
		{
#if BLOCK_SIZE_LIMITS_NUM_DIAG
			int row = (col - offsets[n])*[%rf%];
#else
			int row = (col - A_diaoff[n])*[%rf%];
#endif
			if(row >= 0 && row < A_h)
			{
				value_type A_ij;
				const value_type* v_ptr = v+row;
				
				[% FOREACH lrf IN rfs %]
					A_ij  = A_data[       offset + row];
					[% FOREACH img IN nimgs  %]
						sums[[%bs%] * [% img %] + threadIdx.x] += A_ij * fetch_x<UseCache>(v_ptr,toff+[%lrf%] + A_h*[%img%]);
					[% END %]
					row++;
				[% END %]
			}
		}
		[% FOREACH img IN nimgs %]
			dst[col + [%img%]*A_w] = (wantFactC  ? factC * dst[col + [%img%] * A_w] : 0.f) 
				+                    (wantFactAv ? factAv                           : 1.f) * sums[[%bs%]*[%img%] + threadIdx.x];
		[% END %]
	}
}
template <typename value_type, typename index_type, bool UseCache, bool wantFactAv, bool wantFactC>
__global__ void
spmm_dia_kernel_shared_[%bs%]_[%ni%]_[%rf%]
	(
	 const index_type A_h, 
	 const index_type A_w, 
	 const index_type A_nd,
	 const index_type A_stride,
	 const int        * A_diaoff,
	 const value_type * A_data,
	 const value_type * v, 
	 value_type       * dst,
	 const value_type factAv,
	 const value_type factC,
	 const unsigned int  toff)
{
#if BLOCK_SIZE_LIMITS_NUM_DIAG
	__shared__ int        offsets[[%bs%]];
#endif
	__shared__ value_type    sums[[%bs%] * [%ni%]];

	const index_type thread_id = large_grid_thread_id();
	const index_type grid_size = large_grid_thread_num();

#if BLOCK_SIZE_LIMITS_NUM_DIAG
	// load diagonal offsets into shared memory
	if(threadIdx.x < A_nd)
		offsets[threadIdx.x] = A_diaoff[threadIdx.x];
	__syncthreads();
#endif

	for(index_type row = thread_id; row < A_h; row += grid_size)
	{
		// initialize shared memory
		[% FOREACH img IN nimgs %]
		    sums[[%bs%]*[%img%] + threadIdx.x] = (value_type) 0 ;
		[% END %]
		index_type offset = row;
		for(index_type n = 0; n < A_nd; n++, offset+=A_stride)
		{
#if BLOCK_SIZE_LIMITS_NUM_DIAG
			const int col = row/[%rf%] + offsets[n];
#else
			const int col = row/[%rf%] + A_diaoff[n];
#endif
			if(col >= 0 && col < A_w)
			{
				const value_type   A_ij = A_data[       offset];
				const value_type* v_ptr = v+col;
				[% FOREACH img IN nimgs %]
					/*sums[[%bs%]* [%img%] + threadIdx.x] += A_ij * *v_ptr; v_ptr += A_w;*/
					sums[[%bs%]* [%img%] + threadIdx.x] += A_ij * fetch_x<UseCache>(v_ptr, [%img%]*A_w);
				[% END %]
			}
		}
		[% FOREACH img IN nimgs %]
			dst[row + [%img%]*A_h] = (wantFactC  ? factC * dst[row + [%img%] * A_h] : 0.f)
				+                    (wantFactAv ? factAv                           : 1.f) * sums[[%bs%]*[%img%] + threadIdx.x];
		[% END %]
	}
}


/*
 * Registers are a bit special, they (empirically) make only sense for NUM_IMG = 1
 */
[% IF ni == 1 %]
template <typename value_type, typename index_type, bool UseCache, bool wantFactAv, bool wantFactC>
	__global__ void
spmm_dia_kernel_trans_register_[%bs%]_[%ni%]_[%rf%]
(
		const index_type A_h, 
		const index_type A_w, 
		const index_type A_nd,
		const index_type A_stride,
		const int        * A_diaoff,
		const value_type * A_data,
		const value_type * v, 
		value_type       * dst,
		const value_type factAv,
		const value_type factC,
		const unsigned int toff
		)
{
#if BLOCK_SIZE_LIMITS_NUM_DIAG
	__shared__ int        offsets[[%bs%]];
#endif
	value_type            sums[[%ni%]];

	const index_type thread_id = large_grid_thread_id();
	const index_type grid_size = large_grid_thread_num();

	// load diagonal offsets into shared memory
#if BLOCK_SIZE_LIMITS_NUM_DIAG
	if(threadIdx.x < A_nd)
		offsets[threadIdx.x] = A_diaoff[threadIdx.x];
	__syncthreads();
#endif

	for(index_type col = thread_id; col < A_w; col += grid_size)
	{
		for(unsigned int i=0;i<[%ni%];i++)
			sums[i] = (value_type)0 ;
		index_type offset = 0;
		for(index_type n = 0; n < A_nd; n++, offset+=A_stride)
		{
#if BLOCK_SIZE_LIMITS_NUM_DIAG
			const int row = (col - offsets[n])*[%rf%];
#else
			const int row = (col - A_diaoff[n])*[%rf%];
#endif
			if(row >= 0 && row < A_h)
			{
				value_type        A_ij;
				const value_type* v_ptr ;
				[% FOREACH lrf IN rfs %]
					A_ij  = A_data[       offset + row+[% lrf %]];
					v_ptr = v + row + [% lrf %];
					[% FOREACH img IN nimgs %]
						sums[ [% img %] ] += A_ij * *v_ptr; v_ptr += A_h; [% END %]
				[% END %]
			}
		}
		__syncthreads();
		[% FOREACH img IN nimgs %]
			dst[col + [% img %] *A_w] = (wantFactC  ? factC * dst[col + [% img %]  * A_w] : 0.f) 
				+              (wantFactAv ? factAv                     : 1.f) * sums[[% img %] ];
		[% END %]
	}
}
template <typename value_type, typename index_type, bool UseCache, bool wantFactAv, bool wantFactC>
__global__ void
spmm_dia_kernel_register_[%bs%]_[%ni%]_[%rf%]
(
		const index_type A_h, 
		const index_type A_w, 
		const index_type A_nd,
		const index_type A_stride,
		const int        * A_diaoff,
		const value_type * A_data,
		const value_type * v, 
		value_type       * dst,
		const value_type factAv,
		const value_type factC,
		const unsigned int toff)
{
#if BLOCK_SIZE_LIMITS_NUM_DIAG
	__shared__ int        offsets[[%bs%]];
#endif
	value_type            sums[[%ni%]];

	const index_type thread_id = large_grid_thread_id();
	const index_type grid_size = large_grid_thread_num();

#if BLOCK_SIZE_LIMITS_NUM_DIAG
	// load diagonal offsets into shared memory
	if(threadIdx.x < A_nd)
		offsets[threadIdx.x] = A_diaoff[threadIdx.x];
#endif

	for(index_type row = thread_id; row < A_h; row += grid_size)
	{
		// initialize shared memory
		for(unsigned int i=0;i<[%ni%];i++)
			sums[i] = (value_type) 0 ;
		__syncthreads();
		index_type offset = row;
		for(index_type n = 0; n < A_nd; n++, offset+=A_stride)
		{
#if BLOCK_SIZE_LIMITS_NUM_DIAG
			const int col = row/[%rf%] + offsets[n];
#else
			const int col = row/[%rf%] + A_diaoff[n];
#endif
			if(col >= 0 && col < A_w)
			{
				const value_type A_ij = A_data[       offset];
				const value_type* v_ptr = v+col;
				[% FOREACH img IN nimgs %]
					sums[ [% img %] ] += A_ij * *v_ptr; v_ptr += A_h;
				[% END %]
			}
		}
		__syncthreads();
		[% FOREACH img IN nimgs %]
			dst[row + [%img%]*A_h] = (wantFactC  ? factC * dst[row + [%img%] * A_h] : 0.f) 
				+                    (wantFactAv ? factAv                     : 1.f) * sums[[%img%]];
		[% END %]
	}
}
[% END %]
