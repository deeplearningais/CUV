
/*
 * this is slightly messy.
 * the code in this header can be included more than once,
 * with different pre-defined macros.
 * Make sure these macros ARE set before including this file.
 *   BLOCK_SIZE -- how many values to process at once
 *   NUM_IMG    -- how many images to process at once
 *   ROW_FACT   -- steepness of the diagonal (1: 45-deg, 2: steep, 4: steeeeeep)
 */
/****************************************************************
 *   Device Code
 ****************************************************************/

#ifdef SPMM_NAME
#  undef SPMM_NAME
#endif
#define SPMM_NAME(X) X ## _ ## BLOCK_SIZE ## _ ## NUM_IMG ## _ ## ROW_FACT


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
 const value_type factC)
{
	__shared__ int        offsets[BLOCK_SIZE];
	__shared__ value_type    sums[BLOCK_SIZE * NUM_IMG];
	const index_type thread_id = large_grid_thread_id();
	const index_type grid_size = large_grid_thread_num();

	// load diagonal offsets into shared memory
	if(threadIdx.x < A_nd)
		offsets[threadIdx.x] = A_diaoff[threadIdx.x];
	__syncthreads();

	for(index_type col = thread_id; col < A_w; col += grid_size)
	{
		for(value_type* s_ptr=sums+threadIdx.x; s_ptr<sums+BLOCK_SIZE*NUM_IMG;  s_ptr += BLOCK_SIZE)
			*s_ptr = (value_type)0 ;
		for(index_type n = 0, offset=0; n < A_nd; n++, offset+=A_stride)
		{
			const int row = (col - offsets[n])*[%rf%];
			/*const int row = col - A_diaoff[n];*/
			if(row >= 0 && row < A_h)
			{
				value_type A_ij;
				const value_type* v_ptr;
				
				[% FOREACH lrf IN rfs %]
					A_ij  = A_data[       offset + row + [%lrf%] ];
					v_ptr = v + row + [% lrf %];
					[% FOREACH img IN nimgs  %]
						sums[BLOCK_SIZE * [% img %] + threadIdx.x] += A_ij * *v_ptr;
						v_ptr += A_h;
					[% END %]
				[% END %]
			}
		}
		[% FOREACH img IN nimgs %]
			dst[col + [%img%]*A_w] = (wantFactC  ? factC * dst[col + [%img%] * A_w] : 0.f) 
				+                    (wantFactAv ? factAv                           : 1.f) * sums[BLOCK_SIZE*[%img%] + threadIdx.x];
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
	 const value_type factC)
{
	__shared__ int        offsets[BLOCK_SIZE];
	__shared__ value_type    sums[BLOCK_SIZE * NUM_IMG];

	const index_type thread_id = large_grid_thread_id();
	const index_type grid_size = large_grid_thread_num();

	// load diagonal offsets into shared memory
	if(threadIdx.x < A_nd)
		offsets[threadIdx.x] = A_diaoff[threadIdx.x];
	__syncthreads();

	for(index_type row = thread_id; row < A_h; row += grid_size)
	{
		// initialize shared memory
		[% FOREACH img IN nimgs %]
		    sums[BLOCK_SIZE*[%img%] + threadIdx.x] = (value_type) 0 ;
		[% END %]
		index_type offset = row;
		for(index_type n = 0; n < A_nd; n++, offset+=A_stride)
		{
			const int col = row/[%rf%] + offsets[n];
			if(col >= 0 && col < A_w)
			{
				const value_type   A_ij = A_data[       offset];
				const value_type* v_ptr = v+col;
				[% FOREACH img IN nimgs %]
					sums[BLOCK_SIZE* [%img%] + threadIdx.x] += A_ij * *v_ptr; v_ptr += A_w;
				[% END %]
			}
		}
		[% FOREACH img IN nimgs %]
			dst[row + [%img%]*A_h] = (wantFactC  ? factC * dst[row + [%img%] * A_h] : 0.f)
				+                    (wantFactAv ? factAv                           : 1.f) * sums[BLOCK_SIZE*[%img%] + threadIdx.x];
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
		const value_type factC)
{
	__shared__ int        offsets[BLOCK_SIZE];
	value_type            sums[NUM_IMG];

	const index_type thread_id = large_grid_thread_id();
	const index_type grid_size = large_grid_thread_num();

	// load diagonal offsets into shared memory
	if(threadIdx.x < A_nd)
		offsets[threadIdx.x] = A_diaoff[threadIdx.x];
	__syncthreads();

	for(index_type col = thread_id; col < A_w; col += grid_size)
	{
		for(unsigned int i=0;i<NUM_IMG;i++)
			sums[i] = (value_type)0 ;
		index_type offset = 0;
		for(index_type n = 0; n < A_nd; n++, offset+=A_stride)
		{
			const int row = (col - offsets[n])*[%rf%];
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
		const value_type factC)
{
	__shared__ int        offsets[BLOCK_SIZE];
	value_type            sums[NUM_IMG];

	const index_type thread_id = large_grid_thread_id();
	const index_type grid_size = large_grid_thread_num();

	// load diagonal offsets into shared memory
	if(threadIdx.x < A_nd)
		offsets[threadIdx.x] = A_diaoff[threadIdx.x];

	for(index_type row = thread_id; row < A_h; row += grid_size)
	{
		// initialize shared memory
		for(unsigned int i=0;i<NUM_IMG;i++)
			sums[i] = (value_type) 0 ;
		__syncthreads();
		index_type offset = row;
		for(index_type n = 0; n < A_nd; n++, offset+=A_stride)
		{
			const int col = row/[%rf%] + offsets[n];
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
