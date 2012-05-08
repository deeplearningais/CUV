//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*

#ifndef __REFERENCE_HPP__
#define __REFERENCE_HPP__

#include<boost/type_traits/is_convertible.hpp>
#include<boost/utility/enable_if.hpp>
#include<cuv/tools/cuv_general.hpp>
#include<cuv/tools/meta_programming.hpp>
#include<cuv/basics/accessors.hpp>

namespace cuv
{
	/**
	 * This objects acts like a reference to the object stored at the wrapped pointer.
     * \ingroup data_structures
	 */
	template<class T, class M, class I>
	struct reference
	{
		typedef T*                        pointer_type; ///< the wrapped pointer type
		typedef typename unconst<T>::type value_type;   ///< the type of the pointer
		typedef M                memory_space_type;     ///< the memory space of the pointer
		typedef reference<T,M,I>            my_type;    ///< the type of this reference
		typedef I                        index_type;    ///< hmm this should be size_t eventually

		const pointer_type ptr;                         ///< the wrapped pointer

		/// convert to the stored value
		operator value_type ()const{
			return detail::entry_get(ptr,(index_type)0,memory_space_type());
		}

		/// assign a new value
		void operator=(const value_type& v){
            detail::entry_set(ptr,(index_type)0,v,memory_space_type());
		}

        /// assign a value of a different (but convertible) value type
        template<class _T>
		typename 
        boost::enable_if_c<boost::is_convertible<_T,value_type>::value >::type
            operator=(const _T& v){
                detail::entry_set(ptr,(index_type)0,(value_type)v,memory_space_type());
		}
	
		/// assignment from reference of same type
		reference& operator=(const reference& o)
		{
			if(&o == &(*this)) // operator & is overloaded and returns value_type*
				return *this;
			(*this) = (value_type)o;
			return *this;
		}

		/// assignment from reference of other memory type
		template<class O>
		reference& operator=(const reference<T,O,I>& o)
		{
			(*this) = (T)o;
			return *this;
		}

		
		/// get the wrapped pointer
		pointer_type operator&(void)const{
			return ptr;
		}

		/// construct using a pointer
		reference(const pointer_type& p)
			:ptr(p)
		{
		}

		/// implicit construction using value
		reference(T & p)
			:ptr(&p)
		{
		}

		/// implicit construction using value
		reference(const T & p)
			:ptr(&p)
		{
		}
		
		/// add to the value stored at ptr
		my_type& operator+=(const value_type& v){ *this = (value_type)(*this)+v; return *this; }
		/// subtract from the value stored at ptr
		my_type& operator-=(const value_type& v){ *this = (value_type)(*this)-v; return *this; }
		/// multiply with the value stored at ptr
		my_type& operator*=(const value_type& v){ *this = (value_type)(*this)*v; return *this; }
		/// divide by the value stored at ptr
		my_type& operator/=(const value_type& v){ *this = (value_type)(*this)/v; return *this; }
		/// increment value at ptr
		value_type operator++(int){value_type v=*this; *this=v+1; return v;}
		/// decrement value at ptr
		value_type operator--(int){value_type v=*this; *this=v-1; return v;}
		/// increment value at ptr
		value_type operator++(){value_type v=*this; *this=v+1; return v+1;}
		/// decrement value at ptr
		value_type operator--(){value_type v=*this; *this=v-1; return v-1;}

		/// compare value at ptr with another
		bool operator==(const value_type& v){ return ((value_type)*this)==v;}
		/// compare value at ptr with another
		bool operator<=(const value_type& v){ return ((value_type)*this)<=v;}
		/// compare value at ptr with another
		bool operator< (const value_type& v){ return ((value_type)*this)< v;}
		/// compare value at ptr with another
		bool operator>=(const value_type& v){ return ((value_type)*this)>=v;}
		/// compare value at ptr with another
		bool operator> (const value_type& v){ return ((value_type)*this)> v;}
	};
	
}


#endif /* __REFERENCE_HPP__ */
