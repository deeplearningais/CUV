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






#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/convert/convert.hpp>

using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;

void export_0ary_functors(){
    enum_<cuv::NullaryFunctor>("nullary_functor")
        .value("FILL", NF_FILL)
        .value("SEQ", NF_SEQ);

}
void export_scalar_functors() {
    enum_<cuv::ScalarFunctor>("scalar_functor")
        //.value("EXACT_EXP", SF_EXACT_EXP)
        .value("COPY", SF_COPY)
        .value("EXP", SF_EXP)
        .value("LOG", SF_LOG)
        .value("SIGN", SF_SIGN)
        .value("SIGM", SF_SIGM)
        //.value("EXACT_SIGM", SF_EXACT_SIGM)
        .value("DSIGM", SF_DSIGM)
        .value("TANH", SF_TANH)
        .value("DTANH", SF_DTANH)
        .value("SQUARE", SF_SQUARE)
        .value("SUBLIN", SF_SUBLIN)
        .value("ENERG", SF_ENERG)
        .value("INV", SF_INV)
        .value("SQRT", SF_SQRT)
        .value("NEGATE", SF_NEGATE)
        .value("ABS", SF_ABS)
        .value("SMAX", SF_SMAX)
        .value("POSLIN", SF_POSLIN)
        .value("RECT", SF_RECT)
        .value("DRECT", SF_DRECT)

        .value("ADD", SF_ADD)
        .value("SUBTRACT", SF_SUBTRACT)
        .value("MULT", SF_MULT)
        .value("DIV", SF_DIV)
        .value("MIN", SF_MIN)
        .value("MAX", SF_MAX)
        ;

}

void export_binary_functors(){
    enum_<cuv::BinaryFunctor>("binary_functor")
        .value("OR", BF_OR)
        .value("AND", BF_AND)
        .value("ADD", BF_ADD)
        .value("SUBTRACT", BF_SUBTRACT)
        .value("MULT", BF_MULT)
        .value("DIV", BF_DIV)
        .value("MIN", BF_MIN)
        .value("MAX", BF_MAX)

        .value("AXPY", BF_AXPY)
        .value("XPBY", BF_XPBY)
        .value("AXPBY", BF_AXPBY)
        ;
}

void export_tensor_ops(){
	export_scalar_functors();
	export_binary_functors();
	export_0ary_functors();
}


