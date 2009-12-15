#ifndef __EXCEPTION_HELPER__
#define __EXCEPTION_HELPER__


#include <execinfo.h>
#include <signal.h>

#include <exception>
#include <iostream>

/////////////////////////////////////////////

class ExceptionTracer
{
	public:
		ExceptionTracer()
		{
			void * array[25];
			int nSize = backtrace(array, 25);
			char ** symbols = backtrace_symbols(array, nSize);
			std::cout << "ExceptionTracer():"<<std::endl;

			for (int i = 0; i < nSize; i++)
			{
				std::cout << symbols[i] << std::endl;
			}

			free(symbols);
		}
};


#endif /* __EXCEPTION_HELPER__ */
