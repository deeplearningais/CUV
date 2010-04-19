//*LB*
// Copyright (c) 2010, Hannes Schulz, Andreas Mueller, Dominik Scherer
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




#ifndef __AIS_TIMING_HPP__
#define __AIS_TIMING_HPP__

#include <boost/date_time/posix_time/posix_time_types.hpp>

/**
 * \brief Simple class to perfom timing measurements.
 *
 * The class allows to compute timing information with microsecond (10^-6s)
 * resolution. If required the performance can also be tracked by this class
 * by calling the perf() method instead of the diff() method.
 * If the counter and timing values should be reset the reset() method needs
 * to be called explicitely.
 */
class Timing
{
    public:
        inline Timing();

        inline double  diff() const;
        inline double  perf() const;
        inline int     count() const;
        inline void    update(int inc=1);
        inline void    reset();

    private:
        boost::posix_time::ptime    m_start;
        boost::posix_time::ptime    m_end;
        int                         m_count;

};

/**
 * \brief Creates a new Timing instance.
 *
 * The start and endtime are initialized to the current time.
 */
Timing::Timing()
    :   m_start(boost::posix_time::microsec_clock::universal_time())
      , m_end(boost::posix_time::microsec_clock::universal_time())
      , m_count(0)
{}

/**
 * \brief Returns the time difference between start and end.
 *
 * \return time elapsed between start and end as a double in seconds
 */
inline double Timing::diff() const
{
    return (m_end - m_start).total_microseconds() / (double)1e6;
}

/**
 * \brief Updates the end time of the Timing instance and increments the
 *        counter.
 *
 * \param inc amount to increment the counter
 */
inline void Timing::update(int inc)
{
    m_end = boost::posix_time::microsec_clock::universal_time();
    m_count += inc;
}

/**
 * \brief Computes the performance measured by the Timing instance.
 *
 * The performance is defined as follows
 *   m_counter / diff()
 *
 * \return performance measured by the Timing instance
 */
inline double Timing::perf() const
{
    return diff() / m_count ;
}

/**
 * \brief Initializes the Timing instance.
 *
 * The initialization sets start and end time to the current time.
 */
void Timing::reset()
{
    m_start = boost::posix_time::microsec_clock::universal_time();
    m_end = boost::posix_time::microsec_clock::universal_time();
    m_count = 0;
}

/**
 * \brief Returns the current value of the counter.
 *
 * \return current count value
 */
inline int Timing::count() const
{
    return m_count;
}

#endif /* __AIS_TIMING_HPP__ */
