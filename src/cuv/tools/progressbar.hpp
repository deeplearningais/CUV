#ifndef __PROGRESSBAR_HPP__
#define __PROGRESSBAR_HPP__
#include <iostream>
#include <string>

class ProgressBar_impl;

/**
 * @addtogroup tools
 * @{
 *
 * @class ProgressBar
 * A progressbar :-)
 */
class ProgressBar {
  private:
    ProgressBar_impl* m_pb; ///< implementation details are in here
  public:
    /**
     * Constructor.
     * @param i number of expected steps
     * @param desc what to show while working
     * @param cwidth of progress bar
     */
    ProgressBar(long int i=100, const std::string& desc = "Working" ,int cwidth=30);
    /**
     * increase value of progressbar
     * @param info show this text additionally
     * @param v    increase this much
     */
    void inc(const char*info,int v=1);
    /**
     * @overload
     *
     * increase value of progressbar
     * @param v    increase this much
     */
    void inc(int v=1);
    /**
     * call this when done.
     * @param clear clears line of progressbar
     */
    void finish(bool clear=false);
    /**
     * @overload
     *
     * call this when done.
     * @param s write this in the line of the progressbar
     */
    void finish(char* s);

    /**
     * display a string when updating
     *
     * @param info the string to be shown
     */
    void display(const char* info="");
};

/** @} */ // end group tools

#endif /* __PROGRESSBAR_HPP__ */
