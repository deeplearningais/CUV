#ifndef __PROGRESSBAR_HPP__
#define __PROGRESSBAR_HPP__
#include <iostream>
#include <string>

class ProgressBar_impl;

class ProgressBar {
  private:
    ProgressBar_impl* m_pb;
  public:
    ProgressBar(long int i=100, const std::string& desc = "Working" ,int cwidth=30);
    void inc(const char*info,int v=1);
    void inc(int v=1);
    void finish(bool clear=false);
    void finish(char* s);
    void display(const char* info="");
};

#endif /* __PROGRESSBAR_HPP__ */
