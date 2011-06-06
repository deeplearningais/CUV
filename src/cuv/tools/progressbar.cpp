#include <boost/date_time/posix_time/posix_time.hpp>
#include "progressbar.hpp"

struct ProgressBar_impl {
	public:
		ProgressBar_impl(long int i=100, const std::string& desc = "Working" ,int cwidth=30)
			:ivpDesc(desc),
			ivMax(i),
			ivCurrent(0),
			ivClearLen(0),
			ivCWidth(cwidth),
			ivCPos(-1) {
				ivStartTime = boost::posix_time::second_clock::local_time();
				ivLastUpdate = ivStartTime;
				display();
			}
		inline void inc(const char*info,int v=1){
			ivCurrent += v;
			display(info);
		}
		inline void inc(int v=1){
			ivCurrent += v;
			display();
		}
		inline void finish(bool clear=false){
			//if(clear){
			std::cout << "\r";
			for(uint i=0;i<std::min((unsigned int)79,(unsigned int)(ivCWidth + ivClearLen + 60 + ivpDesc.size())); i++)
				std::cout << " ";
			std::cout<<"\r"<<std::flush;
			//}
			if(!clear){
				boost::posix_time::time_duration td=boost::posix_time::time_period(ivStartTime, boost::posix_time::second_clock::local_time()).length();
				std::cout <<ivpDesc<<": "<< td << std::endl;
			}
		}
		inline void finish(char* s){
			std::cout << s << std::endl<<std::flush;
		}
		std::string ivpDesc;
		long int ivMax;
		long int ivCurrent;
		uint ivClearLen;
		boost::posix_time::ptime ivStartTime, ivLastEndTime, ivLastUpdate;

		int ivCWidth;
		int ivCPos;

		void display(const char* info=""){
			double newpos_f = ((double)ivCurrent / (double) ivMax);
			int    newpos   = (int)(ivCWidth * newpos_f);

			boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
			boost::posix_time::time_duration update_td=boost::posix_time::time_period(ivLastUpdate, now).length();
			boost::posix_time::time_duration td=boost::posix_time::time_period(ivStartTime, now).length();
			int newsec = (1.0f-newpos_f)/newpos_f * td.total_seconds();
			td = boost::posix_time::seconds(newsec);
			boost::posix_time::ptime done(boost::posix_time::second_clock::local_time()+td);

			if(update_td.seconds()<2 ||(newpos == ivCPos && done==ivLastEndTime)) return;
			ivLastEndTime = done;
			ivLastUpdate  = now;

			ivCPos = newpos;

			std::cout << "\r"<<ivpDesc<<" |";
			for(int i=0;i<newpos;i++)
				std::cout << "#";
			for(int i=0;i<ivCWidth-newpos;i++)
				std::cout << " ";
			std::cout << "| [" << (int)(100.0*newpos_f) << "% (";
			std::cout << done <<")] ";
			std::cout << info;
			std::cout << std::flush;
			ivClearLen = ivClearLen>strlen(info)?ivClearLen:strlen(info);
		}
};

ProgressBar::ProgressBar(long int i, const std::string& desc,int cwidth)
{
	m_pb = new ProgressBar_impl(i,desc,cwidth);
}
void ProgressBar::inc(const char*info,int v){
	m_pb->inc(info,v);
}
void ProgressBar::inc(int v){
	m_pb->inc(v);
}
void ProgressBar::finish(bool clear){
	m_pb->finish(clear);
}
void ProgressBar::finish(char* s){
	m_pb->finish(s);
}
void ProgressBar::display(const char* info){
	m_pb->display(info);
}
