
#ifndef HISTOGRAMRECORDER_DEF
#define HISTOGRAMRECORDER_DEF

#include <string>
#include <vector>
#include "Histogram.h"

namespace histogram
{
    class Recorder
    {
        public:
            Recorder(Histogram& histogram);
            void recordData(double time, const std::vector<double>& data);
            void writeData(const std::string& outputFile);
            void clearData();

        private:
            Histogram& histogram;
            std::vector<std::vector<unsigned long>> recording;
            std::vector<double> times;
    };
}
#endif
