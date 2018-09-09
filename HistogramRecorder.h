#ifndef HISTOGRAMRECORDER_DEF
#define HISTOGRAMRECORDER_DEF

#include <string>
#include <vector>
#include "Histogram.h"

#define TIME_TYPE double

namespace histogram
{
    class Recorder
    {
        public:
            Recorder(Histogram *histogram);
            void recordData(TIME_TYPE time, const std::vector<DATA_TYPE> &data);
            void writeData(const std::string &outputFile);
            void clearData();

        private:
            Histogram *histogram;
            std::vector<std::vector<BINH_TYPE>> recording;
            std::vector<TIME_TYPE> times;
    };
}
#endif