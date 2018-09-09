
#include "HistogramRecorder.h"
#include <fstream>

namespace histogram
{
    Recorder::Recorder(Histogram& histogram):
    histogram(histogram)
    {
    }

    void Recorder::recordData(TIME_TYPE time, const std::vector<DATA_TYPE>& data)
    {
        recording.push_back(histogram.sort(data));
        times.push_back(time);
    }

    void Recorder::writeData(const std::string& outputFile)
    {
        std::ofstream file;
        file.open(outputFile, std::fstream::out);

        for(const auto& bin : histogram.getBins())
        {
            file << bin << ',';
        }
        file << '\n';

        for(auto i = 0; i < recording.size(); i++)
        {
            file << "t=" << times[i] << ':';
            for(const auto& point : recording[i])
            {
                file << point << ',';
            }
            file << '\n';
        }

        file.close();
    }

    void Recorder::clearData()
    {
        recording.clear();
        times.clear();
    }
}
