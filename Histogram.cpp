
#include "Histogram.h"

namespace histogram
{
    Histogram::Histogram(unsigned long bins, double minimum, double maximum):
    binCount(bins + 2), min(minimum), max(maximum)
    {
    }

    LinearHistogram::LinearHistogram(unsigned long bins, double minimum, double maximum):
    Histogram(bins, minimum, maximum), width((maximum - minimum) / bins)
    {
    }

    std::vector<unsigned long> LinearHistogram::sort(const std::vector<double>& data)
    {
        std::vector<unsigned long> bins(binCount);

        for(const auto& point : data)
        {
            if(point <= min)
            {
                bins.front()++;
            } else
            if(point >= max)
            {
                bins.back()++;
            } else{
                bins[(unsigned long)((point - min) / width) + 1]++;
            }
        }

        return bins;
    }

    std::vector<double> LinearHistogram::getBins()
    {
        std::vector<double> bins(binCount - 1);
        for(unsigned long i = 0; i < (binCount - 2); i++)
        {
            bins[i] = min + (width * i);
        }
        bins.back() = max;

        return bins;
    }

    CustomHistogram::CustomHistogram(std::vector<double> bins, double minimum, double maximum):
    Histogram(bins.size(), minimum, maximum), binBounds(bins)
    {
    }

    std::vector<unsigned long> CustomHistogram::sort(const std::vector<double>& data)
    {
        std::vector<unsigned long> bins(binCount);

        for(const auto& point : data)
        {
            if(point < binBounds.front())
            {
                bins.front()++;
            } else
            if(point > binBounds.back())
            {
                bins.back()++;
            } else{
                for(auto j = 1; j < binBounds.size(); j++)
                {
                    if(point < bins[j])
                    {
                        bins[j]++;
                        break;
                    }
                }
            }
        }

        return bins;
    }

    std::vector<double> CustomHistogram::getBins()
    {
        return binBounds;
    }
}
