
#include "Histogram.h"
#include <cmath>

namespace histogram
{
    Histogram::Histogram(unsigned long bins, double minimum, double maximum):
    binCount(bins + 2), min(minimum), max(maximum)
    {
    }

    LinearHistogram::LinearHistogram(double minimum, double maximum, double dx):
    Histogram(ceil((maximum - minimum) / dx), minimum, maximum), width(dx)
    {
    }
    //TODO & NOTE: The second the last (overflow included) bin won't be the same size as the others until (max - min)/dx is a whole number. Otherwise it's smaller than the rest.
    std::vector<unsigned long> LinearHistogram::sort(const std::vector<double>& data)
    {
        std::vector<unsigned long> values(binCount);

        for(const auto& point : data)
        {
            if(point <= min)
            {
                values.front()++;
            } else
            if(point >= max)
            {
                values.back()++;
            } else{
                values[(unsigned long)((point - min) / width) + 1]++;
            }
        }

        return values;
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

    CustomHistogram::CustomHistogram(std::vector<double> bins):
    Histogram(bins.size(), binBounds.front(), binBounds.back()), binBounds(bins)
    {
    }

    std::vector<unsigned long> CustomHistogram::sort(const std::vector<double>& data)
    {
        std::vector<unsigned long> bins(binCount);

        for(const auto& point : data)
        {
            if(point < min)
            {
                bins.front()++;
            } else
            if(point > max)
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
