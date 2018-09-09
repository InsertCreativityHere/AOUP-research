
#include "Histogram.h"

namespace histogram
{
    Histogram::Histogram(BINW_TYPE bins, double minimum, double maximum):
    binCount(bins + 2), min(minimum), max(maximum)
    {
    }

    LinearHistogram::LinearHistogram(BINW_TYPE bins, double minimum, double maximum):
    Histogram(bins, minimum, maximum), width((maximum - minimum) / bins)
    {
    }

    std::vector<BINH_TYPE> LinearHistogram::sort(const std::vector<DATA_TYPE>& data)
    {
        std::vector<BINH_TYPE> bins(binCount);

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
                bins[(BINW_TYPE)((point - min) / width) + 1]++;
            }
        }

        return bins;
    }

    std::vector<DATA_TYPE> LinearHistogram::getBins()
    {
        std::vector<DATA_TYPE> bins(binCount - 1);
        for(BINW_TYPE i = 0; i < (binCount - 2); i++)
        {
            bins[i] = min + (width * i);
        }
        bins.back() = max;

        return bins;
    }

    CustomHistogram::CustomHistogram(std::vector<DATA_TYPE> bins, double minimum, double maximum):
    Histogram(bins.size(), minimum, maximum), binBounds(bins)
    {
    }

    std::vector<BINH_TYPE> CustomHistogram::sort(const std::vector<DATA_TYPE>& data)
    {
        std::vector<BINH_TYPE> bins(binCount);

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
                for(BINW_TYPE j = 1; j < binBounds.size(); j++)
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

    std::vector<DATA_TYPE> CustomHistogram::getBins()
    {
        return binBounds;
    }
}
