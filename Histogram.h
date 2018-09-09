#ifndef HISTOGRAM_DEF
#define HISTOGRAM_DEF

#include <vector>

#define BINH_TYPE unsigned long
#define BINW_TYPE unsigned short
#define DATA_TYPE double
#define DATA_LENGTH unsigned long

namespace histogram
{
    class Histogram
    {
        public:
            Histogram(BINW_TYPE bins, double minimum, double maximum);
            virtual std::vector<BINH_TYPE> sort(const std::vector<DATA_TYPE> &data) = 0;
            virtual std::vector<DATA_TYPE> getBins();

        protected:
            BINW_TYPE binCount;
            double min;
            double max;
    };

    class LinearHistogram : public Histogram
    {
        public:
            LinearHistogram(BINW_TYPE bins, double minimum, double maximum);
            std::vector<BINH_TYPE> sort(const std::vector<DATA_TYPE> &data);
            std::vector<DATA_TYPE> getBins();

        private:
            double width;
    };

    class CustomHistogram : public Histogram
    {
        public:
            CustomHistogram(std::vector<DATA_TYPE> bins, double minimum, double maximum);
            std::vector<BINH_TYPE> sort(const std::vector<DATA_TYPE> &data);
            std::vector<DATA_TYPE> getBins();

        private:
            std::vector<DATA_TYPE> binBounds;
    };
}
#endif