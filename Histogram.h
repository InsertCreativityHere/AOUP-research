
#ifndef HISTOGRAM_DEF
#define HISTOGRAM_DEF

#include <vector>

namespace histogram
{
    class Histogram
    {
        public:
            Histogram(unsigned long bins, double minimum, double maximum);
            virtual std::vector<unsigned long> sort(const std::vector<double>& data) = 0;
            virtual std::vector<double> getBins() = 0;

        protected:
            unsigned long binCount;
            double min;
            double max;
    };

    class LinearHistogram : public Histogram
    {
        public:
            LinearHistogram(unsigned long bins, double minimum, double maximum);
            std::vector<unsigned long> sort(const std::vector<double>& data);
            std::vector<double> getBins();

        private:
            double width;
    };

    class CustomHistogram : public Histogram
    {
        public:
            CustomHistogram(std::vector<double> bins, double minimum, double maximum);
            std::vector<unsigned long> sort(const std::vector<double>& data);
            std::vector<double> getBins();

        private:
            std::vector<double> binBounds;
    };
}
#endif
