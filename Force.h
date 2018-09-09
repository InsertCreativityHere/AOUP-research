
#ifndef FORCE_DEF
#define FORCE_DEF

#include <vector>

namespace force
{
    class Force
    {
        public:
            virtual double getForce(double position) = 0;
            std::vector<double> getForceForAll(const std::vector<double>& positions);
    };

    class LinearForce : public Force
    {
        public:
            LinearForce(double center, double slope);
            double getForce(double position);

        private:
            const double center;
            const double intensity;
    };

    class PolyForce : public Force
    {
        public:
            PolyForce(std::vector<double> coefficients);
            double getForce(double position);

        private:
            const std::vector<double> coefficients;
    };
}
#endif
