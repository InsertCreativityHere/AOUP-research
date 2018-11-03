
#ifndef FORCE_DEF
#define FORCE_DEF

#include <vector>

namespace force
{
    class Force
    {
        public:
            virtual double getForce(double position) const = 0;
            std::vector<double> getForceForAll(const std::vector<double>& positions) const;
    };

    class LinearForce : public virtual Force
    {
        public:
            LinearForce(double center, double slope);
            double getForce(double position) const;

        private:
            const double center;
            const double intensity;
    };

    class PolyForce : public virtual Force
    {
        public:
            PolyForce(std::vector<double> coefficients);
            double getForce(double position) const;

        private:
            const std::vector<double> coefficients;
    };

    class PieceForce : public virtual Force
    {
        public:
            PieceForce(std::vector<Force*> forces, std::vector<double> bounds, std::vector<bool> directions);
            double getForce(double position) const;

        private:
            const std::vector<Force*> forces;
            const  std::vector<double> bounds;
            const std::vector<bool> directions;
    };
}
#endif
