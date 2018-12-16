
#ifndef FORCE_DEF
#define FORCE_DEF

#include <vector>

namespace force
{
    class Force
    {
        public:
            virtual double getForce(double position) const = 0;
            virtual Force* negate() const = 0;
            virtual Force* derivative() const = 0;
            virtual Force* integrateI(double c) const = 0;
            virtual double integrateD(double a, double b) const = 0;
            std::vector<double> getForceForAll(const std::vector<double>& positions) const;
    };

    class PolyForce : public virtual Force
    {
        public:
            PolyForce(std::vector<double> coefficients);
            double getForce(double position) const;
            PolyForce* negate() const;
            PolyForce* derivative() const;
            PolyForce* integrateI(double c) const;
            double integrateD(double a, double b) const;

        private:
            const std::vector<double> coefficients;
    };

    class PieceForce : public virtual Force
    {
        public:
            PieceForce(std::vector<Force*> forces, std::vector<double> bounds, std::vector<bool> directions);
            double getForce(double position) const;
            PieceForce* negate() const;
            PieceForce* derivative() const;
            PieceForce* integrateI(double c) const;
            double integrateD(double a, double b) const;

        private:
            const std::vector<Force*> forces;
            const  std::vector<double> bounds;
            const std::vector<bool> directions;
    };
}
#endif
