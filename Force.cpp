
#include "Force.h"

namespace force
{
    std::vector<double> Force::getForceForAll(const std::vector<double>& positions)
    {
        auto forces = positions;
        for(auto& convert : forces)
        {
            convert = getForce(convert);
        }

        return forces;
    }

    LinearForce::LinearForce(double center, double slope):
    center(center), intensity(slope)
    {
    }
    
    double LinearForce::getForce(double position)
    {
        return (position - center) * intensity;
    }

    PolyForce::PolyForce(std::vector<double> coefficients):
    coefficients(coefficients)
    {
    }

    double PolyForce::getForce(double position)
    {
        double totalForce = 0;
        double temp = 1;
        for(const auto& coeffecient : coefficients)
        {
            totalForce += temp * coeffecient;
            temp *= position;
        }

        return totalForce;
    }
}
