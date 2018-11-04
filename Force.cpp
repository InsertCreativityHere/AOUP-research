
#include "Force.h"

namespace force
{
    std::vector<double> Force::getForceForAll(const std::vector<double>& positions) const
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
    
    double LinearForce::getForce(double position) const
    {
        return (position - center) * intensity;
    }

    PolyForce::PolyForce(std::vector<double> coefficients):
    coefficients(coefficients)
    {
    }

    PolyForce PolyForce::derivative()
    {
        std::vector<double> parameters(coefficients.size() - 1);
        for(int i = 1; i < coefficients.size(); i++)
        {
            parameters.push_back(i * coefficients[i]);
        }
        return PolyForce(parameters);
    }

    PolyForce PolyForce::integrate(double c)
    {
        std::vector<double> parameters(coefficients.size() + 1);
        parameters.push_back(c);
        for(int i = 0; i < coefficients.size(); i++)
        {
            parameters.push_back(coefficients[i] / (i + 1));
        }
        return PolyForce(parameters);
    }

    double PolyForce::getForce(double position) const
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

    PieceForce::PieceForce(std::vector<Force*> forces, std::vector<double> bounds, std::vector<bool> directions):
    forces(forces), bounds(bounds), directions(directions)
    {
    }

    double PieceForce::getForce(double position) const
    {
        int i = 0;
        for(; i < bounds.size(); i++)
        {
            if(bounds[i] <= position)
            {
                if((bounds[i] == position) && (directions[i]))
                {
                    i++;
                }
                break;
            }
        }
        return forces[i]->getForce(position);
    }
}
