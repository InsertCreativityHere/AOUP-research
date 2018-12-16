
#include <string>
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

    PolyForce::PolyForce(std::vector<double> coefficients):
    coefficients(coefficients)
    {
    }

    double PolyForce::getForce(double position) const
    {
        double totalForce = 0;
        double temp = 1;
        for(const auto& coefficient : coefficients)
        {
            totalForce += temp * coefficient;
            temp *= position;
        }

        return totalForce;
    }

    PolyForce* PolyForce::negate() const
    {
        std::vector<double> parameters(coefficients.size());
        for(const auto& coefficient : coefficients)
        {
            parameters.push_back(-coefficient);
        }
        return new PolyForce(parameters);
    }

    PolyForce* PolyForce::derivative() const
    {
        std::vector<double> parameters(coefficients.size() - 1);
        for(int i = 1; i < coefficients.size(); i++)
        {
            parameters.push_back(i * coefficients[i]);
        }
        return new PolyForce(parameters);
    }

    PolyForce* PolyForce::integrateI(double c) const
    {
        std::vector<double> parameters(coefficients.size() + 1);
        parameters.push_back(c);
        for(int i = 0; i < coefficients.size(); i++)
        {
            parameters.push_back(coefficients[i] / (i + 1));
        }
        return (new PolyForce(parameters));
    }

    double PolyForce::integrateD(double a, double b) const
    {
        double value = 0;
        double tempA = a;
        double tempB = b;
        for(int i = 0; i < coefficients.size(); i++)
        {
            value += (tempB - tempA) * (coefficients[i] / (i + 1));
            tempA *= a;
            tempB *= b;
        }
        return value;
    }

    PieceForce::PieceForce(std::vector<Force*> forces, std::vector<double> bounds, std::vector<bool> directions):
    forces(forces), bounds(bounds), directions(directions)
    {
        if(forces.size() != (bounds.size() + 1))
        {
            throw std::invalid_argument("Incorrect number of piecewise forces provided. Expected " + std::to_string(bounds.size() + 1) + " recieved " + std::to_string(forces.size()) + ".");
        }
    }

    double PieceForce::getForce(double position) const
    {
        int i = 0;
        for(; i < bounds.size(); i++)
        {
            if(bounds[i] >= position)
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

    PieceForce* PieceForce::negate() const
    {
        std::vector<Force*> newForces(forces.size());
        for(const Force* force : forces)
        {
            newForces.push_back(force->negate());
        }
        std::vector<double> newBounds = bounds;
        std::vector<bool> newDirections = directions;
        return new PieceForce(newForces, newBounds, newDirections);
    }

    PieceForce* PieceForce::derivative() const
    {
        std::vector<Force*> newForces(forces.size());
        for(const Force* force : forces)
        {
            newForces.push_back(force->derivative());
        }
        std::vector<double> newBounds = bounds;
        std::vector<bool> newDirections = directions;
        return new PieceForce(newForces, newBounds, newDirections);
    }

    PieceForce* PieceForce::integrateI(double c) const
    {
        std::vector<Force*> newForces(forces.size());
        for(const Force* force : forces)
        {
            newForces.push_back(force->integrateI(c));
        }
        std::vector<double> newBounds = bounds;
        std::vector<bool> newDirections = directions;
        return new PieceForce(newForces, newBounds, newDirections);
    }

    double PieceForce::integrateD(double a, double b) const
    {
        if(b < a)
        {
            return -integrateD(b, a);
        }
        if(b == a)
        {
            return 0;
        }

        double value = 0;
        int i = 0;
        for(; i < bounds.size(); i++)
        {
            double bound = bounds[i];
            if(bound > a)
            {
                if(bound < b)
                {
                    return forces[i]->integrateD(a, b);
                } else{
                    value += forces[i]->integrateD(a, bound);
                }
                break;
            }
        }

        double lastBound = bounds[i++];
        while(i < bounds.size())
        {
            double bound = bounds[i];
            if(bound >= b)
            {
                return value + forces[i]->integrateD(lastBound, b);
            } else{
                value += forces[i]->integrateD(lastBound, bound);
            }
            lastBound = bounds[i++];
        }

        Force* lastForce = forces.back();
        if(a > lastBound)
        {
            return lastForce->integrateD(a, b);
        } else{
            return value + lastForce->integrateD(lastBound, b);
        }
    }
}
