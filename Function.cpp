
#include <string>
#include "Function.h"

namespace function
{
    std::vector<double> Function::getValueForAll(const std::vector<double>& positions) const
    {
        std::vector<double> values(positions.size());
        for(auto& p : positions)
        {
            values.push_back(getValue(p));
        }

        return values;
    }

    PolyFunction::PolyFunction(std::vector<double> coefficients):
    coefficients(coefficients)
    {
    }

    double PolyFunction::getValue(double position) const
    {
        double value = 0;
        double temp = 1;
        for(const auto& coefficient : coefficients)
        {
            value += temp * coefficient;
            temp *= position;
        }

        return value;
    }

    PolyFunction* PolyFunction::negate() const
    {
        std::vector<double> parameters(coefficients.size());
        for(const auto& coefficient : coefficients)
        {
            parameters.push_back(-coefficient);
        }
        return new PolyFunction(parameters);
    }

    PolyFunction* PolyFunction::derivative() const
    {
        std::vector<double> parameters(coefficients.size() - 1);
        for(int i = 1; i < coefficients.size(); i++)
        {
            parameters.push_back(i * coefficients[i]);
        }
        return new PolyFunction(parameters);
    }

    PolyFunction* PolyFunction::integrateI(double c) const
    {
        std::vector<double> parameters(coefficients.size() + 1);
        parameters.push_back(c);
        for(int i = 0; i < coefficients.size(); i++)
        {
            parameters.push_back(coefficients[i] / (i + 1));
        }
        return (new PolyFunction(parameters));
    }

    double PolyFunction::integrateD(double a, double b) const
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

    PieceFunction::PieceFunction(std::vector<Function*> functions, std::vector<double> bounds, std::vector<bool> directions):
    functions(functions), bounds(bounds), directions(directions)
    {
        if(functions.size() != (bounds.size() + 1))
        {
            throw std::invalid_argument("Incorrect number of component functions provided. Expected " + std::to_string(bounds.size() + 1) + " recieved " + std::to_string(functions.size()) + ".");
        }
    }

    double PieceFunction::getValue(double position) const
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
        return functions[i]->getValue(position);
    }

    PieceFunction* PieceFunction::negate() const
    {
        std::vector<Function*> newFunctions(functions.size());
        for(const Function* function : functions)
        {
            newFunctions.push_back(function->negate());
        }
        std::vector<double> newBounds = bounds;
        std::vector<bool> newDirections = directions;
        return new PieceFunction(newFunctions, newBounds, newDirections);
    }

    PieceFunction* PieceFunction::derivative() const
    {
        std::vector<Function*> newFunctions(functions.size());
        for(const Function* function : functions)
        {
            newFunctions.push_back(function->derivative());
        }
        std::vector<double> newBounds = bounds;
        std::vector<bool> newDirections = directions;
        return new PieceFunction(newFunctions, newBounds, newDirections);
    }

    PieceFunction* PieceFunction::integrateI(double c) const
    {
        std::vector<Function*> newFunctions(functions.size());
        for(const Function* function : functions)
        {
            newFunctions.push_back(function->integrateI(c));
        }
        std::vector<double> newBounds = bounds;
        std::vector<bool> newDirections = directions;
        return new PieceFunction(newFunctions, newBounds, newDirections);
    }

    double PieceFunction::integrateD(double a, double b) const
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
                    return functions[i]->integrateD(a, b);
                } else{
                    value += functions[i]->integrateD(a, bound);
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
                return value + functions[i]->integrateD(lastBound, b);
            } else{
                value += functions[i]->integrateD(lastBound, bound);
            }
            lastBound = bounds[i++];
        }

        Function* lastFunction = functions.back();
        if(a > lastBound)
        {
            return lastFunction->integrateD(a, b);
        } else{
            return value + lastFunction->integrateD(lastBound, b);
        }
    }
}
