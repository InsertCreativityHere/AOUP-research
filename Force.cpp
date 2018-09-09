#include "Force.h"

namespace force
{
    std::vector<FORCE_TYPE> Force::getForceForAll(const std::vector<POS_TYPE> &positions)
    {
        auto forces = positions;
        for(auto &convert : forces)
        {
            convert = getForce(convert);
        }

        return forces;
    }

    LinearForce::LinearForce(FORCE_TYPE center, FORCE_TYPE slope)
    : center(center), intensity(slope)
    {}
    
    FORCE_TYPE LinearForce::getForce(POS_TYPE position)
    {
        return (position - center) * intensity;
    }

    PolyForce::PolyForce(std::vector<FORCE_TYPE> coefficients)
    : coefficients(coefficients)
    {}

    FORCE_TYPE PolyForce::getForce(POS_TYPE position)
    {
        FORCE_TYPE totalForce = 0;
        FORCE_TYPE temp = 1;
        for(auto &coeffecient : coefficients)
        {
            totalForce += temp * coeffecient;
            temp *= position;
        }

        return totalForce;
    }
}