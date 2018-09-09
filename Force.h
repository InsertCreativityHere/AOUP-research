
#ifndef FORCE_DEF
#define FORCE_DEF

#include <vector>

namespace force
{
    class Force
    {
        public:
            virtual FORCE_TYPE getForce(POS_TYPE position) = 0;
            std::vector<FORCE_TYPE> getForceForAll(const std::vector<POS_TYPE>& positions);
    };

    class LinearForce : public Force
    {
        public:
            LinearForce(FORCE_TYPE center, FORCE_TYPE slope);
            FORCE_TYPE getForce(POS_TYPE position);

        private:
            const FORCE_TYPE center;
            const FORCE_TYPE intensity;
    };

    class PolyForce : public Force
    {
        public:
            PolyForce(std::vector<FORCE_TYPE> coefficients);
            FORCE_TYPE getForce(POS_TYPE position);

        private:
            const std::vector<FORCE_TYPE> coefficients;
    };
}
#endif
