#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include "HistogramRecorder.h"
#include "Force.h"

std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());

std::vector<double> generateUniform(unsigned long size, double min, double max)
{
    std::vector<double> population(size);

    std::uniform_real_distribution<double> distribution(min, max);

    for(auto &pop : population)
    {
        pop = distribution(generator);
    }

    return population;
}

std::vector<double> generateNormal(unsigned long size, double mean, double stddev)
{
    std::vector<double> population(size);

    std::normal_distribution<double> distribution(mean, stddev);

    for(auto &pop : population)
    {
        pop = distribution(generator);
    }

    return population;
}

std::vector<double> coefficients = COEFFICIENTS;
force::PolyForce polyForce(coefficients);

histogram::LinearHistogram posHisto(BINCOUNT, BINMINP, BINMAXP);
histogram::Recorder posRecorder(&posHisto);
histogram::LinearHistogram forceHisto(BINCOUNT, BINMINF, BINMAXF);
histogram::Recorder forceRecorder(&forceHisto);

unsigned long totalSteps = (unsigned long)(DURATION / TIMESTEP);

void runSimulation()
{
    std::vector<double> positions = generateUniform(PARTICLES, -5, 5);
    std::vector<double> activeForces = generateNormal(PARTICLES, 0, 0.2);
    std::vector<double> noise;

    unsigned long currentStep = 0;
    while(currentStep < totalSteps)
    {
        noise =  generateNormal(PARTICLES, 0, 1);

        for(unsigned long i = 0; i < PARTICLES; i++)
        {
            positions[i] += (activeForces[i] + polyForce.getForce(positions[i])) * TIMESTEP;
            activeForces[i] += (-(activeForces[i] * TIMESTEP) + (sqrt(2 * DIFFUSION * TIMESTEP)) * noise[i]) / TAU;
        }

        if(currentStep % DATADELAY == 0)
        {
            auto currentTime =(currentStep * TIMESTEP);
            posRecorder.recordData(currentTime, positions);
            forceRecorder.recordData(currentTime, activeForces);
        }

        currentStep++;
    }
}

int main()
{
    runSimulation();
    auto tau_str = std::to_string(TAU);
    if(tau_str.find('.') != std::string::npos)
    {
        tau_str.erase((tau_str.find_last_not_of('0') + 1), std::string::npos);
        if(tau_str.back() == '.')
        {
            tau_str.append(1, '0');
        }
    }
    posRecorder.writeData("T=" + tau_str + ".pos");
    forceRecorder.writeData("T=" + tau_str + ".for");
}