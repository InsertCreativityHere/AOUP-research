
#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include "HistogramRecorder.h"
#include "Force.h"

/**
 * Engine for generating random distributions both for starting conditions and random noise.
 **/
std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());

/**
 * Creates a uniform distribution where all values inside the range are equally likely.
 * @param size The number of points to generate within the distribution.
 * @param min The minimum value to generate within the range.
 * @param max The maximum value to generate within the range.
 **/
std::vector<double> generateUniform(unsigned long size, double min, double max)
{
    std::vector<double> population(size);
    std::uniform_real_distribution<double> distribution(min, max);

    for(auto& pop : population)
    {
        pop = distribution(generator);
    }

    return population;
}

/**
 * Creates a normal distribution where all values are generated according to a gaussian distribution.
 * @param size The number of points to generate within the distribution.
 * @param mean The average value, the one most likely to be generated.
 * @param stddev The standard deviation, how spread out the data is from the mean.
 **/
std::vector<double> generateNormal(unsigned long size, double mean, double stddev)
{
    std::vector<double> population(size);
    std::normal_distribution<double> distribution(mean, stddev);

    for(auto& pop : population)
    {
        pop = distribution(generator);
    }

    return population;
}

/**
 * Runs the actual simulation.
 * @param force The external force to impose on the particles.
 * @param posHisto Histogram for logging the particle's positions over time. NULL indicates it shouldn't be tracked.
 * @param forceHisto Histogram for logging the particle's active forces over time. NULL indicates it shouldn't be tracked.
 * @param noiseHisto Histogram for loggint the random noise applied to the system over time. NULL indicates it shouldn't be tracked.
 * @param particleCount The number of particles to simulate.
 * @param duration The length of the simulation in seconds.
 * @param timestep The amount of time to let pass between updating the simulation.
 * @param diffusion //TODO
 * @param memory //TODO
 * @param dataDelay The number of timesteps to wait before logging data.
 **/
void runSimulation(Force& force, Histogram* posHisto, Histogram* forceHisto, Histogram* noiseHisto, const unsigned long particleCount, const double duration, const double timestep, const double diffusion, const double memory, const unsigned int dataDelay)
{
    std::vector<double> positions = generateUniform(particleCount, -5, 5);
    std::vector<double> activeForces = generateNormal(particleCount, 0, 0.2);
    std::vector<double> noise;

    const auto totalSteps = (unsigned long)(duration / timestep);
    for(auto currentStep = 0; currentStep < totalSteps; currentStep++);
    {
        noise =  generateNormal(particleCount, 0, 1);

        for(auto i = 0; i < particleCount; i++)
        {
            positions[i] += (activeForces[i] + polyForce.getForce(positions[i])) * timestep;
            activeForces[i] += (-(activeForces[i] * timestep) + (sqrt(2 * diffusion * timestep)) * noise[i]) / memory;
        }

        if(currentStep % dataDelay == 0)
        {
            auto currentTime = (currentStep * timestep);
            if(posHisto)
            {
                posRecorder->recordData(currentTime, positions);
            }
            if(forceHisto)
            {
                posRecorder->recordData(currentTime, activeForces);
            }
            if(noiseHisto)
            {
                posRecorder->recordData(currentTime, noise);
            }
        }
    }
}

//TODO
int main(int argc, char[]* argv)
{
    std::vector<double> coefficients = COEFFICIENTS;
    force::PolyForce polyForce(coefficients);

    histogram::LinearHistogram posHisto(BINCOUNT, BINMINP, BINMAXP);
    histogram::Recorder posRecorder(&posHisto);
    histogram::LinearHistogram forceHisto(BINCOUNT, BINMINF, BINMAXF);
    histogram::Recorder forceRecorder(&forceHisto);

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
