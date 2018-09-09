
#include <chrono>
#include <cmath>
#include <random>
#include <sstream>
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
 * TODO
 **/
std::vector<std::string> parseCommaDelimitedString(const std::string& str)
{
    std::vector<std::string> result;
    std::stringstream stringStream;

    if(str.empty())
    {
        return result;
    }

    while(stringStream.good())
    {
        std::string subString;
        std::getLine(stringStream, subString, ',');
        result.push_back(subString);
    }

    return result;
}

/**
 * 
 **/
histogram::Recorder createRecorder(const std::string& type, const std::string& parameters)
{
    std::vector<std::string> paramVector = parseCommaDelimitedString(parameters);
    if(type == "linear")
    {
        return Histogram::Recorder(histogram::LinearHistogram(std::stoul(paramVector[0]), std::stod(paramVector[1]), std::stod(paramVector[2])));
    } else
    if(type == "custom")
    {
        return histogram::Recorder(histogram::CustomHistogram(paramVector));
    }
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
void runSimulation(force::Force& force, histogram::Recorder* posHisto, histogram::Recorder* forceHisto, histogram::Recorder* noiseHisto, const unsigned long particleCount, const double duration, const double timestep, const double diffusion, const double memory, const unsigned long dataDelay)
{
    std::vector<double> positions = generateUniform(particleCount, -5, 5);//TODO
    std::vector<double> activeForces = generateNormal(particleCount, 0, 0.2);//TODO
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
//first parameter is always output filename
//-f <force_type> <force_parameters>
//-m <memory amount>
//-d <diffusion amount>
//-p <particle amount>
//-dd <datadelay amount>
//-t <duration amount>
//-dt <timestep amount>
//-pr <histo_type> <histo_parameters>
//-fr <histo_type> <histo_parameters>
//-nr <histo_type> <histo_parameters>
int main(int argc, char[]* argv)
{
    std::string outputFile = argv[1];
    //Load default values for simulation parameters
    force::Force* force = NULL;
    histogram::Recorder* posRecorder = NULL;
    histogram::Recorder* forceRecorder = NULL;
    histogram::Recorder* noiseRecorder = NULL;
    unsigned long particleCount = 100;
    double duration = 60;
    double timestep = 0.05;
    double diffusion = 1;
    double memory = 1;
    unsigned long dataDelay = 20;

    //Parse command line parameters TODO make this better at not messing up when incorrect parameters are parsed...
    for(auto i = 1; i < argc; i++)
    {
        if(argv[i] == "-f")
        {
            if(argv[i+1] == "poly")
            {
                std::vector<std::string> parameters = parseCommaDelimitedString(argv[i+2]);
                std::vector<std::double> coeffecients;
                for(const auto& p : parameters)
                {
                    coeffecients.push_back(std::stod(p));
                }

                force = &force::PolyForce(coeffecients);
            } else
            if(argv[i+1] == "linear")
            {
                std::vector<std::string> parameters = parseCommaDelimitedString(argv[i+2]);
                force = &force::LinearForce(std::stod(parameters[0]), std::stod(parameters[1]));
            } else{
                //TODO unknown parameter
                return 1;
            }
        } else
        if(argv[i] == "-pr")
        {
            posRecorder = &createRecorder(argv[i+1], argv[i+2]);
        } else
        if(argv[i] == "-fr")
        {
            forceRecorder = &createRecorder(argv[i+1], argv[i+2]);
        } else
        if(argv[i] == "-nr")
        {
            noiseRecorder = &createRecorder(argv[i+1], argv[i+2]);
        } else
        if(argv[i] == "-p")
        {
            particleCount = std::stoul(argv[++i]);
        } else
        if(argv[i] == "-t")
        {
            duration = std::stod(argv[++i]);
        } else
        if(argv[i] == "-dt")
        {
            timeStep = std::stod(argv[++i]);;
        } else
        if(argv[i] == "-d")
        {
            diffusion = std::stod(argv[++i]);
        } else
        if(argv[i] == "-m")
        {
            memory = std::stod(argv[++i]);
        } else
        if(argv[i] == "-dd")
        {
            dataDelay = std::stoul(argv[++i]);
        } else{
            //TODO unknown specified
            return 1;
        }
    }

    if(!force)
    {
        //TODO no force?
        return 1;
    }

    runSimulation(force, posRecorder, forceRecorder, noiseRecorder, particleCount, duration, timestep, diffusion, memory, dataDelay);

    if(posRecorder)
    {
        posRecorder.writeData(outputFile + ".pos");
    }
    if(forceRecorder)
    {
        forceRecorder.writeData(outputFile + ".force");
    }
    if(noiseRecorder)
    {
        noiseRecorder.writeData(outputFile + ".noise");
    }
}
