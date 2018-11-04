
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
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
 * Converts a comma-delimited string into a vector of substrings split across the occurence of commas.
 * @param str The string to parse comma delimited substrings from.
 * @return A vector of substrings containing the characters in between commas within each entry.
 **/
std::vector<std::string> parseDelimitedString(const std::string& str, const char delimiter)
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
        std::getline(stringStream, subString, delimiter);
        result.push_back(subString);
    }

    return result;
}

/**
 * Converts string arguments into a histogram recorder.
 * @param type The type of histogram recorder to create.
 *        linear = A histogram with equally spaced bins.
 *        custom = A histogram with customly specified bins.
 * @param parameters A comma delimited list of parameters, these are used to initialize the specified histogram type.
 *                   For further information on these, check the parameters for each type of histogram in Histogram.h.
 * @return A new histogram and a recorder on top of it, all according to the specified type and parameeters,
 **/
histogram::Recorder* createRecorder(const std::string& type, const std::string& parameters)
{
    std::vector<std::string> paramVector = parseDelimitedString(parameters, ',');
    if(type == "linear")
    {
        auto histo = histogram::LinearHistogram(std::stoul(paramVector[0]), std::stod(paramVector[1]), std::stod(paramVector[2]));
        return new histogram::Recorder(histo);
    } else
    if(type == "custom")
    {
        std::vector<double> bins(paramVector.size());
        for(const auto& p : paramVector)
        {
            bins.push_back(std::stod(p));
        }
        auto histo = histogram::CustomHistogram(bins);
        return new histogram::Recorder(histo);
    } else{
        return NULL;
    }
}

/**
 * Converts string arguments into a force.
 * @param type The type of force to create.
 *        linear = A force that varies linearly over space.
 *        poly = A force modeled by a polynomial function.
 *        piece = A force conprised of separate pieces. (it's parameters are further delineated by '~'s)
 * @param parameters A comma delimited list of parameters, these are used to intialize the specified force type.
 *                   For further information on these, check the parameters for each type of force in Force.h.
 * @return a new force created according to the specified type and parameters.
 **/
force::Force* createForce(const std::string& type, const std::string& parameters)
{
    std::vector<std::string> paramVector = parseDelimitedString(parameters, ',');
    if(type == "linear")
    {
        return new force::LinearForce(std::stod(paramVector[0]), std::stod(paramVector[1]));
    } else
    if(type == "poly")
    {
        std::vector<double> coeffecients(paramVector.size());
        for(const auto& p : paramVector)
        {
            coeffecients.push_back(std::stod(p));
        }
        return new force::PolyForce(coeffecients);
    } else
    if(type == "piece")
    {
        std::vector<std::string> forceParams = parseDelimitedString(paramVector[0], '~');
        std::vector<std::string> boundParams = parseDelimitedString(paramVector[1], '~');
        std::vector<std::string> directionParams = parseDelimitedString(paramVector[2], '~');

        std::vector<force::Force*> forces(forceParams.size());
        std::vector<double> bounds(boundParams.size());
        std::vector<bool> directions(directionParams.size());

        for(auto& f : forceParams)
        {
            std::replace(f.begin(), f.end(), '~', ',');
            auto pos = f.find(",");
            forces.push_back(createForce(f.substr(0, pos), f.substr(pos)));
        }
        for(const auto& b : boundParams)
        {
            bounds.push_back(std::stod(b));
        }
        for(const auto& d : directionParams)
        {
            if(d == "true")
            {
                directions.push_back(true);
            } else
            if(d == "false")
            {
                directions.push_back(false);
            } else{
                throw std::invalid_argument("No valid conversion found from string to boolean for malformed string.");
            }
        }
    } else{
        return NULL;
    }
}

/**
 * Runs the actual simulation.
 * @param force The external force to impose on the particles.
 * @param posRecorder Recorder for logging the particle's positions over time. NULL indicates it shouldn't be tracked.
 * @param forceRecorder Recorder for logging the particle's active forces over time. NULL indicates it shouldn't be tracked.
 * @param noiseRecorder Recorder for loggint the random noise applied to the system over time. NULL indicates it shouldn't be tracked.
 * @param particleCount The number of particles to simulate.
 * @param duration The length of the simulation in seconds.
 * @param timestep The amount of time to let pass between updating the simulation.
 * @param diffusion //TODO
 * @param memory //TODO
 * @param dataDelay The number of timesteps to wait before logging data.
 **/
void runSimulation(force::Force& force, histogram::Recorder* posRecorder, histogram::Recorder* forceRecorder, histogram::Recorder* noiseRecorder, const unsigned long particleCount, const double duration, const double timestep, const double diffusion, const double memory, const unsigned long dataDelay)
{
    std::vector<double> positions = generateUniform(particleCount, -5, 5);//TODO
    std::vector<double> activeForces = generateNormal(particleCount, 0, 0.2);//TODO
    std::vector<double> noise;

    const auto totalSteps = (unsigned long)(duration / timestep);
    for(auto currentStep = 0; currentStep < totalSteps; currentStep++)
    {
        noise =  generateNormal(particleCount, 0, 1);

        for(auto i = 0; i < particleCount; i++)
        {
            positions[i] += (activeForces[i] + force.getForce(positions[i])) * timestep;
            activeForces[i] += (-(activeForces[i] * timestep) + (sqrt(2 * diffusion * timestep)) * noise[i]) / memory;
        }

        if(currentStep % dataDelay == 0)
        {
            auto currentTime = (currentStep * timestep);
            if(posRecorder)
            {
                posRecorder->recordData(currentTime, positions);
            }
            if(forceRecorder)
            {
                forceRecorder->recordData(currentTime, activeForces);
            }
            if(noiseRecorder)
            {
                noiseRecorder->recordData(currentTime, noise);
            }
        }
    }
}

//TODO just fixing this up somehow, oh and also have it convert all the strings into lowercase!
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
int main(int argc, char* argv[])
{
    //Convert char arrays to lowercase strings for parsing.
    std::string args[argc];
    for(int i = 0; i < argc; i++)
    {
        args[i] = std::string(argv[i]);
        std::transform(args[i].begin(), args[i].end(), args[i].begin(), tolower);
    }

    //Load the path of the output file.
    std::string outputFile = args[1];

    //Load the force.
    force::Force* force = NULL;
    if(args[2] == "poly")
    {
        std::vector<std::string> parameters = parseDelimitedString(args[3], ',');
        std::vector<double> coeffecients;
        for(const auto& p : parameters)
        {
            coeffecients.push_back(std::stod(p));
        }

        force = new force::PolyForce(coeffecients);
    } else
    if(args[2] == "linear")
    {
        std::vector<std::string> parameters = parseDelimitedString(args[3], ',');
        force = new force::LinearForce(std::stod(parameters[0]), std::stod(parameters[1]));
    } else{
        std::cout << "Unknown force type: " + args[2];
        return 1;
    }

    //Set default values for simulation parameters.
    histogram::Recorder* posRecorder = NULL;
    histogram::Recorder* forceRecorder = NULL;
    histogram::Recorder* noiseRecorder = NULL;
    unsigned long particleCount = 100;
    double duration = 60;
    double timestep = 0.05;
    double diffusion = 1;
    double memory = 1;
    unsigned long dataDelay = 20;

    //Parse additional command line parameters.
    for(auto i = 4; i < argc; i++)
    {
        if(args[i] == "-pr")
        {
            posRecorder = createRecorder(args[i+1], args[i+2]);
            i += 2;
        } else
        if(args[i] == "-fr")
        {
            forceRecorder = createRecorder(args[i+1], args[i+2]);
            i += 2;
        } else
        if(args[i] == "-nr")
        {
            noiseRecorder = createRecorder(args[i+1], args[i+2]);
            i += 2;
        } else
        if(args[i] == "-p")
        {
            particleCount = std::stoul(args[++i]);
        } else
        if(args[i] == "-t")
        {
            duration = std::stod(args[++i]);
        } else
        if(args[i] == "-dt")
        {
            timestep = std::stod(args[++i]);;
        } else
        if(args[i] == "-d")
        {
            diffusion = std::stod(args[++i]);
        } else
        if(args[i] == "-m")
        {
            memory = std::stod(args[++i]);
        } else
        if(args[i] == "-dd")
        {
            dataDelay = std::stoul(args[++i]);
        } else{
            std::cout << "Unknown parameter: " + args[i];
        }
    }

    runSimulation(*force, posRecorder, forceRecorder, noiseRecorder, particleCount, duration, timestep, diffusion, memory, dataDelay);

    if(posRecorder)
    {
        posRecorder->writeData(outputFile + ".pos");
    }
    if(forceRecorder)
    {
        forceRecorder->writeData(outputFile + ".force");
    }
    if(noiseRecorder)
    {
        noiseRecorder->writeData(outputFile + ".noise");
    }
}
