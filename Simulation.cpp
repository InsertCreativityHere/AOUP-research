
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
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
 * Runs the actual simulation.
 * @param force The external force to impose on the particles.
 * @param posRecorder Recorder for logging the particle's positions over time. NULL indicates it shouldn't be tracked.
 * @param forceRecorder Recorder for logging the particle's active forces over time. NULL indicates it shouldn't be tracked.
 * @param noiseRecorder Recorder for loggint the random noise applied to the system over time. NULL indicates it shouldn't be tracked.
 * @param particleCount The number of particles to simulate.
 * @param duration The length of the simulation in seconds.
 * @param timestep The amount of time to let pass between updating the simulation.
 * @param diffusion The diffusion rate of the system.
 * @param memory The time period over which a particle's motion persists.
 * @param dataDelay The number of timesteps to wait before logging data.
 * @param startBoundLeft The left bound of the region where particles are initially generated.
 * @param startBoundRight The right bound of the region where particles are initially generated.
 * @param activeForcesMean The mean of the active forces.
 * @param activeForcesStddev The standard deviation of the active forces.
 * @param noiseMean The mean of the random noise.
 * @param noiseStddev The standard deviation of the random noise.
 **/
void runSimulation(force::Force* force, histogram::Recorder* posRecorder, histogram::Recorder* forceRecorder, histogram::Recorder* noiseRecorder, const unsigned long particleCount, const double duration, const double timestep, const double diffusion, const double memory, const unsigned long dataDelay, const double startBoundLeft, const double startBoundRight, const double activeForcesMean, const double activeForcesStddev, const double noiseMean, const double noiseStddev)
{
    std::vector<double> positions = generateUniform(particleCount, startBoundLeft, startBoundRight);
    std::vector<double> activeForces = generateNormal(particleCount, activeForcesMean, activeForcesStddev);
    std::vector<double> noise;

    const auto totalSteps = (unsigned long)(duration / timestep);
    for(auto currentStep = 0; currentStep < totalSteps; currentStep++)
    {
        noise =  generateNormal(particleCount, noiseMean, noiseStddev);

        for(auto i = 0; i < particleCount; i++)
        {
            positions[i] += (activeForces[i] + force->getForce(positions[i])) * timestep;
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

/**TODO THIS IS INNACURATE AND NOT WELL WRITTEN
 * Generates a force from a stringified representation OF A POTENTIAL!!!!
 * @param str The stringified representation of a force structured as follows: "<type> [ param1 param2 ... ]". The extra spaces matter ALOT.
 *            type is the type of the force, and what follows is a space delimited sequence of force specific parameters, there are currently 3 types of forces.abort
 *            - Linear (deprecated, should remove this at some point...)
 *            - Poly   Force specified by a polynomial function. It's parameters represent the coeffecients of said polynomial, given in increasing order: (param1) + (param2)x + (param3)x^2 + ...
 *            - Piece  Force specified by a piecewise function. It's parameters are a force reference, followed by a bound, and a bit representing whether to evaluate on the left or right at the bound.
 *                     force references are given by @3, where 3 indicates to use the 3rd force generated thus far. As an example "piece [ @0 5 0 @1 ]" creates a piecewise function that takes on f0 to the
 *                     left of 5 (including at it), and f1 to the left of it. A direction bit of 0 means evaluate to the left, whereas a bit of 1 indicates the right.
 **/
force::Force* createForce(const std::string& str)
{
    std::vector<std::string> paramVector;
    std::istringstream iss(str);
    std::string s;

    while(iss >> std::quoted(s, '\''))
    {
        paramVector.push_back(s);
    }

    if(paramVector[0] == "poly")
    {
        std::vector<double> coeffecients(paramVector.size() - 2);
        std::transform((paramVector.begin() + 2), paramVector.end(), coeffecients.begin(), [](const std::string& str) {return std::stod(str);});
        for(int i = 0; i < coeffecients.size(); i++)
        {
            coeffecients[i] *= -(i + 1);
        }
        return new force::PolyForce(coeffecients);
    } else
    if(paramVector[0] == "piece")
    {
        int size = (paramVector.size() - 1) / 3;
        std::vector<force::Force*> forces(size);
        std::vector<double> bounds(size);
        std::vector<bool> directions(size);

        for(int i = 0; i < size; i++)
        {
            forces[i] = createForce(paramVector[3 * i]);
            bounds[i] = std::stod(paramVector[(3 * i) + 1]);
            directions[i] = !!std::stoi(paramVector[(3 * i) + 2]);
        }

        return new force::PieceForce(forces, bounds, directions);
    } else{
        return NULL;
    }
}

/**TODO THIS IS INNACURATE AND NOT WELL WRITTEN
 * Converts string arguments into a histogram recorder.
 * @param type The type of histogram recorder to create.
 *        linear = A histogram with equally spaced bins.
 *        custom = A histogram with customly specified bins.
 * @param parameters A comma delimited list of parameters, these are used to initialize the specified histogram type.
 *                   For further information on these, check the parameters for each type of histogram in Histogram.h.
 * @return A new histogram and a recorder on top of it, all according to the specified type and parameeters,
 **/
histogram::Recorder* createRecorder(const std::string& str)
{
    std::vector<std::string> paramVector;
    std::istringstream iss(str);
    std::string s;

    while(iss >> std::quoted(s, '\''))
    {
        paramVector.push_back(s);
    }

    if(paramVector[0] == "linear")
    {
        histogram::Histogram* histo = new histogram::LinearHistogram(std::stoul(paramVector[1]), std::stod(paramVector[2]), std::stod(paramVector[3]));
        return new histogram::Recorder(*histo);
    } else
    if(paramVector[0] == "custom")
    {
        std::vector<double> bins(paramVector.size() - 1);
        std::transform(++paramVector.begin(), paramVector.end(), bins.begin(), [](const std::string& str) {return std::stod(str);});
        histogram::Histogram* histo = new histogram::CustomHistogram(bins);
        return new histogram::Recorder(*histo);
    } else{
        return NULL;
    }
}

//TODO write something here!
int main(int argc, char* argv[])
{
    try
    {
        // Convert char arrays to lowercase strings for parsing.
        std::string args[argc];
        for(int i = 0; i < argc; i++)
        {
            args[i] = std::string(argv[i]);
            std::transform(args[i].begin(), args[i].end(), args[i].begin(), tolower);
        }

        // Generate the force.
        force::Force* force = createForce(args[1]);

        // Assign default values for everything else
        std::string outputFile = "./results";
        histogram::Recorder* posRecorder = NULL;
        histogram::Recorder* forceRecorder = NULL;
        histogram::Recorder* noiseRecorder = NULL;
        unsigned long particleCount = 100;
        double duration = 20;
        double timestep = 0.05;
        double diffusion = 1;
        double memory = 1;
        unsigned long dataDelay = 10;
        double startBoundLeft = -5;
        double startBoundRight = 5;
        double activeForcesMean = 0;
        double activeForcesStddev = 0.2;
        double noiseMean = 0;
        double noiseStddev = 1;

        for(int i = 2; i < argc; i++)
        {
            if(args[i] == "-of")
            {
                outputFile = args[++i];
            } else
            if(args[i] == "-pr")
            {
                posRecorder = createRecorder(args[++i]);
            } else
            if(args[i] == "-fr")
            {
                forceRecorder = createRecorder(args[++i]);
            } else
            if(args[i] == "-nr")
            {
                noiseRecorder = createRecorder(args[++i]);
            } else
            if(args[i] == "-n")
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
            } else
            if(args[i] == "-sb")
            {
                startBoundLeft = std::stod(args[i+1]);
                startBoundRight = std::stod(args[i+2]);
                i += 2;
            } else
            if(args[i] == "-af")
            {
                activeForcesMean = std::stod(args[i+1]);
                activeForcesStddev = std::stod(args[i+2]);
                i += 2;
            } else
            if(args[i] == "-no")
            {
                noiseMean = std::stod(args[i+1]);
                noiseStddev = std::stod(args[i+2]);
                i += 2;
            } else{
                std::cerr << "Skipping unknown parameter: " + args[i];
            }
        }

        runSimulation(force, posRecorder, forceRecorder, noiseRecorder, particleCount, duration, timestep, diffusion, memory, dataDelay, startBoundLeft, startBoundRight, activeForcesMean, activeForcesStddev, noiseMean, noiseStddev);

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
        return 0;
    } catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
    }
}

//TODO WRITE THIS SOMEWHERE ELSE!
//lists of parameters must be declared as "[thing1, thing2, thing3]", the whole list must be wrapped in quotes.
//if needed up to two layers of quotes can be nested, starting with "" outside, and using '' inside.

//TODO just fixing this up somehow, oh and also have it convert all the strings into lowercase!
//first parameter is always output filename
//second parameter is always the forces
//-m <memory amount>
//-d <diffusion amount>
//-p <particle amount>
//-dd <datadelay amount>
//-t <duration amount>
//-dt <timestep amount>
//-pr <histo_type> <histo_parameters>
//-fr <histo_type> <histo_parameters>
//-nr <histo_type> <histo_parameters>
