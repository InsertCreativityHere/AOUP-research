
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
#include "Function.h"

/**
 * Engine for generating random distributions both for starting conditions and random noise.
 **/
std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());

/**
 * Creates a uniform distribution where all values inside the range are equally likely.
 * @param size: The number of points to sample from within the distribution.
 * @param min: The minimum value to value of the range to generate from.
 * @param max: The maximum value to value of the range to generate from.
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
 * Creates a normal distribution where all values are generated from a gaussian curve.
 * @param size: The number of points to sample from within the distribution.
 * @param mean: The average value, the one most likely to be generated.
 * @param stddev: The standard deviation, how spread out the data is from the mean.
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
 * @param force: The external force to impose on the particles.
 * @param posRecorder: Recorder for logging the particle's positions over time. NULL indicates positions shouldn't be recorded.
 * @param forceRecorder: Recorder for logging the particle's active forces over time. NULL indicates forces shouldn't be recorded.
 * @param noiseRecorder: Recorder for loggint the random noise applied to the system over time. NULL indicates noise shouldn't be recorded.
 * @param particleCount: The number of particles to simulate.
 * @param duration: The length of the simulation in seconds.
 * @param timestep: The amount of time to let pass between updating the simulation.
 * @param diffusion: The diffusion rate of the system.
 * @param memory: The time period over which a particle's motion persists.
 * @param dataDelay: The number of timesteps to wait before logging data.
 * @param startBoundLeft: The left bound of the region where particles are initially generated.
 * @param startBoundRight: The right bound of the region where particles are initially generated.
 * @param activeForcesMean: The mean of the active forces.
 * @param activeForcesStddev: The standard deviation of the active forces.
 * @param noiseMean: The mean of the random noise.
 * @param noiseStddev: The standard deviation of the random noise.
 **/
void runSimulation(function::Function* force, histogram::Recorder* posRecorder, histogram::Recorder* forceRecorder, histogram::Recorder* noiseRecorder, const unsigned long particleCount, const double duration, const double timestep, const double diffusion, const double memory, const unsigned long dataDelay, const double startBoundLeft, const double startBoundRight, const double activeForcesMean, const double activeForcesStddev, const double noiseMean, const double noiseStddev)
{
    unsigned int percent = 10;
    std::vector<double> positions = generateUniform(particleCount, startBoundLeft, startBoundRight);
    std::vector<double> activeForces = generateNormal(particleCount, activeForcesMean, activeForcesStddev);
    std::vector<double> noise;

    const auto totalSteps = (unsigned long)(duration / timestep);
    for(auto currentStep = 0; currentStep < totalSteps; currentStep++)
    {
        noise =  generateNormal(particleCount, noiseMean, noiseStddev);

        for(auto i = 0; i < particleCount; i++)
        {
            positions[i] += (activeForces[i] + force->getValue(positions[i])) * timestep;
            activeForces[i] += (-(activeForces[i] * timestep) + (sqrt(2 * diffusion * timestep)) * noise[i]) / memory;
        }

        if(currentStep % dataDelay == 0)
        {
            if(((currentStep * 100) / totalSteps) > percent)
            {
                std::cout << "Simulation Progress: " << percent << "%" << std::endl;
                percent += 10;
            }

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

/**
 * Creates a function from it's stringified representation. In general these look like:
 *     <type> <param1 param2 param3...>
 * Where type is the function type, currently only the following function types are supported:
 *     "poly" = A polynomial function, where parameters is a list of coeffecients in ascending order.
 *     "piece" = A piecewise function made of multiple component functions.
 * The additional parameters must be separated by only whitespace, and are passed directly into the
 * constructor of the specified function type. For information on the parameters, check Function.cpp
 *
 * @param str: The stringified representation of a function to create.
 * @returns: A new function, all according to the specified type and parameeters.
 **/
function::Function* createFunction(const std::string& str)
{
    // Split the string across whitespace (ignoring whitespace contained within single quotes) for parsing.
    std::vector<std::string> paramVector;
    std::istringstream iss(str);
    std::string s;

    while(iss >> std::quoted(s, '\''))
    {
        paramVector.push_back(s);
    }

    // Convert the first parameter (function type) to  lowercase.
    std::transform(paramVector[0].begin(), paramVector[0].end(), paramVector[0].begin(), tolower);

    // Create the specified function with it's provided parameters.
    if(paramVector[0] == "poly")
    {
        // Parse the list of coeffecients and create a new polynomial from them.
        std::vector<double> coeffecients(paramVector.size() - 1);
        std::transform((paramVector.begin() + 1), paramVector.end(), coeffecients.begin(), [](const std::string& str) {return std::stod(str);});
        return new function::PolyFunction(coeffecients);
    } else
    if(paramVector[0] == "piece")
    {
        // Allocate vectors for storing the piecewise function's parameters.
        int size = (paramVector.size() - 1) / 3;
        std::vector<function::Function*> functions(size + 1);
        std::vector<double> bounds(size);
        std::vector<bool> directions(size);

        // Parse every triple of arguments as another piecewise component.
        for(int i = 0; i < size; i++)
        {
            functions[i] = createFunction(paramVector[(3 * i) + 1]);
            bounds[i] = std::stod(paramVector[(3 * i) + 2]);
            directions[i] = !!std::stoi(paramVector[(3 * i) + 3]);
        }
        // The last argument should always be the function in the rightmost region.
        functions[size] = createFunction(paramVector[paramVector.size() - 1]);

        return new function::PieceFunction(functions, bounds, directions);
    } else{
        throw std::invalid_argument(paramVector[0] + " is not a valid function type.");
    }
}

/**
 * Creates a histogram recorder from it's stringified representation. In general these look like:
 *     <type> <param1 param2 param3...>
 * Where type is the histogram type, currently only the following histogram types are supported:
 *     "linear" = A histogram with equally spaced bins.
 *     "custom" = A histogram with customly specified bins.
 * The additional parameters must be separated by only whitespace, and are passed directly into the
 * constructor of the specified histogram type. For information on the parameters, check Histogram.cpp
 *
 * @param str: The stringified representation of a histogram to create.
 * @returns: A new histogram and a recorder on top of it, all according to the specified type and parameeters.
 **/
histogram::Recorder* createRecorder(const std::string& str)
{
    // Split the string across whitespace (ignoring whitespace contained within single quotes) for parsing.
    std::vector<std::string> paramVector;
    std::istringstream iss(str);
    std::string s;

    while(iss >> std::quoted(s, '\''))
    {
        paramVector.push_back(s);
    }

    // Convert the first parameter (histogram type) to  lowercase.
    std::transform(paramVector[0].begin(), paramVector[0].end(), paramVector[0].begin(), tolower);
    // Create the specified histogram with it's provided parameters.
    if(paramVector[0] == "linear")
    {
        histogram::Histogram* histo = new histogram::LinearHistogram(std::stod(paramVector[1]), std::stod(paramVector[2]), std::stod(paramVector[3]));
        return new histogram::Recorder(*histo);
    } else
    if(paramVector[0] == "custom")
    {
        // Parse all the bins of the histogram and create a new histogram from them.
        std::vector<double> bins(paramVector.size() - 1);
        std::transform(++paramVector.begin(), paramVector.end(), bins.begin(), [](const std::string& str) {return std::stod(str);});
        histogram::Histogram* histo = new histogram::CustomHistogram(bins);
        return new histogram::Recorder(*histo);
    } else{
        throw std::invalid_argument(paramVector[0] + " is not a valid histogram type.");
    }
}

/**
 * Main method which parses command line arguments into simulation parameters, then runs the simulation, exporting
 * any recorded data into their respective files at the end. This can only run a single simulation, single-threaded.
 * For running multiple simulations at once, use the Python wrapper Main.py instead.
 *
 * Every simulation parameter has a long name which can be specified with "--longName", and a short name for "-shortName".
 * Parameter values are all space delimited. Complex parameters like potentials and recorders should always be enclosed in
 * single quotes, with sub parameters being space delimited within said quotes.
 *
 * The following is a complete list of the simulation parameters:
 * The first parameter is always the potential.
 * outputFile    (of): The file name to save results to. NOTE, this should NOT include an extension, as separate extensions
 *                         are used for different data types, all of which are generated automatically and internally.
 * posRecorder   (pr): Stringified representation of the recorder to use for tracking particle positions over time.
 * forceRecorder (fr): Stringified representation of the recorder to use for tracking the active forces of particles over time.
 * noiseRecorder (nr): Stringified representation of the recorder to use for tracking the noise imposed on particles over time.
 * particleCount (n): The number of particles to use in the simulation.
 * duration      (t): The duration of time to run the simulation for. The simulation will run for $ceil(t/dt)$ total steps.
 * timestep      (dt): The interval of time to let pass in between simulation updates.
 * diffusion     (d): The diffusion coefficient to use in the simulation. Represents the strength of thermal flucuations.
 * memory        (m): The memory coeffecient to use in the simulation. Represents the strength of motion persistence.
 * dataDelay     (dd): The number of simulation updates to let pass in between each time data is recorded.
 * startBound    (sb): Takes two parameters for the left and right limits of where to generate partices at, in that order.
 * activeForce   (af): Takes two parameters for the mean and standard deviation of the distribution that initial active forces are drawn from, in that order.
 * noise         (no): Takes two parameters for the mean and standard deviation of the distribution that noise is drawn from, in that order.
 **/
int main(int argc, char* argv[])
{
    try
    {
        // Convert char arrays to strings for parsing.
        std::string args[argc];
        for(int i = 0; i < argc; i++)
        {
            args[i] = std::string(argv[i]);
        }

        // Generate the force from a potetential function. This is always the first argument.
        function::Function* force = createFunction(args[1]);
        force = force->derivative()->negate();

        // Assign default values for everything else before parsing. (these must mirror the ones in the Python side)
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

        // Parse the additional command line arguments for simulation parameters.
        for(int i = 2; i < argc; i++)
        {
            if((args[i] == "-of") || (args[i] == "--outputFile"))
            {
                outputFile = args[++i];
            } else
            if((args[i] == "-pr") || (args[i] == "--posRecorder"))
            {
                posRecorder = createRecorder(args[++i]);
            } else
            if((args[i] == "-fr") || (args[i] == "--forceRecorder"))
            {
                forceRecorder = createRecorder(args[++i]);
            } else
            if((args[i] == "-nr") || (args[i] == "--noiseRecorder"))
            {
                noiseRecorder = createRecorder(args[++i]);
            } else
            if((args[i] == "-n") || (args[i] == "--particleCount"))
            {
                particleCount = std::stoul(args[++i]);
            } else
            if((args[i] == "-t") || (args[i] == "--duration"))
            {
                duration = std::stod(args[++i]);
            } else
            if((args[i] == "-dt") || (args[i] == "--timestep"))
            {
                timestep = std::stod(args[++i]);;
            } else
            if((args[i] == "-d") || (args[i] == "--diffusion"))
            {
                diffusion = std::stod(args[++i]);
            } else
            if((args[i] == "-m") || (args[i] == "--memory"))
            {
                memory = std::stod(args[++i]);
            } else
            if((args[i] == "-dd") || (args[i] == "--dataDelay"))
            {
                dataDelay = std::stoul(args[++i]);
            } else
            if((args[i] == "-sb") || (args[i] == "--startBound"))
            {
                startBoundLeft = std::stod(args[i+1]);
                startBoundRight = std::stod(args[i+2]);
                i += 2;
            } else
            if((args[i] == "-af") || (args[i] == "--activeForce"))
            {
                activeForcesMean = std::stod(args[i+1]);
                activeForcesStddev = std::stod(args[i+2]);
                i += 2;
            } else
            if((args[i] == "-no") || (args[i] == "--noise"))
            {
                noiseMean = std::stod(args[i+1]);
                noiseStddev = std::stod(args[i+2]);
                i += 2;
            } else{
                std::cerr << "Unknown paramter: " + args[i];
                return 1;
            }
        }

        // Run the simulation.
        runSimulation(force, posRecorder, forceRecorder, noiseRecorder, particleCount, duration, timestep, diffusion, memory, dataDelay, startBoundLeft, startBoundRight, activeForcesMean, activeForcesStddev, noiseMean, noiseStddev);

        // Write any recorded data to a file.
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
