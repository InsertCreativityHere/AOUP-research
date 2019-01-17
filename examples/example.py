# IF YOU HAVEN'T COMPILED THE C++ SIDE YET, YOU MUST DO SO BEFORE RUNNING ANYTHING ELSE!!!
import sys; sys.path.append(".."); # since SimMain.py is one directory up from examples, and I don't know a better way.

# First, you obviously have to import the module
import SimMain as sm



#####===== THERE ARE 3 WAYS TO RUN SIMULATIONS =====#####
# 'runFromFile' runs a simulation straight from a parameter file:
sm.runFromFile("example.par")
# There's notes in the parameter file explaining how to write them.

# 'runSimulation' runs a single simulation as a direct python call:
sm.runSimulation(potential=sm.PolyFunc([0,0,1]),   outputFile="res2/out",   posRecorder=sm.LinearHistogram(-3, 3, dx=0.1))
# This simulation will run a polynomial potential (0 + 0x + 1x^2 to be exact), and
# will save it's results to a directory named 'res2'. All result files will be named
# 'out.*', where different extensions are used for different types of data.
# There's also a 'posRecorder' passed in. This means that the positions of the particles is
# recorded and saved to a file 'out.pos'. In this case the data will be formatted as a
# histogram that tracks positions between -3 and 3, and has bin widths of 0.1.

# Finally 'runSimulationMulti' can be used to run multiple simulations at once:
sm.runSimulationMulti(memory=[0.1, 1, 10],   potential=sm.PolyFunc([0,0,1]),   outputFile=["res3/out0.1", "res3/out1", "res3/out10"],   posRecorder=sm.LinearHistogram(-3, 3, dx=0.1))
# This shares all the same parameters as 'runSimulation' in the same order, but also accepts lists.
# Here, 3 simulations will be run, with tau=0.1, tau=1, tau=10 (Each also has it's own outputFile).
# Any parameters that aren't lists just have their value reused everytime. So all of these have the
# same position recorder and potential.

# All of the parameters are discussed in their own section 2 below this one.



#####===== WORKING WITH THE DATA =====#####
# Let's take the data written to 'res2/out'. Here we only tracked the position, and it got saved to
# 'out.pos', but there's also an out.png file. By default, the simulation will render a basic graph
# displaying the potential and results. (and compares it against predictions, but we didn't use any)
# That's all that is. The real data is saved in out.pos though.
# To load it in for heavier analysis run:
histos = sm.HistogramGroup("res2/out.pos")
# This reads in the file as a list of histograms over time. It's useful fields are:
#   histos.times : numpy array of the time each histogram occured at in the simulation.
#   histos.data  : numpy array of arrays. The inner arrays are just the values of the histograms,
#                  and the outer array is just all the histograms over time.
histos.times[-1]
histos.data[-1]
# So the above commands will print out the stopping time of the simulation, and the final distribution
# of positions. NOTE the first and last entries (data[i][0] and data[i][-1]) are overflow bins. Here
# we had our bounds as -3 and 3, so any particles to the left of -3 are counted in data[i][0] and
# particles to the right of 3 are in data[i][-1]. As long as you set good bounds, these should just be
# 0 though. For this reason it's more useful to skip the overflow bins and only look at the 'real' data:
histos.data[-1][1:-1]

# From here you can really do whatever you want with the data...

# There's also some helpful visualization stuff already packed in though.
sm.BarValueAnimator.viewFromFile("res2/out.pos")
# Will play an animation of the positions as a bar graph over time.
# (Useful for visually checking if equilibrium was reached). And also,
sm.BarFluxAnimator.viewFromFile("res2/out.pos")
# Will play an animation of the net flux of particles through each bin over time.
# Negative values indicate particles left the region, positive is particles entered.



#####===== SHORTNAME CODES =====#####
# This is pretty pedantic, but I found typing some names out to be annoying, so every class
# has a short-name code that's way faster to type out. Might save you a couple seconds.
# (These don't work in parameter files, only in direct python code.).
#   ThermalDensityPredictor               = t_pred
#   SingleWellPersistentDensityPredictor  = swp_pred
#   DoubleWellPersistentDensityPredictor  = dwp_pred
#   PersistentDensityPredictor            = p_pred
#   HistogramGroup                        = hist_g
#   BarValueAnimator                      = bv_anim
#   BarFluxAnimator                       = bf_anim
#   LinearHistogram                       = l_hist
#   CustomHistogram                       = c_hist
#   runSimulationMulti                    = rsm
#   runSimulation                         = rss



#####===== PARAMETERS (IN ORDER) =====#####
# Any parameter starting with * means it has it's own section underneath.
# These names are the same for both 'runSimulation' and 'runSimulationMulti',
# and are also the same as the paramter file's and C++ command line arguments
# (except for the last 3)
#      NAME                   DEFAULT           DESCRIPTION
#  *potential:                --              The external potential to impose on the particles.
#   outputFile:               'result'        The file to save results to. [string]
#  *predictions:              None            List of predictions to compare results against.
#  *posRecorder:              None            Recorder for logging the particle's positions over time. None indicates positions shouldn't be recorded.
#  *forceRecorder:            None            Recorder for logging the particle's active forces over time. None indicates forces shouldn't be recorded.
#  *noiseRecorder:            None            Recorder for logging the random noise applied to the system over time. None indicates noise shouldn't be recorded.
#	particleCount:            100             The number of particles to simulate. [positive integer]
#	duration:                 20              The length of the simulation in seconds. [positive float]
#	timestep:                 0.05            The amount of time to let pass between updating the simulation. [positive float]
#	diffusion:                1               The diffusion rate of the system. [float]
#	memory:                   1               The time period over which a particle's motion persists. [float]
#	dataDelay:                10              The number of timesteps to wait before logging data. [positive integer]
# THE NEXT 3 ENTRIES HAVE DIFFERENT NAMES IN PARAMETER FILES THAN THEY DO HERE!
#	startBounds:              (-5,5)          The left and right bound of the region where particles are initially generated. [2-tuple of floats]
#	activeForcesMean:         (0,0.2)         The mean and std-dev of the active forces. [2-tuple of floats]
#	noiseMean:                (0,1)           The mean and std-dev of the random noise.  [2-tuple of floats]


# There are 3 kinds of potentials currently implemented:
### POLYNOMIALS ###
f1 = sm.PolyFunc([ 1 , 0 , 3 ])
# This creates a polynomial function with the coeffecients 1 + 0x + 3x^2.
f2 = sm.PolyFunc([ -1 , 2 ])
# This creates a polynomial function with the coeffecients -1 + 2x.

### PERIODIC ###
f3 = sm.PeriodFunc(f2, 1, 3)
# This generates a periodic function with period T=(3-1)=2 that repeats the values of f2
# over the interval [1,3) indefinitely. In this case f3 is a sawtooth wave, since it repeats
# the section of f2 (just a line) between 1 and 3 indefinitely. Note that this always has offset=0,
# so f3(0)=f2(1).


### PIECEWISE ###
f4 = sm.PieceFunc([f1, f2, f3], [1, 7])
# This creates a piecewise function with:
#     f1  whenever  x<1
#     f2  whenever  1<x<7
#     f3  whenever  x>7
# By default, boundaries are evaluated to the left. so f4(1)=f1(1) and not f2(2), since f1(1) is to the
# left of x=1. You can change this behavior with directions:
f4 = sm.PieceFunc([f1, f2, f3], [1, 7], directions=[True, False])
# True means evaluate on the right, and False means evaluate on the left.
# So now f4(1) will evaluate to it's right function f2(1), and f4(7) will evaluate to it's left function, also f2(1).

# All functions support some basic methods:
f1.negate()    # Flips the sign of all the function's output (mirrors it over y=0)
f1.derive()    # Returns the derivative of the function
f1.integD(4,9) # Integrates the function between 4 and 9
# And calling str(function) will return the same stringified version used in parameter files, and C++
# command line arguments.
str(f1)


### RECORDERS ###
# There's two kinds of recorders, a linear histogram with equally spaced bins, and a custom histogram where you
# can specify the exact bin bounds.
# For the linear histogram, you specify the left and right bounds along with the bin width:
linear = sm.LinearHistogram(3, 6, dx=0.1)
# This creates a recorder that only tracks particles with 3<position<6 and has bins of width 0.1
# Alternatively you can specify the total number of bins instead:
linear = sm.LinearHistogram(3, 6, binCount=1000)
# This creates a recorder with the same bounds, but will have 1000 equally spaced bins now.

# For custom histograms, just pass in a list of boundaries:
custom = sm.CustomHistogram([0, 1, 2, 4, 8, 16])
# So this creates a recorder that tracks particles with 0<position<16, but that mimics a log scale.


### PREDICTORS ###




#####===== FULL WORKING EXAMPLES =====#####
# Run a simulation that records position, force, and noise, and only checks against thermal predictions
potential = sm.PolyFunc([0, 0.1, -0.5, 0, 0.2])       # This is 0.2x^4 - 0.5x^2 +0.1x
recorder = sm.LinearHistogram(3, -3, dx=0.1)          # Tracks values between 3 and -3 with a bin width of 0.1
predictors = [sm.ThermalDensityPredictor(potential)]  # Create a thermal limit prediction for the specified potential.
memory = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
diffusion = [np.sqrt(1 + m**2) for m in memory]
outputFiles = ["output" + str(m) for m in memory]
sm.runSimulationMulti(potential, outputFiles, predictors, posRecorder=recorder, forceRecorder=recorder, noiseRecorder=recorder, particleCount=10000, memory=memory, diffusion=diffusion)
# idk why I'm writing these, you're just going to use the parameter files anyways... TODO finish writing this!