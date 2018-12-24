
from matplotlib import pyplot as plt;
from matplotlib import patches as patches;
from matplotlib import path as path;
from matplotlib import animation as animation;
import copy as cp;
import numpy as np;
import scipy as sp;
import subprocess as sproc;
import inspect;
import os;
import shlex;
import threading;

#==================================================================================================================
#                                       ---PREDICTION GENERATION---
# The classes in this section generate prediction profiles for various potentials and situations. First a prediction is
# initialized with the external potential, and optional parameters that vary over the different prediction types.
# Then one can call 'generateProfile', which returns a function representing the prediction. These functions are normalized,
# and designed to handle NumPy arrays. generateProfile always takes the form of Predictor.generateProfile(index, memory, diffusion).
# Where index is the index of the simulation generating the profile (for logging purposes), and memory and diffusion are
# the respective constants being used in the simulation.

'''
Generates the predicted density profile for a system in the thermal limit (ie. diffusion >> memory).
'''
class ThermalDensityPredictor:
    '''Creates a new predictor for a potential in the thermal limit.
       @param potential: Callable that returns the external potential of the system as a function of position. (Must accept NumPy arrays).
       @param xMin: The lower position bound used in calculating the normalization constant, larger is better. (defaults to -1000)
       @param xMax: The upper position bound used in calculating the normalization constant, larger is better. (defaults to 1000)
       @param dx: The position difference to use in calculating the normaliztion constant, smaller is better. (defaults to 0.001)'''
    def __init__(self, potential, xMin=-1000, xMax=1000, dx=0.001):
        self.potential = potential;
        self.dx = dx;
        # Pre-compute a sample of the potential to speed up normalization later.
        self.Us = -self.potential(np.arange(xMin, xMax, self.dx));
        # Set the name and preferred color for this predictor (this is used for generating plots).
        self.name = "thermal";
        self.color = "orange";

    '''Generates the prediction profile as a normalized density distribution.
       @param index: The index of the simulation, for logging purposes. This should be an integer, but can actually be anything.
       @param memory: The memory constant being used in the simulation.
       @param diffusion: The diffusion constant being used in the simulation.
       @returns: A function specifying the predicted density of particles at every position.'''
    def generateProfile(self, index, memory, diffusion):
        # Compute the normalization constant by integrating the un-normalized density with the trapezoidal algorithm.
        Y = np.exp(self.Us / diffusion);
        normalization = np.trapz(Y, dx=self.dx);
        # Return the normalized density function.
        return lambda x: np.exp(-(self.potential(x)) / diffusion) / normalization;
# Create a shortname alias for ThermalDensityPredictor.
t_pred = ThermalDensityPredictor;

'''
Generates the predicted density profile for a system in the persistent limit (ie. memory >> diffusion), for a single well potential.
'''
class SingleWellPersistentDensityPredictor:
    '''Creates a new predictor for a single well potential in the persistent limit.
       @param potential: Callable that returns the external potential of the system as a function of position. (Must accept NumPy arrays).'''
    def __init__(self, potential):
        self.dU = potential.derive();
        self.d2U = self.dU.derive();
        # Set the name and preferred color for this predictor (this is used for generating plots).
        self.name = "persistent(sw)";
        self.color = "green";

    '''Generates the prediction profile as a normalized density distribution.
       @param index: The index of the simulation, for logging purposes. This should be an integer, but can actually be anything.
       @param memory: The memory constant being used in the simulation.
       @param diffusion: The diffusion constant being used in the simulation.
       @returns: A function specifying the predicted density of particles at every position.'''
    def generateProfile(self, index, memory, diffusion):
        # Pre-compute some constants.
        c0 = memory / (2 * diffusion);
        c1 = np.sqrt(memory / (2 * np.pi * diffusion));

        # Return the predicted density distribution.
        return lambda x: (c1 * self.d2U(x)) * np.exp(-c0 * (self.dU(x)**2));
# Create a shortname alias for  SingleWellPersistentDensityPredictor.
swp_pred = SingleWellPersistentDensityPredictor;

'''
Generates the predicted density profile for a system in the persistent limit (ie.  memory >> diffusion), for double well potentials.
'''
class DoubleWellPersistentDensityPredictor:
    '''Creates a new predictor for a double well potential in the persistent limit.
       @param potential: Callable that returns the external potential of the system as a function of position. (Must accept NumPy arrays).
                         This must be a PiecewiseCustom2ndOrder at the moment. #TODO make this more general one day.
       @param xMin: The lower position bound used in calculating the normalization constant, larger is better. (defaults to -1000)
       @param xMax: The upper position bound used in calculating the normalization constant, larger is better. (defaults to 1000)
       @param dx: The position difference to use in calculating the normaliztion constant, smaller is better. (defaults to 0.001)'''
    def __init__(self, potential, xMin=-1000, xMax=1000, dx=0.0001):
        self.dU = potential.derive();
        self.d2U = self.dU.derive();
        self.dx = dx;
        # Set the name and preferred color for this predictor (this is used for generating plots).
        self.name = "persistent(dw)";
        self.color = "purple";
        # Store a sample of x values for normalizing the prediction later.
        self.Xs = np.arange(xMin, xMax, self.dx);

        #TODO MAKE THIS WORK FOR POLYNOMIALS TOO!
        # Store the locations of the points of inflection.
        self.B = self.dU.bounds[1];
        self.C = self.dU.bounds[3];
        # Ensure that the provided points are inflection points.
        if(np.abs(self.d2U(self.B)) > 0.001):
            raise ValueError("Point of inflection not present at specified location B=" + str(self.A));
        if(np.abs(self.d2U(self.C)) > 0.001):
            raise ValueError("Point of inflection not present at specified location C=" + str(self.B));

        # Find the points that match the slopes at the inflection points, by exploiting the fact that the end-functions are quadratics.
        self.A = (self.dU(self.C) - self.dU.functions[0].c[0]) / self.dU.functions[0].c[1];
        self.D = (self.dU(self.B) - self.dU.functions[-1].c[0]) / self.dU.functions[-1].c[1];
        # Ensure that the slopes match those at the inflection points.
        if(np.abs(self.dU(self.A) - self.dU(self.C)) > 0.001):
            raise ValueError("Failed to locate slope matching C. Expected location was A=" + str(self.A));
        if(np.abs(self.dU(self.D) - self.dU(self.B)) > 0.001):
            raise ValueError("Failed to locate slope matching B. Expected location was D=" + str(self.D));


    '''Generates the prediction profile as a normalized density distribution.
       @param index: The index of the simulation, for logging purposes. This should be an integer, but can actually be anything.
       @param memory: The memory constant being used in the simulation.
       @param diffusion: The diffusion constant being used in the simulation.
       @returns: A function specifying the predicted density of particles at every position.'''
    def generateProfile(self, index, memory, diffusion):#TODO THE INTEGRALS ARE SUPER INEFFECIENT
        # Pre-compute some constants.
        c0 = memory / (2 * diffusion);
        c1 = np.sqrt(memory / (2 * np.pi * diffusion));
        # Compute the distribution integrals over the regions of AB and CD.
        Z1 = -self.i(self.A, self.B, c0);
        Z2 = self.i(self.C, self.D, c0);

        # Create a function for generating the steady state distribution within a single well.
        p0 = lambda x: (c1 * self.d2U(x)) * np.exp(-c0 * (self.dU(x)**2));

        # Create a list for storing the piecewise functions that comprise the prediction.
        functions = [];
        functions.append(p0);
        functions.append(lambda x: (p0(x) * (-np.vectorize(self.i)(x, self.B, c0)) / Z1));
        functions.append(lambda x: np.zeros(x.shape));
        functions.append(lambda x: (p0(x) * (np.vectorize(self.i)(self.C, x, c0)) / Z2));
        functions.append(p0);
        # Combine the functions into a piecewise function.
        #TODO THIS MIGHT BE VERY BROKEN AT INTEGER BOUNDARIES!!!
        pred = np.piecewise(x, [(x < self.A), (x >= self.A), (x >= self.B), (x >= self.C), (x >= self.D)], functions);

        # Compute the normalization constant for the prediction.
        Z = np.trapz(pred(self.Xs), dx=self.dx);

        # Return the normalized piecewise density function.
        return lambda x: pred(x) / Z;

    '''Function for computing distribution integrals internally.
       @param a: The lower bound of integration. This must be a scalar smaller than, or equal to b.
       @param b: The upper bound of integration. This must be a scalar larger or equal to than a.
       @param c: Constant computed from the simulation parameters.'''
    def i(self, a, b, c):
        Y = np.exp(c * self.dU(np.arange(a, b, self.dx))**2);
        return np.trapz(Y, dx=self.dx);
# Create a shortname alias for DoubleWellPersistentDensityPredictor.
dwp_pred = DoubleWellPersistentDensityPredictor;



#TODO CHECK THIS THING, IT'S PROBABLY BROKEN, ALSO COMMENTS
from scipy.special import erfi
import scipy.linalg
import itertools as itt

'''THIS IS LARGELY NON-FUNCTIONAL'''
class PersistentDensityPredictor:
    def __init__(self, potential, sampleCount, xMin=-1000, xMax=1000):
        self.dU_ = potential.derive();
        self.d2U_ = self.dU_.derive();
        self.X = sp.linspace(xMin, xMax, sampleCount);
        # Set the name and preferred color for this predictor (this is used for generating plots).
        self.name = "persistent";
        self.color = "blue";

    def generateProfile(self, index, memory, diffusion):
        # Find the extrema of dU, which are also the jump take offs.
        dU  = self.dU_(self.X)
        d2U = self.d2U_(self.X)
        I1  = sp.nonzero(d2U[:-1]*d2U[1:]<0)[0] # indices where d2U changes sign between points
        # Indices where d2U is 0 and the sign changes across the 0 point
        I1  = sp.append(I1, sp.intersect1d(sp.nonzero(d2U==0)[0], sp.nonzero(d2U[:-2]*d2U[2:]<0)[0] + 1))
        X1  = self.X[I1]
        dU1 = dU[I1]
        # Find the convex intervals.
        IIm = sp.vstack([[0]+I1.tolist(),I1.tolist()+[len(self.X)-1]]).T # intervals where dU is monotonic
        IIc = IIm[::2] # intervals where U is convex
        Nc  = IIc.shape[0] # number of convex regions
        # Make a list of jumps with their starting and landing point and their starting and landing region.
        jumps = []
        for i1 in I1:
            # Index of the take off region.
            j1 = next( j for j,I in enumerate(IIc) if i1==I[0] or i1==I[-1] )
            # Direction of the jump.
            s  = int(sp.sign(dU[i1]))
            # Index of the landing point.
            d  = dU[i1+s::s] - dU[i1]
            i2 = i1 + s + s*sp.nonzero(sp.logical_and(d2U[i1+s::s][:-1]>0,d[1:]*d[:-1]<0))[0][0]
            # Index of the landing region.
            j2 = next( j for j,I in enumerate(IIc) if I[0]<i2<I[1] )
            jumps.append([i1,j1,i2,j2,s])
        # Solve for the jump rates.
        # First build the corresponding matrix and column vector.
        A  = sp.zeros((2*Nc,2*Nc-2))
        B  = sp.zeros((2*Nc))
        E  = lambda v,v0=sp.sqrt(2*diffusion/memory): memory**2/diffusion*v0*erfi(v/v0);
        for k,(i1,j1,i2,j2,s) in enumerate(jumps):
            # Zero total flux over each region.
            A[j1,k] = -1
            A[j2,k] =  1
            # q=0 at the right end of each region except
            # q=sqrt(memory/2*pi*diffusion for the last region.
            A[Nc+j1,k] =  E(dU[IIc[j1,1]]) - E(dU[i1]);
            A[Nc+j2,k] = -E(dU[IIc[j2,1]]) + E(dU[i2]);
        B[Nc]     = -1
        B[2*Nc-1] = 1
        A = sp.delete(A,[Nc-1,2*Nc-1],0)
        B = sp.delete(B,[Nc-1,2*Nc-1],0)
        J = sp.linalg.solve(A,B) # jump rates
        # Construct density profile.
        # Make an ordered list of jumps in each region.
        jumps_per_region = [ [] for I in IIc ]
        for k,(i1,j1,i2,j2,s) in enumerate(jumps):
            jumps_per_region[j1].append([i1,k,-1]) # outgoing jump in region j1
            jumps_per_region[j2].append([i2,k, 1]) # incoming jump in region j2
        jumps_per_region = [ sorted(jpr) for jpr in jumps_per_region ]
        bins = [-sp.inf];
        vals = [[0,1]];
        for j,jpr in enumerate(jumps_per_region):
            # Iterate over the subregions of the convex region.
            a,b = 0,1 if j==0 else 0
            for i,k,s in jpr:
                bins.append(self.X[i]);
                a -= s*J[k]
                b += s*J[k]*E(dU[i])
                vals.append([a,b]);
            # If things go well the last value pair of the region is [0,0]
            # and describe the density in the following concave region.
            b0 = 1 if j==Nc-1 else 0
            if sp.absolute(vals[-1][0])>1e-4 or sp.absolute(vals[-1][1]-b0)>1e-4:
                updateStatus("There are numerical precision issues.\n", index);
        bins.append(sp.inf);

        return lambda x: self.p(x, memory, diffusion, E, sp.array(bins), sp.array(vals), (sp.sqrt(memory/(2*sp.pi*diffusion))), (-memory / (2 * diffusion)))

    def p(self, x, memory, diffusion, E, bins, vals, q0, q1):
        a,b = vals[sp.digitize(x,bins)-1].T
        return self.d2U_(x)*q0*(a*E(self.dU_(x))+b)*sp.exp(q1*self.dU_(x)**2);
# Create a shortname alias for PersistentDensityPredictor.
p_pred = PersistentDensityPredictor;



# Dictionary of all the predictors. This is used for parsing stringified versions of predictors.
predictorsDict = {"thermaldensitypredictor":ThermalDensityPredictor, "t_pred":ThermalDensityPredictor, "thermal":ThermalDensityPredictor,
                  "singlewellpersistentdensitypredictor":SingleWellPersistentDensityPredictor, "swp_pred":SingleWellPersistentDensityPredictor, "singlewellpersistent":SingleWellPersistentDensityPredictor,
                  "doublewellpersistentdensitypredictor":DoubleWellPersistentDensityPredictor, "dwp_pred":DoubleWellPersistentDensityPredictor, "doublewellpersistent":DoubleWellPersistentDensityPredictor,
                  "persistentdensitypredictor":PersistentDensityPredictor, "p_pred":PersistentDensityPredictor, "persistent":PersistentDensityPredictor};

#==================================================================================================================
#                                              ---DATA VISUALIZATION---
# The classes in this section provide utilities for handling and visualizing data. Principally it contains a class for generating
# histograms from data files, both as bar graphs, and as interpolated functions. It also provides a class for animating histograms
# if their content changes with respect to some variable. There's an additional convenience method included for creating and
# playing said animations.

'''
Class that stores large amounts of data sorted into histograms. In practice, this actually stores a list of histograms, often with
them corresponding to histograms varying over time.
'''
class HistogramGroup:
    '''Loads the histograms stored in a specified file.
       @param filePath: Path of the data file to load into the histogram.'''
    def __init__(self, filePath):
        with open(filePath) as file:
            # Load the bins from the first line of the data file.
            self.bins = np.array(list(map(float, next(file)[:-2].split(','))));
            times = [];
            data = [];
            # Loop through the remaining lines, each of which is just a comma separated list of the bin values.
            for line in file:
                times.append(float(line[(line.index("t=") + 2):line.index(':')]));
                data.append(np.array(list(map(float, line[(line.index(':') + 1):-2].split(',')))));
            # Convert the data into numpy arrays, and store them for later use.
            self.times = np.array(times);
            self.data = np.array(data);

    '''Interpolates the values of a histogram's bins into a set of X and Y coordinates.
       @param index: The histogram to interpolate, with 0 being the first histogram in the file, must be an integer. (defaults to -1, the last histogram)
                     If a collection of indexes are given, this interpolates the average between them.
       @param normalize: True if the histogram's values should be normalized, False to use the raw data. (defaults to True)
       @returns: A tuple of X and Y coordinates that represent the piecewise linear interpolation of the histogram's data, where each X
                 value is the X position midway between the bin's bounds, and the corresponding Y is the value of that bin.'''
    def interpolate(self, index=-1, normalize=True):
        # Compute the points in the middle of each bin's bounds for the X values.
        X = (self.bins[1:] + self.bins[:-1]) / 2;
        # Fetch the data from all the non-overflow bins for the Y values.
        Y = self.data[index][1:-1];
        # If data was pulled from multiple histograms, average the data together.
        if(Y.ndim > 1):
            Y = np.sum(Y, axis=0) / Y.shape[0];
        if(normalize):
            # Compute the normalization constant by taking the integral over the function with the trapazoidal algorithm.
            normalization = np.trapz(Y, dx=(self.bins[1] - self.bins[0]));
            # Normalize the histogram values.
            Y /= normalization;
        return (X, Y);

    '''Retrieves the bin values for histograms within the group.
       @param index: The histogram to retrieve the values of, with 0 being the first histogram in the file, must be an integer. (defaults to -1, the last histogram)
                     If a collection of indexes are given, this averages over all of them.
       @param normalize: True if the histogram's values should be normalized, False to use the raw data. (defaults to False)
       @returns: A list of the bin values for a histogram in order.'''
    def fetchBins(self, index=-1, normalize=False):
        # Fetch the bin data for the specified indexes.
        Y = self.data[index];
        # If data was pulled from multiple histograms, average them together.
        if(Y.ndim > 1):
            Y = np.sum(Y, axis=0) / Y.shape[0];
        if(normalize):
            # Compute the normalization constant by integrating the data with the trapazoidal algorithm.
            normalization = np.trapz(Y, dx=(self.bins[1] - self.bins[0]));
            # Normalize the histogram values.
            Y /= normalization;
        return Y;
# Create a shortname alias for HistogramGroup.
hist_g = HistogramGroup;

'''
Utility class for handling and creating bar graph animations from histogram groups.
'''
class BarGraphAnimator:
    '''Creates a set of bar graphs for each histogram in the group passed into it, and allows for animating through them.
       @param histograms: The histogram group to animate.
       @param smoothing: The number of neighbors in each direction to average each histogram's data with to produce smoother data.
                      If 0, then no averaging occurs. (defaults to 0)'''
    def __init__(self, histograms, smoothing=0):
        self.left = histograms.bins[:-1];
        self.right = histograms.bins[1:];
        self.bottom = np.zeros(len(self.left));
        self.top = histograms.data[0][1:-1];
        self.histograms = histograms;
        self.smoothing = smoothing;

        vertexCount = 5 * len(self.left);
        self.vertices = np.zeros((vertexCount, 2));
        codes = np.ones(vertexCount, int) * path.Path.LINETO;
        codes[0::5] = path.Path.MOVETO;
        codes[4::5] = path.Path.CLOSEPOLY;
        self.vertices[0::5, 0] = self.left;
        self.vertices[0::5, 1] = self.bottom;
        self.vertices[1::5, 0] = self.left;
        self.vertices[1::5, 1] = self.top;
        self.vertices[2::5, 0] = self.right;
        self.vertices[2::5, 1] = self.top;
        self.vertices[3::5, 0] = self.right;
        self.vertices[3::5, 1] = self.bottom;

        self.fig, self.ax = plt.subplots();
        graphPath = path.Path(self.vertices, codes);
        self.patch = patches.PathPatch(graphPath, facecolor='blue', edgecolor='black');
        self.ax.add_patch(self.patch);
        self.ax.set_xlim(self.left[0], self.right[-1]);
        self.ax.set_ylim(0, (np.sum(histograms.data[0]) / 2));
        self.fig.suptitle("Time = 0s");

    '''Internal function for generating animation frames out of histograms.
       @param index: The index of the histogram to generate the frame from.'''
    def animate(self, index):
        self.top = self.histograms.fetchBins(list(range(max(0, (index - self.smoothing)), min(len(self.histograms.data), (index + self.smoothing + 1)))))[1:-1];
        self.vertices[1::5, 1] = self.top;
        self.vertices[2::5, 1] = self.top;
        self.fig.suptitle("Time = " + str(self.histograms.times[index]) + "s");
        return self.patch;

    '''Convenience function for loading an animation from a histogram data file. After loading the data in, this creates
    and displays an animation of bar graphs over time, with each frame representing a single histogram.
    @param filePath: The file to load the histogram data from.
    @param repeat: Whether to loop the animation after it's finished playing. (defaults to False)
    @param smoothing: The number of neighbors in each direction to average each histogram's data with to produce smoother data.
                      If 0, then no averaging occurs. (defaults to 0)
    @param step: The number of histograms to skip between each frame. (defaults to 0)'''
    def viewFromFile(filePath, smoothing=0, repeat=False, step=0):
        histograms = HistogramGroup(filePath);
        animator = BarGraphAnimator(histograms, smoothing);
        a = animation.FuncAnimation(animator.fig, animator.animate, np.arange(1, len(histograms.data), (step + 1)), repeat=repeat);
        plt.show();
        plt.close();
# Create a shortname alias for BarGraphAnimator.
bg_anim = BarGraphAnimator;

# Dictionary of all the animators. This is used for parsing stringified versions of animators.
animatorsDict = {"bargraphanimator":BarGraphAnimator, "bg_anim":BarGraphAnimator, "bargraph":BarGraphAnimator};

#==================================================================================================================
#                                               ---HISTOGRAM TYPES---
# Classes in this section are really just for nicely specifying the types of histograms to use. These are never used internally,
# only stringified and passed along to the C++ side that runs the actual simulation.
# There's two types of histograms currently supported by the C++ side:
# - Linear All the bins are equally spaced over a specified range of values.
# - Custom All the bins are specified by hand, in a single collection at initialization.
# All histograms also have 'overflow' bins, which capture values outside the specified range the histograms cover. These
# sit directly next to the ends of the range, and any values below or above the range go into that respective overflow bin.

'''
Class for specifying the parameters of a linear histogram.
'''
class LinearHistogram:
    '''Creates a new linear histogram with the specified parameters.
       @param minimum: The minimum value of data the histogram should track.
       @param maximum: The maximum value of data the histogram should track.
       @param dx: The spacing to place between consecutive bins. This is only used if the bin count isn't
                  manually specified. (defaults to 0.1)
       @param binCount: The number of bins the histogram should have. These are equally spaced throughout the range.
                        If left as none, the bin count is computed using the bin density. (defaults to None)'''
    def __init__(self, minimum, maximum, dx=0.1, binCount=None):
        self.binMin = minimum;
        self.binMax = maximum;
        if(binCount):
            self.dx = (maximum - minimum) / binCount;
        else:
            self.dx = dx;

    '''Returns the string representation of the histogram. This stringifies the histogram, so it can be passed along to the C++ side
       of the simulation, in a way that it can be properly parsed. For linear histograms, this consists of the type-id "linear", followed
       by the minimum and maximum values of the range the histogram should expect data to be within, and the spacing interval to have
       between consecutive bins dx.
       @returns: The stringified version of the histogram.'''
    def __str__(self):
        return ("\"linear " + str(self.binMin) + " " + str(self.binMax) + " " + str(self.dx) + "\"");
# Create a shortname alias for LinearHistogram.
l_hist = LinearHistogram;

'''
Class for specifying the parameters of a custom binned histogram.
'''
class CustomHistogram:
    '''Creates a new custom histogram with the specified bins.
       @param bins: A collection containing the bounds of the histogram's bins, from lowest to highest.'''
    def __init__(self, bins):
        self.bins = bins;

    '''Returns the string representation of the histogram. This stringifies the histogram, so it can be passed along to the C++ side
       of the simulation, in a way that it can be properly parsed. For custom histograms, this consists of the type-id "custom", followed
       a space delimited list of the bounds of the bins, from least to greatest.
       @returns: The stringified version of the histogram.'''
    def __str__(self):
        return ("\"custom " + " ".join(map(str, self.bins)) + " \"");
# Create a shortname alias for CustomHistogram.
c_hist = CustomHistogram;

# Dictionary of all the histograms. This is used for parsing stringified versions of histograms.
histogramsDict = {"linearhistogram":LinearHistogram, "l_hist":LinearHistogram, "linear":LinearHistogram,
                 "customhistogram":CustomHistogram, "c_hist":CustomHistogram, "custom":CustomHistogram};

#==================================================================================================================
#                                           ---SIMULATION EXECUTION---
# Code in this section is used to create, execute, and generally manage simulations.

'''TODO make this method do fancier stuff!'''
def updateStatus(status, index=-1):
    if(index == -1):
        print(status, end='');
    else:
        print(str(index) + ":\t" + str(status), end='');

'''TODO COMMENTS'''
def decodeHistogram(params, name):
    params = params.split();
    if(len(params) == 0):
        raise ValueError("Missing histogram type for " + name + ". Usage:\n    " + name + " <type> <parameters...>");
    # Convert the histogram identifier to lowercase for parsing.
    params[0] = params[0].lower();
    if(params[0] == "linear"):
        if(len(params) > 5):
            raise ValueError("Too many arguments passed for linear histogram. Expected 3, received " + (len(params) - 1));
        return LinearHistogram(*tuple(map(float, params[1:4])));
    elif(params[0] == "custom"):
        if(len(params) < 4):
            print("Warning " + name + ": Having less than 2 bins in a custom histogram can cause recording instability.");
        return CustomHistogram(tuple(map(float, params[1:])));

'''TODO COMMENTS'''
def decodePotential(params):
    params = params.split();
    if(len(params) == 0):
        raise ValueError("Missing potential type. Usage:\n   potential <type> <parameters...>");
    # Convert the potential identifier to lowercase for parsing.
    params[0] = params[0].lower();
    return PolyFunc([0, 0.02, -0.025, 0, 0.001]);#TODO TODO TODO TODO

'''TODO COMMENTS'''
def decodePrediction(params, potential):
    params = params.split();
    if(len(params) == 0):
        raise ValueError("Missing potential type. Usage:\n   potential <type> <parameters...>");
    # Convert the potential identifier to lowercase for parsing.
    params[0] = params[0].lower();
    if(params[0] == "thermal"):
        if(len(params) > 4):
            raise ValueError("Too many arguments passed for thermal predictor. Expected 3, received " + (len(params) - 1));
        return ThermalDensityPredictor(potential, *tuple(map(float, params[1:4])));
    elif(params[0] == "singlewellpersistent"):
        if(len(params) > 1):
            raise ValueError("Too many arguments passed for single well persistent predictor. Expected 0, received " + (len(params) - 1));
        return SingleWellPersistentDensityPredictor(potential);
    elif(params[0] == "doublewellpersistent"):
        if(len(params) > 4):
            raise ValueError("Too many arguments passed for double well persistent predictor. Expected 3, received " + (len(params) - 1));
        return DoubleWellPersistentDensityPredictor(potential, *tuple(map(float, params[1:4])));
    elif(params[0] == "persistent"):
        if(len(params) > 4):
            raise ValueError("Too many arguments passed for persistent predictor. Expected 3, received " + (len(params) - 1));
        return PersistentDensityPredictor(potential, *tuple(map(float, params[1:4])));

'''TODO COMMENTS'''
def runFromFile(file):
    # Assign default values for all the parameters. (these must mirror the ones in the C++ side)
    potential = [None];
    outputFile = ["./results"];
    posRecorder = [None];
    forceRecorder = [None];
    noiseRecorder = [None];
    particleCount = [100];
    duration = [20];
    timestep = [0.05];
    diffusion = [1];
    memory = [1];
    dataDelay = [10];
    startBoundLeft = [-5];
    startBoundRight = [5];
    activeForcesMean = [0];
    activeForcesStddev = [0.2];
    noiseMean = [0];
    noiseStddev = [1];
    predictions = [None];
    # This parameter is only used by the Python wrapper.
    maxThreadCount = 1;

    # List of flags for storing whether variables have already been set. Used for throwing warnings when parameters are set twice.
    setFlags = [False] * 19;

    # Parse the parameter file line by line.
    with open(file) as file:
        for line in file:
            # Ignore comment lines starting with an '#' symbol.
            if(line.startswith('#')):
                continue;
            params = shlex.split(line);
            if(len(params) == 0):
                print("No parameters detected. Consult the Readme for instructions on using parameter files.");
                break;
            # Convert the parameter identifier to lowercase for parsing.
            params[0] = params[0].lower();
            if(params[0] == "potential"):
                potential = [decodePotential(p) for p in params[1:]];
                if(setFlags[0]):
                    print("Warning: potential was set multiple times.");
                else:
                    setFlags[0] = True;
            elif(params[0] == "outputfile"):
                outputFile = [s.strip() for s in params[1:]];
                if(setFlags[1]):
                    print("Warning: outputFile was set multiple times.");
                else:
                    setFlags[1] = True;
            elif(params[0] == "posrecorder"):
                posRecorder = [decodeHistogram(p, "posRecorder") for p in params[1:]];
                if(setFlags[2]):
                    print("Warning: posRecorder was set multiple times.");
                else:
                    setFlags[2] = True;
            elif(params[0] == "forcerecorder"):
                forceRecorder = [decodeHistogram(p, "forceRecorder") for p in params[1:]];
                if(setFlags[3]):
                    print("Warning: forceRecorder was set multiple times.");
                else:
                    setFlags[3] = True;
            elif(params[0] == "noiserecorder"):
                noiseRecorder = [decodeHistogram(p, "noiseRecorder") for p in params[1:]];
                if(setFlags[4]):
                    print("Warning: noiseRecorder was set multiple times.");
                else:
                    setFlags[4] = True;
            elif(params[0] == "particlecount"):
                particleCount = tuple(map(int, params[1:]));
                if(setFlags[5]):
                    print("Warning: particleCount was set multiple times.");
                else:
                    setFlags[5] = True;
            elif(params[0] == "duration"):
                duration = tuple(map(float, params[1:]));
                if(setFlags[6]):
                    print("Warning: duration was set multiple times.");
                else:
                    setFlags[6] = True;
            elif(params[0] == "timestep"):
                timestep = tuple(map(float, params[1:]));
                if(setFlags[7]):
                    print("Warning: timestep was set multiple times.");
                else:
                    setFlags[7] = True;
            elif(params[0] == "diffusion"):
                diffusion = tuple(map(float, params[1:]));
                if(setFlags[8]):
                    print("Warning: diffusion was set multiple times.");
                else:
                    setFlags[8] = True;
            elif(params[0] == "memory"):
                memory = tuple(map(float, params[1:]));
                if(setFlags[9]):
                    print("Warning: memory was set multiple times.");
                else:
                    setFlags[9] = True;
            elif(params[0] == "datadelay"):
                dataDelay = tuple(map(float, params[1:]));
                if(setFlags[10]):
                    print("Warning: dataDelay was set multiple times.");
                else:
                    setFlags[10] = True;
            elif(params[0] == "startboundleft"):
                startBoundLeft = tuple(map(float, params[1:]));
                if(setFlags[11]):
                    print("Warning: startBoundLeft was set multiple times.");
                else:
                    setFlags[0] = True;
            elif(params[0] == "startboundright"):
                startBoundRight = tuple(map(float, params[1:]));
                if(setFlags[12]):
                    print("Warning: startBoundRight was set multiple times.");
                else:
                    setFlags[12] = True;
            elif(params[0] == "activeforcesmean"):
                activeForcesMean = tuple(map(float, params[1:]));
                if(setFlags[13]):
                    print("Warning: activeForcesMean was set multiple times.");
                else:
                    setFlags[13] = True;
            elif(params[0] == "activeforcesstddev"):
                activeForcesStddev = tuple(map(float, params[1:]));
                if(setFlags[14]):
                    print("Warning: activeForcesStddev was set multiple times.");
                else:
                    setFlags[14] = True;
            elif(params[0] == "noisemean"):
                noiseMean = tuple(map(float, params[1:]));
                if(setFlags[15]):
                    print("Warning: noiseMean was set multiple times.");
                else:
                    setFlags[15] = True;
            elif(params[0] == "noisestddev"):
                noiseStddev = tuple(map(float, params[1:]));
                if(setFlags[16]):
                    print("Warning: noiseStddev was set multiple times.");
                else:
                    setFlags[16] = True;
            elif(params[0] == "maxthreadcount"):
                if(len(params) != 2):
                    raise ValueError("MaxThreadCount can only take one value, however " + str(len(params) - 1) + " were given.");
                maxThreadCount = int(params[1]);
                if(setFlags[17]):
                    print("Warning: maxThreadCount was set multiple times.");
                else:
                    setFlags[17] = True;
            elif(params[0] == "predictions"):
                predictions = [[decodePrediction(params[i+1], potential[min((len(potential)-1), i)]) for i in range(len(params)-1)]];
                if(setFlags[18]):
                    print("Warning: Predictions was set multiple times.");
                else:
                    setFlags[18] = True;
            else:
                raise ValueError("Unknown parameter: " + params[0]);

    if(not potential):
        raise ValueError("Cannot run simulation with an unspecified potential.");
    return runSimulationMulti(potential, outputFile, predictions, posRecorder, forceRecorder, noiseRecorder, particleCount, duration, timestep, diffusion, memory, dataDelay, list(zip(startBoundLeft, startBoundRight)), list(zip(activeForcesMean, activeForcesStddev)), list(zip(noiseMean, noiseStddev)), maxThreadCount);

'''TODO COMMENTS'''
def runSimulationMulti(potential, outputFile=["result"], predictions=[None], posRecorder=[None], forceRecorder=[None], noiseRecorder=[None], particleCount=[100], duration=[20], timestep=[0.05], diffusion=[1], memory=[1], dataDelay=[10], startBounds=[(-5,5)], activeForces=[(0,0.2)], noise=[(0,1)], maxThreadCount=0):
    args = [potential, outputFile, predictions, posRecorder, forceRecorder, noiseRecorder, particleCount, duration, timestep, diffusion, memory, dataDelay, startBounds, activeForces, noise];
    names = ["potential", "outputFile", "predictions", "posRecorder", "forceRecorder", "noiseRecorder", "particleCount", "duration", "timestep", "diffusion", "memory", "dataDelay", "startBounds", "activeForces", "noise"];
    simCount = max(*[len(arg) for arg in args]);
    for i in range(len(args)):
        length = len(args[i]);
        if(length == 1):
            args[i] = args[i] * simCount;
        elif(length != simCount):
            raise ValueError("Argument lists must all be the same size. " + names[i] + " was only " + str(length) + " long, when " + str(simCount) + " was expected.");

    if(simCount < 1):
        print("No simulations to run... Exiting.");
    elif(simCount == 1):
        runSimulation(*[args[j][0] for j in range(len(args))]);
    else:
        # If the user specified 0 for maxThreadCount, change it to run one thread per processor.
        if(maxThreadCount == 0):
            maxThreadCount = os.cpu_count();
        index = 0;
        completed = 0;
        finished = [];
        threads = [];
        params = [];
        simLock = threading.Condition(threading.RLock());
        with simLock:
            while(completed < simCount):
                if((index < simCount) and ((len(threads) - completed) < maxThreadCount)):
                    params.append([args[j][index] for j in range(len(args))]);
                    threads.append(threading.Thread(target=runSimulation, args=[*params[-1], index, finished, simLock]));
                    threads[-1].start();
                    index += 1;
                elif(len(finished) > 0):
                    i = finished.pop(0);
                    threads[i].join();
                    exportResults(*params[i], i);
                    completed += 1;
                else:
                    simLock.wait(10);

'''TODO COMMENTS'''#(NOTE THAT THIS CANNOT PARSE NESTED PIECEWISE OR PERIODIC FUNCTIONS! TODO?)
def runSimulation(potential, outputFile="result", predictions=None, posRecorder=None, forceRecorder=None, noiseRecorder=None, particleCount=100, duration=20, timestep=0.05, diffusion=1, memory=1, dataDelay=10, startBounds=(-5,5), activeForces=(0,0.2), noise=(0,1), index=-1, finished=None, simLock=None):
    updateStatus("Initializing...\n", index);

    # Get the current directory of the program.
    progDir = str(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))));

    # Build a command to run the simulation on the C++ side.
    command = progDir + "/simulate.exe " + str(potential);
    if(outputFile):
        command += " -of ";
        # Ensure that the file path is wrapped in quotes if it isn't already.
        if(outputFile[0] != "\""):
            command += "\"";
        command += str(outputFile);
        if(outputFile[-1] != "\""):
            command += "\"";
    if(posRecorder):
        command += " -pr " + str(posRecorder);
    if(forceRecorder):
        command += " -fr " + str(forceRecorder);
    if(noiseRecorder):
        command += " -nr " + str(noiseRecorder);
    if(particleCount):
        command += " -n " + str(particleCount);
    if(duration):
        command += " -t " + str(duration);
    if(timestep):
        command += " -dt " + str(timestep);
    if(diffusion):
        command += " -d " + str(diffusion);
    if(memory):
        command += " -m " + str(memory);
    if(dataDelay):
        command += " -dd " + str(dataDelay);
    if(startBounds):
        command += " -sb " + str(startBounds[0]) + " " + str(startBounds[1]);
    if(activeForces):
        command += " -af " + str(activeForces[0]) + " " + str(activeForces[1]);
    if(noise):
        command += " -no " + str(noise[0]) + " " + str(noise[1]);

    # Create the output directory if it doesn't exist (C++ can't export to non-existant directories)
    outputDir = progDir + "/" + str(os.path.dirname(outputFile));
    if(not os.path.exists(outputDir)):
        updateStatus("Creating output directory: " + outputDir + '\n', index);
        # If this is being run in parallel, synchronize on simLock to avoid multiple runs trying to create the same directory.
        if(simLock):
            with(simLock):
                if(not os.path.exists(outputDir)):
                    os.makedirs(outputDir);
        else:
            os.makedirs(outputDir);

    # Run the simulaion.
    updateStatus("Running Simulation...\n", index);
    with sproc.Popen(command, stdout=sproc.PIPE, stderr=sproc.STDOUT, bufsize=1, universal_newlines=True) as proc:
        stdBuffer = "Simulation Progress: 0%\n";
        while((stdBuffer != "") or (proc.poll() == None)):
            if(stdBuffer):
                updateStatus(stdBuffer, index);
            stdBuffer = proc.stdout.readline();
        updateStatus("Simulation finished with exit code: " + str(proc.poll()) + '\n', index);
        if(proc.poll() != 0):
            raise sproc.CalledProcessError(proc.poll(), command);

    if(simLock):
        # If this is part of a parallel run, notify the main thread that this simulation has finished.
        with(simLock):
            finished.append(index);
            simLock.notifyAll();
        return 0;
    else:
        # If this is a stand-alone run, export the results.
        return exportResults(potential, outputFile, predictions, posRecorder, forceRecorder, noiseRecorder, particleCount, duration, timestep, diffusion, memory, dataDelay, startBounds, activeForces, noise);

'''TODO COMMENTS'''
def exportResults(potential, outputFile, predictors, posRecorder, forceRecorder, noiseRecorder, particleCount, duration, timestep, diffusion, memory, dataDelay, startBounds, activeForces, noise, index=-1):
    updateStatus("Exporting results...\n", index);
    if(posRecorder):
        n = ((posRecorder.binMax - posRecorder.binMin) / posRecorder.dx) * 100;
        X = sp.linspace(posRecorder.binMin, posRecorder.binMax, n);
        ax = plt.gca();
        ax.set_xlabel("position");
        ax.set_ylabel("particle density");
        legend = [];

        # Create the prediction profiles.
        predictions = [predictor.generateProfile(index, memory, diffusion) for predictor in predictors];
        for i in range(len(predictions)):
            ax.plot(X, predictions[i](X), predictors[i].color);
            legend.append(patches.Patch(color=predictors[i].color, label=predictors[i].name));

        # Create the results profile.
        posXY = HistogramGroup(str(outputFile) + ".pos").interpolate();
        ax.plot(posXY[0], posXY[1], "red");
        legend.append(patches.Patch(color="red", label="result"));

        ax = ax.twinx();

        # Create the potential profile.
        ax.set_ylabel("potential");
        ax.plot(X, potential(X), "black");
        legend.append(patches.Patch(color="black", label="potential"));

        plt.legend(handles=legend, loc=1);
        plt.savefig(str(outputFile) + "P.png", fmt=".png", dpi=200);
        plt.close();
    if(forceRecorder):
        pass;#TODO
    if(noiseRecorder):
        pass;#TODO

#==================================================================================================================
# ---FUNCTION CREATION--- TODO EVERYTHING UNDER THIS NEEDS BETTER COMMENTS

'''
Class encapsulating a polynomial, it's intialized with coeffecients in ascending order (lowest order coeffecient
to highest order), then can be called like any normal function.
'''
class PolyFunc:
    def __init__(self, coeffecients):
        self.c = np.array(coeffecients, dtype=np.float64);

    def __len__(self):
        return len(self.c);

    def __getitem__(self, index):
        return self.c[index];

    def __str__(self):
        return ("\"poly " + str(self.c)[1:-1] + '\"');

    def __call__(self, x):
        if((type(x) != np.ndarray) or (x.ndim == 0)):
            x = np.array([x], dtype=np.float64);
        return np.sum((self.c * np.power(x[:, None], np.arange(len(self)))), axis=1);

    def negate(self):
        return PolyFunc(-self.c);

    def derive(self):
        return PolyFunc(self.c[1:] * np.arange(len(self))[1:]);

    def integD(self, a, b):
        powerRange = np.arange(len(self)) + 1;
        return np.sum((self.c * (np.power(a[:, None], powerRange) - np.power(b[:, None], powerRange)) / powerRange), axis=1);

    def integI(self, C=0):
        return PolyFunc(np.array([C], dtype=self.c.dtype).append(np.multiply(self.c, np.reciprocal(np.arange(len.self) + 1))));

'''
Class encapsulating a periodic function, it's intialized by a generator function and a region. The peridic
function is then generated by repeating the specified region of the generator function over and over. The start
of the region on the generator is always mapped to 0 on the periodic function, but otherwise lengths are preserved.
The period of the periodic function is the length of the region specified.
'''
class PeriodicFunction:
    def __init__(self, function, start, stop):
        self.generator = function;
        self.start = start;
        self.stop = stop;
        self.period = (stop - start);
        self.area = function.integD(start, stop);

    def __str__(self):
        return ("\"periodic '" + str(self.generator)[1:-1] + "' " + str(self.start) + " " + str(self.stop) + "\"");

    def __call__(self, x):
        return self.generator((x % self.period) + self.start);

    def negate(self):
        return PeriodicFunc(self.generator.negate(), self.start, self.stop);

    # Note that even though the derivative is probably undefined at the boundaries, this will always return
    # the value generator'(offset) at both boundaries of periodicity instead. TODO
    def derive(self):
        return PeriodicFunction(self.generator.derive(), self.start, self.stop);

    def integD(self, a, b):
        if(b < a):
            return -self.integD(a, b);
        elif(b == a):
            return 0;

        periods = (b // period) - (a // period);
        a = a % self.period;
        b = b % self.period;
        if(periods == 0):
            return self.generator.integD((a + self.start), (b + self.start));
        else:
            return self.generator.integD((a + self.start), self.stop) + self.generator.integD(self.start, (b + self.start)) * (area * periods);

'''
Class encapsulating a piecewise function, it's intialized with functions, and the boundary points between them.
After initialization it can be called like any normal function.
Functions must be passed in left to right, as must the bounds. There is also an additional field to specify
the direction to evaulate the piecewise function from at the boundaries. False indicates to evaulate it from the
left, True to evaulate it from the right. If left unused, by default boundaries are always evaulated from the
right.
'''
class PieceFunc:
    def __init__(self, functions, bounds, directions=None):
        if(len(functions) != (len(bounds) + 1)):
            raise ValueError("There must be exactly 1 less bound than there are functions.");
        for i in range(1, len(bounds)):
            if(bounds[i] <= bounds[i-1]):
                raise ValueError("Bounds must be in increasing order, and no bounds can be the same.");
        self.functions = functions;
        self.bounds = np.array(bounds, dtype=np.float64);
        if(directions):
            self.directions = directions;
        else:
            self.directions = [True]*len(bounds);

    def __len__(self):
        return len(self.functions);

    def __getitem__(self, index):
        return self.functions[index];

    def __str__(self):
        s = "\"piece ";
        for i in range(len(self.bounds)):
            s += "\'" + str(self.functions[i]) + "\' " + str(self.bounds[i]) + " "
            if(self.directions[i]):
                s += "1 ";
            else:
                s += "0 ";
        s += "\'" + str(self.functions[-1]) + "\'\"";
        return s;

    def __call__(self, x):
        conditions = ([x < self.bounds[0]] if self.directions[0] else [x <= self.bounds[0]]);
        for i in range(len(self.bounds)):
            conditions.append((x >= self.bounds[i]) if self.directions[i] else (x > self.bounds[i]));
        #conditions = [(x <= -2), (x > -2), (x > -1), (x > 0), (x > 1), (x > 2)]; #TODO THIS ONLY WORKS WHEN >= is used. SOMETHING SERIOUSLY WRONG HERE
        return np.piecewise(x, conditions, self.functions);

    def negate(self):
        negated = [func.negate() for func in self.functions];
        return PieceFunc(negated, cp.deepcopy(self.bounds), cp.deepcopy(self.directions));

    # Note that even though the derivative is probably undefined at the boundaries, this will always return
    # the derivative of the function that is evaulated at the boundary instead. TODO
    def derive(self):
        derivatives = [func.derive() for func in self.functions];
        return PieceFunc(derivatives, cp.deepcopy(self.bounds), cp.deepcopy(self.directions));

    def integD(self, a, b):
        if(a > b):
            return -integD(b, a);
        elif(a == b):
            return 0;

        min = None;
        total = 0;
        for i in range(len(self.bounds)):
            if(a < self.bounds[i]):
                min = a;
            if(min):
                if(b < self.bounds[i]):
                    total += self.functions[i].integD(a, b);
                    break;
                else:
                    total += self.functions[i].integD(a, self.bounds[i]);
                    min = self.bounds[i];
        else:
            total += self.functions[-1].integD(self.bounds[-1], b);
        return total;

'''WRITE BETTER DIRECTIONS TODO
each parameter is [x, f(x), f'(x), f''(x)],
then all of these go together in a list.'''
class PiecewiseCustom2ndOrder:
    def __init__(self, points, check=False):
        splines = [];
        splines.append(create1Way2ndOrderSpline(*points[0], check));
        for i in range(len(points) - 1):
            splines.append(create2Way2ndOrderSpline(*points[i], *points[i+1], check));
        splines.append(create1Way2ndOrderSpline(*points[-1], check));
        self.function = PieceFunc(list(zip(*splines))[0], list(zip(*points))[0]);
        if(check):
            for i in range(len(list(zip(*splines))[1])):
                if(splines[i][1]):
                    print("Instability present on spline component " + str(i) + "!");

    def __call__(self, x):
        return self.function(x);

    def __str__(self):
        return str(self.function);

    def derive(self):
        return self.function.derive();

    def integ(self, C):
        return self.function.integ(C);

#                            ax, ay, ady, ad2y, bx, by, bdy, bd2y
def create2Way2ndOrderSpline(a, A, S, U, b, B, T, V, check=-1):
    # Pre-calculate the powers of delta.
    delta1 = b-a;
    delta2 = delta1**2;
    delta3 = delta1**3;
    # Compute the parameterization coeffecients.
    D = (a*U) - (((A*delta1) - ((7*a) - b)*((S*delta1) + (2*A))) / delta2);
    G = (b*V) + (((B*delta1) - ((7*b) - a)*((T*delta1) - (2*B))) / delta2);
    E = U + ((6 * ((S*delta1) + (2*A))) / delta2);
    H = V - ((6 * ((T*delta1) - (2*B))) / delta2);

    # Create the intermediate splining values.
    i1 = E / 2;
    j1 = H / 2;
    i3 = -(i1 * (a**2)) + (D * a) + A;
    j3 = -(j1 * (b**2)) + (G * b) + B;

    # Construct the spline polynomial.
    c0 = -(-((b**3) * i3) + ((a**3) * j3)) / delta3;
    c1 = (-((b**3) * D) + ((a**3) * G) - (3 * (b**2) * i3) + (3 * (a**2) * j3)) / delta3;
    c2 = -(-((b**3) * i1) + ((a**3) * j1) - (3 * (b**2) * D) + (3 * (a**2) * G) - (3 * b * i3) + (3 * a * j3)) / delta3;
    c3 = (-(3 * (b**2) * i1) + (3 * (a**2) * j1) - (3 * b * D) + (3 * a * G) - i3 + j3) / delta3;
    c4 = -(-(3 * b * i1) + (3 * a * j1) - D + G) / delta3;
    c5 = (-i1 + j1) / delta3;
    spline = PolyFunc((c0, c1, c2, c3, c4, c5));

    # Ensure there's no spurious extrema up to the specified degree.
    instabilities = [];
    for i in range(min(4, check + 1)): #It's only useful to check up to the 4th derivative of a 5th order polynomial. Higher order derivatives are just constant.
        Dspline = spline.derive();
        extrema = np.roots(Dspline.c);
        instabilities.append(len(np.where((extrema > a) & (extrema < b))[0]) == 0);

    # Return the spline and it's ordered instabilities.
    return (spline, instabilities);

#                            ax, ay, ady, ad2y, direction: false-left, true-right
def create1Way2ndOrderSpline(a, A, S, U, direction=False, check=-1):
    # Create the intermediate splining values.
    i1 = U / 2;
    i2 = S - (a * U);

    # Construct the spline polynomial.
    c0 = -(i1 * (a**2)) - (i2 * a) + A;
    c1 = i2;
    c2 = i1;
    spline = PolyFunc((c0, c1, c2));

    # Ensure there's no spurious extrema up to the specified degree.
    instabilities = [];
    for i in range(min(1, check + 1)): #It's only useful to check the 1st derivative of a 2nd order polynomial. Higher order derivatives are just constant.
        Dspline = spline.derive();
        extrema = np.roots(Dspline.c);
        if(direction):
            instabilities.append(len(np.where(extrema > a)[0]) == 0);
        else:
            instabilities.append(len(np.where(extrema < b)[0]) == 0);

    # Return the spline and it's ordered instabilities.
    return (spline, instabilities);

# Deriving and integrating these wrapper functions, don't return the same type. type(DoubleWellFunc.derive) != DoubleWellFunc
'''
Class encapsulating a double well potential. At initialization the relevant properties of the well
are specified, and afterwards it's callable as any normal function.
TODO This is not the most stable, we added some warnings to alert us about instabilities being
present, but there might be more that we haven't found, and at the least, the warnings don't
give the most accurate numbers, because there's ALOT of really complicated feedback loops in
creating this thing...
'''
class DoubleWellFunc:
    def __init__(self, points, fa, f1b, fc, f2c, f1d, fe):
        self.stable = true;
        bc = createSpline12200(points[2], points[1], fc, 0, f1b, f2c, 0);
        cd = createSpline12200(points[2], points[3], fc, 0, f1d, f2c, 0);
        ab = createSpline22100(points[1], points[0], bc(points[1]), fa, f1b, 0, 0);
        de = createSpline22100(points[3], points[4], cd(points[3]), fe, f1d, 0, 0);

        ab1 = ab.derive();
        ab2 = ab1.derive();
        ab3 = ab2.derive();
        ab4 = ab3.derive();
        aa = createSpline11111(points[0], ab(points[0]), ab1(points[0]), ab2(points[0]), ab3(points[0]), ab4(points[0]));

        de1 = de.derive();
        de2 = de1.derive();
        de3 = de2.derive();
        de4 = de3.derive();
        ee = createSpline11111(points[4], de(points[4]), de1(points[4]), de2(points[4]), de3(points[4]), de4(points[4]));

        if(f1b <= 0):
            print("WARNING: Having non-positive slope at b causes instability");
            self.stable = false;
        if(f1d >= 0):
            print("WARNING: Having non-negative slope at d causes instability");
            self.stable = false;

        slopeBAmax = 1.5 * (bc(points[1]) - fa) / (points[1] - points[0]);
        slopeDEmin = 1.5 * (cd(points[3]) - fe) / (points[3] - points[4]);
        if(f1b >= slopeBAmax):
            print("WARNING: Having derivative more than 1.5*slope{BA} at b causes instability. max=" + str(slopeBAmax));
            self.stable = false;
        if(f1b <= slopeDEmin):
            print("WARNING: Having derivative less than 1.5*slope{DE} at d causes instability. min=" + str(slopeDEmin));
            self.stable = false;

        concavityCBmax = -2 * f1b / (points[2] - points[1]);
        concavityCDmax = -2 * f1d / (points[2] - points[3]);
        if(f2c >= concavityCBmax):
            print("WARNING: Having 2nd derivative more than -2*concavity{CB} at c causes instability. max=" + str(concavityCBmax));
            self.stable = false;
        if(f2c >= concavityCDmax):
            print("WARNING: Having 2nd derivative more than -2*concavity{CD} at c causes instability. max=" + str(concavityCDmax));
            self.stable = false;

        self.function = PieceFunc([aa,ab, bc, cd, de,ee], points);

    def __call__(self, x):
        return self.function(x);

    def __str__(self_):
        return str(self.function);

    def derive(self):
        return self.function.derive();

    def integ(self, C):
        return self.function.integ(C);

'''
Creates a quartic polynomial function with with a specified value, 1st, 2nd, 3rd,
and 4th derivative at a point
@param a The point to specify all it's properties at.
@param fa The value of the function at a.
@param f1a The value of the function's 1st derivative at a.
@param f2a The value of the function's 2nd derivative at a.
@param f3a The value of the function's 3nd derivative at a.
@param f4a The value of the function's 4nd derivative at a.
@return A PolyFunc that contains the specified properties.
'''
def createSpline11111(a, fa, f1a, f2a, f3a, f4a):
#This just interpolates it as a parabola for simplicity, TODO
    c4 = 0;
    c3 = 0;
    c2 = f2a / 2;
    c1 = f1a - (f2a * a);
    c0 = fa - (c2 * (a**2)) - (c1 * a);

    return PolyFunc((c0,c1,c2,c3,c4));

'''
Creates a quartic polynomial function with 1 specified value, 2 specified
1st derivatives, and 2 2nd derivatives.
@param a The 1st point to specify values at.
@param b The 2nd point to specify values at.
@param fa The value of the function at a.
@param f1a The value of the function's 1st derivative at a.
@param f1b The value of the function's 1st derivative at b.
@param f2a The value of the function's 2nd derivative at a.
@param f2b The value of the function's 2nd derivative at b.
@return A PolyFunc that contains the specified properties.
@raises ValueError If a and b are the same points.
'''
def createSpline12200(a, b, fa, f1a, f1b, f2a, f2b):
    if(a == b):
        raise ValueError("Cannot use the same point for both endpoints of a spline.");
    # Calculate the difference between the endpoints.
    dx = a-b;

    # Calculate the intermediary parameters of the spline.
    A0 = ((f2a * dx) - (2 * f1a)) / (dx**3);
    B0 = ((f2b * dx) + (2 * f1b)) / (dx**3);
    if(A0 == 0):
        c = 1;
    else:
        c = a - (f1a / (A0 * (dx**2)));
    if(B0 == 0):
        d = 1;
    else:
        d = b - (f1b / (B0 * (dx**2)));
    f = fa - ((a**4) * (A0 + B0) / 4) + ((a**3) * ((A0 * ((2 * b) + c)) + (B0 * ((2 * a) + d))) / 3) - ((a**2) * ((A0 * ((b**2) + (2 * b * c))) + (B0 * ((a**2) + (2 * a * d)))) / 2) + (a * (A0 * c * (b**2)) + (B0 * d * (a**2)));

    # Compute the polynomial coeffecients.
    c0 = f;
    c1 = -(A0 * c * (b**2)) - (B0 * d * (a**2));
    c2 = ((A0 * ((b**2) + (2 * b * c))) + (B0 * ((a**2) + (2 * a * d)))) / 2;
    c3 = -((A0 * ((2 * b) + c)) + (B0 * ((2 * a) + d))) / 3;
    c4 = (A0 + B0) / 4;

    return PolyFunc((c0,c1,c2,c3,c4));

'''
Creates a quartic polynomial function with 2 specified values, 2 specified
1st derivatives, and 1 2nd derivative.
@param a The 1st point to specify values at.
@param b The 2nd point to specify values at.
@param fa The value of the function at a.
@param fb The value of the function at b.
@param f1a The value of the function's 1st derivative at a.
@param f1b The value of the function's 1st derivative at b.
@param f2a The value of the function's 2nd derivative at a.
@return A PolyFunc that contains the specified properties.
@raises ValueError If a and b are the same points.
'''
def createSpline22100(a,b,fa,fb,f1a,f1b,f2a):
    if(a == b):
        raise ValueError("Cannot use the same point for both endpoints of a spline.");
    # Calculate the difference between the endpoints.
    dx = a-b;

    # Calculate the intermediary parameters of the spline.
    A0 = +((fa * dx) - (a * ((f1a * dx) - (3 * fa)))) / (dx**4);
    B0 = -((fb * dx) - (b * ((f1b * dx) + (3 * fb)))) / (dx**4);
    if(A0 == 0):
        c = 1;
    else:
        c = +((f1a * dx) - (3 * fa)) / (A0 * (dx**4));
    if(B0 == 0):
        d = 1;
    else:
        d = -((f1b * dx) + (3 * fb)) / (B0 * (dx**4));
    f = (f2a / (2 * (dx**2))) - (3 * A0 * (((c * dx) + (c * a) + 1) / dx));

    # Compute the polynomial coeffcients.
    c0 = (f * (a**2) * (b**2)) - (A0 * (b**3)) - (B0 * (a**3));
    c1 = ((b**2) * A0 * (3 - (b * c))) + ((a**2) * B0 * (3 - (a * d))) - (2 * f * ((a * (b**2)) + (b * (a**2))));
    c2 = -(b * A0 * (3 - (3 * b * c))) - (a * B0 * (3 - (3 * a * d))) + (f * ((b**2) + (4 * a * b) + (a **2)));
    c3 = (A0 * (1 - (3 * b * c))) + (B0 * (1 - (3 * a * d))) - (2 * f * (a + b));
    c4 = (A0 * c) + (B0 * d) + f;

    return PolyFunc((c0,c1,c2,c3,c4));

'''
Creates a quartic polynomial function with 2 specified values, 2 specified
1st derivatives, and 1 3rd derivative.
@param a The 1st point to specify values at.
@param b The 2nd point to specify values at.
@param fa The value of the function at a.
@param fb The value of the function at b.
@param f1a The value of the function's 1st derivative at a.
@param f1b The value of the function's 1st derivative at b.
@param f3a The value of the function's 3rd derivative at a.
@return A PolyFunc that contains the specified properties.
@raises ValueError If a and b are the same points.
'''
def createSpline22010(a,b,fa,fb,f1a,f1b,f3a):
    if(a == b):
        raise ValueError("Cannot use the same point for both endpoints of a spline.");
    # Calculate the difference between the endpoints.
    dx = a-b;

    # Calculate the intermediary parameters of the spline.
    A0 = +((fa * dx) - (a * ((f1a * dx) - (3 * fa)))) / (dx**4);
    B0 = -((fb * dx) - (b * ((f1b * dx) + (3 * fb)))) / (dx**4);
    if(A0 == 0):
        c = 1;
    else:
        c = +((f1a * dx) - (3 * fa)) / (A0 * (dx**4));
    if(B0 == 0):
        d = 1;
    else:
        d = -((f1b * dx) + (3 * fb)) / (B0 * (dx**4));
    f = (((f3a / 6) - (B0 * (1 - (d * a))) - (A0 * (1 - (b * c)))) / (2 * dx)) - (c * A0)

    # Compute the polynomial coeffcients.
    c0 = (f * (a**2) * (b**2)) - (A0 * (b**3)) - (B0 * (a**3));
    c1 = ((b**2) * A0 * (3 - (b * c))) + ((a**2) * B0 * (3 - (a * d))) - (2 * f * ((a * (b**2)) + (b * (a**2))));
    c2 = -(b * A0 * (3 - (3 * b * c))) - (a * B0 * (3 - (3 * a * d))) + (f * ((b**2) + (4 * a * b) + (a **2)));
    c3 = (A0 * (1 - (3 * b * c))) + (B0 * (1 - (3 * a * d))) - (2 * f * (a + b));
    c4 = (A0 * c) + (B0 * d) + f;

    return PolyFunc((c0,c1,c2,c3,c4));
