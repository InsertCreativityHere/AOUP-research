
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib import path as path
from matplotlib import animation as animation
import copy as cp;
import numpy as np;
import scipy as sp;
import subprocess as sproc;
import os;
import threading;

#==================================================================================================================
#                                       ---PREDICTION GENERATION---
# The classes in this section generate prediction profiles for various potentials and situations. They all follow the
# same ideas. First one initialized a new object of the class with the external potential of the simulation. Then one
# can call 'generateProfile', which returns a function representing the prediction. These functions are normalized,
# and designed to handle NumPy arrays.

'''
Generates the predicted density profile for a system in the thermal limit (ie. diffusion >> memory).
'''
class ThermalDensityPredictor:
    '''Generates a new ThermalDensityPredictor.
       @param potential: Callable that returns the external potential of the system as a function of position. (Must accept NumPy arrays).'''
    def __init__(self, potential):
        self.potential = potential;

    '''Generates the prediction profile as a normalized density.
       @param index: The index of the simulation, for logging purposes, this should be an integer, but can actually be anything.
       @param diffusion: The diffusion constant being used in the simulation.
       @param xMin: The lower position bound used in calculating the normalization constant, larger is better. (defaults to -1000)
       @param xMax: The upper position bound used in calculating the normalization constant, larger is better. (defaults to 1000)
       @param dx: The position difference to use in calculating the normaliztion constant, smaller is better. (defaults to 0.001)
       @returns: A function specifying the predicted density of particles at every position.'''
    def generateProfile(self, index, diffusion, xMin=-1000, xMax=1000, dx=0.001):
        # Compute the normalization constant by integrating the un-normalized density with the trapezoidal algorithm.
        Y = np.exp(-(self.potential(np.arange(xMin, xMax, dx))) / diffusion);
        normalization = np.sum(((Y[1:] + Y[:-1]) / 2) * dx);
        # Return the normalized density function.
        return lambda x: np.exp(-(self.potential(x)) / diffusion) / normalization;

'''
Generates the predicted density profile for a system in the persistent limit (ie.  memory >> diffusion), for double well potentials.
'''
class DoubleWellPersistentDensityPredictor:
    '''POTENTIAL MUST BE PIECEWISE, MADE OF POLYNOMIALS AND MUST BE MIN, MAX, MIN!!!!!!!!'''#TODO COMMENTS
    def __init__(self, potential):
        self.dU = potential.derive();
        self.d2U = self.dU.derive();
        self.B = self.dU.bounds[1];
        self.C = self.dU.bounds[3];
        # Find the points that match the slopes at the inflection points.
        self.A = (self.dU(self.C) - self.dU.functions[0].c[0]) / self.dU.functions[0].c[1];
        self.D = (self.dU(self.B) - self.dU.functions[-1].c[0]) / self.dU.functions[-1].c[1];

    def generateProfile(self, index, memory, diffusion, dx=0.001):#TODO COMMENTS, THIS IS ALSO SUPER INEFFECIENT
        # Pre-compute some constants.
        c0 = memory / (2 * diffusion);
        c1 = np.sqrt(memory / (2 * np.pi * diffusion));
        # Compute the normalization integrals.
        Z1 = -self.i(self.A, self.B, c0, dx);
        Z2 = self.i(self.C, self.D, c0, dx);

        # Create a function for generating the steady state distribution of a normal convex potential.
        p0 = lambda x: (c1 * self.d2U(x)) * np.exp(-c0 * (self.dU(x)**2));

        # Create a list for storing the piecewise functions that comprise the prediction.
        functions = [];
        functions.append(p0);
        functions.append(lambda x: (p0(x) * -np.vectorize(self.i)(x, self.B, c0, dx) / Z1));
        functions.append(lambda x: (x * 0));
        functions.append(lambda x: (p0(x) * np.vectorize(self.i)(self.C, x, c0, dx) / Z2));
        functions.append(p0);

        # Return the piecewise density function. TODO IS THIS ALREADY NORMALIZED??
        return lambda x: np.piecewise(x, [(x < self.A), (x >= self.A), (x >= self.B), (x >= self.C), (x >= self.D)], functions);

    def i(self, xMin, xMax, c, dx, Z=1):#TODO COMMENTS
        Y = np.exp(c * self.dU(np.arange(xMin, xMax, dx))**2);
        return (np.sum((Y[1:] + Y[:-1]) / 2) * dx) / Z;

#TODO CHECK THIS THING, IT'S PROBABLY BROKEN, ALSO COMMENTS
from scipy.special import erfi
import scipy.linalg
import itertools as itt

'''THIS IS LARGELY NON-FUNCTIONAL'''
class PersistentDensityPredictor:
    def __init__(self, potential):
        self.dU_ = potential.derive();
        self.d2U_ = self.dU_.derive();

    def generateProfile(self, index, xMin, xMax, sampleCount, tau=1, diffusion=1):
        X = sp.linspace(xMin, xMax, sampleCount);
        # Find the extrema of dU, which are also the jump take offs.
        dU  = self.dU_(X)
        d2U = self.d2U_(X)
        I1  = sp.nonzero(d2U[:-1]*d2U[1:]<0)[0] # indices where d2U changes sign between points
        # Indices where d2U is 0 and the sign changes across the 0 point
        I1  = sp.append(I1, sp.intersect1d(sp.nonzero(d2U==0)[0], sp.nonzero(d2U[:-2]*d2U[2:]<0)[0] + 1))
        X1  = X[I1]
        dU1 = dU[I1]
        # Find the convex intervals.
        IIm = sp.vstack([[0]+I1.tolist(),I1.tolist()+[len(X)-1]]).T # intervals where dU is monotonic
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
        E  = lambda v,v0=sp.sqrt(2*diffusion/tau): tau**2/diffusion*v0*erfi(v/v0);
        for k,(i1,j1,i2,j2,s) in enumerate(jumps):
            # Zero total flux over each region.
            A[j1,k] = -1
            A[j2,k] =  1
            # q=0 at the right end of each region except
            # q=sqrt(tau/2*pi*diffusion for the last region.
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
                bins.append(X[i]);
                a -= s*J[k]
                b += s*J[k]*E(dU[i])
                vals.append([a,b]);
            # If things go well the last value pair of the region is [0,0]
            # and describe the density in the following concave region.
            b0 = 1 if j==Nc-1 else 0
            if sp.absolute(vals[-1][0])>1e-4 or sp.absolute(vals[-1][1]-b0)>1e-4:
                print(str(index) + ":\tThere are numerical precision issues.");
        bins.append(sp.inf);

        return lambda x: self.p(x, tau, diffusion, E, sp.array(bins), sp.array(vals), (sp.sqrt(tau/(2*sp.pi*diffusion))), (-tau / (2 * diffusion)))

    def p(self, x, tau, diffusion, E, bins, vals, q0, q1):
        a,b = vals[sp.digitize(x,bins)-1].T
        return self.d2U_(x)*q0*(a*E(self.dU_(x))+b)*sp.exp(q1*self.dU_(x)**2);

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
class Histogram:
    '''Loads the histograms stored in a specified file.
       @param filePath: Path of the data file to load into the histogram.'''
    def __init__(self, filePath):
        with open(filePath) as file:
            # Load the bins from the first line of the data file.
            self.bins = np.array(list(map(float, next(file)[:-2].split(','))));
            times = [];
            data = [];
            # Loop through the renaming lines, each line is just a comma separated list of the bin values.
            for line in file:
                times.append(float(line[(line.index("t=") + 2):line.index(':')]));
                data.append(np.array(list(map(float, line[(line.index(':') + 1):-2].split(',')))));
            # Convert the data into numpy arrays, and store them for later use.
            self.times = np.array(times);
            self.data = np.array(data);

    '''Interpolates the values of a histogram's bins into a set of X and Y coordinates.
       @param frame: The histogram to interpolate, with 0 being the first histogram in the file, must be an integer. (defaults to -1, the last histogram)
       @param normalize: True if the histogram's values should be normalize, False to use the raw data. (defaults to True)
       @returns: A tuple of X and Y coordinates that represent the piecewise linear interpolation of the histogram's data, where each X
                 value is the X position midway between the bin's bounds, and the corresponding Y is the value of that bin.'''
    def interpolate(self, frame=-1, normalize=True):
        # Compute the points in the middle of each bin's bounds for the X values.
        X = (self.bins[1:] + self.bins[:-1]) / 2;
        # Fetch the data from all the non-overflow bins for the Y values.
        Y = self.data[frame][1:-1];

        if(normalize):
            # Compute the normalization constant by taking the integral over the function with the trapazoidal algorithm.
            normalization = np.sum(((Y[1:] + Y[:-1]) / 2) * (self.bins[1] - self.bins[0]));
            # Normalize the histogram values.
            Y /= normalization;

        return (X, Y);

'''
TODO COMMENTS
'''
class Animator:
    def __init__(self, histogram):
        self.left = histogram.bins[:-1];
        self.right = histogram.bins[1:];
        self.bottom = np.zeros(len(self.left));
        self.top = histogram.data[0][1:-1];
        self.histogram = histogram;

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
        self.ax.set_ylim(0, (np.sum(histogram.data[0]) / 2));
        self.fig.suptitle("Time = 0");

    def animate(self, index):
        self.top = self.histogram.data[index][1:-1];
        self.vertices[1::5, 1] = self.top;
        self.vertices[2::5, 1] = self.top;
        self.fig.suptitle("Time = " + str(self.histogram.times[index]) + "s");
        return self.patch;

'''
Convenience function for loading an animation from a histogram data file. After loading the data in, it will immediately
launch a matplotlib window that plays through an animation of the histogram's bins.
@param filePath: Path of the data file to load the animation from.
'''
def viewHistogram(filePath):
    histogram = Histogram(filePath);
    animator = Animator(histogram);
    ani = animation.FuncAnimation(animator.fig, animator.animate, np.arange(1, len(histogram.data)), repeat=False);
    plt.show();
    plt.close();

#==================================================================================================================
# ---FUNCTION CREATION--- TODO EVERYTHING UNDER THIS NEEDS BETTER COMMENTS

'''
Class encapsulating a polynomial, it's intialized with coeffecients in ascending order (lowest order coeffecient
to highest order), then can be called like any normal function.
'''
class PolyFunc:
    def __init__(self, coeffecients):
        self.c = np.array(coeffecients, dtype=np.float64);
        self.derivative = None;
        self.integral = None;

    def __len__(self):
        return len(self.c);

    def __str__(self):
        return ("poly " + str(self.c)[1:-1]);

    def __call__(self, x):
        if((type(x) != np.ndarray) or (x.ndim == 0)):
            x = np.array([x], dtype=np.float64);
        return np.sum(np.multiply(self.c, np.power(x[:, None], np.arange(len(self)))), axis=1);

    def derive(self):
        if(self.derivative == None):
            self.derivative = PolyFunc(np.multiply(self.c, np.arange(len(self)))[1:]);
        return self.derivative;

    def integ(self, C):
        if(self.integral == None):
            self.integral = PolyFunc(np.array([C], dtype=self.c.dtype).append(np.multiply(self.c, np.reciprocal(np.arange(len.self) + 1))));
        return self.integral;

'''
Class encapsulating a piecewise function, it's intialized with functions, and the boundary points between them.
After initialization it can be called like any normal function.
Functions must be passed in left to right, as must the bounds. There is also an additional field to specify
the direction to evaulate the piecewise function from at the boundaries. False indicates to evaulate it from the
left, True to evaulate it from the right. If left unused, by default boundaries are always evaulated from the
left.
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
        self.derivative = None;
        self.integral = None;
        if(directions):
            self.directions = directions;
        else:
            self.directions = [False]*len(bounds);

    def __len__(self):
        return len(self.functions);

    def __getitem__(self, index):
        return self.functions[index];

    def __str__(self):
        s = "piece ";
        for i in range(len(self.bounds)):
            s += "\'" + str(self.functions[i]) + "\' " + str(self.bounds[i]) + " "
            if(self.directions[i]):
                s += "1 ";
            else:
                s += "0 ";
        s += "\'" + str(self.functions[-1]) + "\'";
        return s;

    def __call__(self, x):
        conditions = ([x < self.bounds[0]] if self.directions[0] else [x <= self.bounds[0]]);
        for i in range(len(self.bounds)):
            conditions.append((x >= self.bounds[i]) if self.directions[i] else (x > self.bounds[i]));
        conditions2 = [(x <= -2), (x > -2), (x > -1), (x > 0), (x > 1), (x > 2)]; #TODO THIS ONLY WORKS WHEN >= is used. SOMETHING SERIOUSLY WRONG HERE
        return np.piecewise(x, conditions2, self.functions);

    def derive(self):
        if(self.derivative == None):
            derivatives = [func.derive() for func in self.functions];
            self.derivative = PieceFunc(derivatives, cp.deepcopy(self.bounds), cp.deepcopy(self.directions));
        return self.derivative;

    def integ(self, C):
        if(self.integral == None):
            integrals = [func.integ(C) for func in self.functions];
            self.integral = PieceFunc(integrals, cp.deepcopy(self.bounds), cp.deepcopy(self.directions));
        return self.integral;

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

#Deriving and integrating these wrapper functions, don't return the same type. type(DoubleWellFunc.derive) != DoubleWellFunc
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
       @param binCount: The number of bins the histogram should have. These are equally spaced throughout the range.
                        If left as none, the bin count is computed using the bin density. (defaults to None)
       @param dx: The spacing to place between consecutive bins. This is only used if the bin count isn't
                  manually specified. (defaults to 0.1)'''
    def __init__(self, minimum, maximum, binCount=None, dx=0.1):
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

'''
Class for specifying the parameters of a custom histogram.
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

#==================================================================================================================
'''Memmory MUST BE A COLLECTION (preferably list)''' #TODO
def runSimulationTEMP(potential, predictorT=None, predictorP=None, outputFile="./result", posRecorder=None, forceRecorder=None, noiseRecorder=None, particleCount=None, duration=None, timestep=None, memory=[1], dataDelay=None, startBounds=None, activeForces=None, noise=None):
    i = 0;
    args = [];
    threads = [];

    dirIndex = max(outputFile.rfind('/'), outputFile.rfind('\\'));
    if((dirIndex > -1) and not os.path.exists(outputFile[:dirIndex])):
        os.makedirs(outputFile[:dirIndex]);

    for m in memory:
        d = np.sqrt(1 + (m**2));
        args.append([i, potential, predictorT, predictorP, outputFile + "t=" + str(m), posRecorder, forceRecorder, noiseRecorder, particleCount, duration, timestep, d, m, dataDelay, startBounds, activeForces, noise]);
        thread = threading.Thread(target=runSimulation, args=args[i]);
        threads.append(thread);
        thread.start();
        i += 1;

    for j in range(i):
        threads[j].join();

    for j in range(i):
        exportSimulation(*args[j]);

#TODO
def runSimulation(index, potential, predictorT=None, predictorP=None, outputFile="./result", posRecorder=None, forceRecorder=None, noiseRecorder=None, particleCount=None, duration=None, timestep=None, diffusion=1, memory=1, dataDelay=None, startBounds=None, activeForces=None, noise=None):
    print(str(index) + ":\tInitializing...");

    # Create the command for running the simulation.
    command = str(os.path.abspath(os.path.join(__file__, "../simulate"))) + " \"" + str(potential) + "\"";
    if(outputFile):
        command += " -of ";
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

    # Create the output directory if it doesn't exist (C++ can't export to non-existant directories) (ONLY IF IT'S NOT IN PARALLEL)
    if(threading.current_thread() == threading.main_thread()):
        dirIndex = max(outputFile.rfind('/'), outputFile.rfind('\\'));
        if((dirIndex > -1) and not os.path.exists(outputFile[:dirIndex])):
            os.makedirs(outputFile[:dirIndex]);

    # Run the simulation.
    print(str(index) + ":\tRunning Simulation...");
    with sproc.Popen(command, stdout=sproc.PIPE, stderr=sproc.STDOUT, bufsize=1, universal_newlines=True) as proc:
        stdBuffer = "Simulation Started: 0%\n";
        while((stdBuffer != "") or (proc.poll() == None)):
            if(stdBuffer):
                print(str(index) + ":\t" + stdBuffer, end='');
            stdBuffer = proc.stdout.readline();
        print(str(index) + ":\tSimulation finished with exit code: " + str(proc.poll()));
        if(proc.poll() != 0):
            raise sproc.CalledProcessError(proc.poll(), command);

    # Check if this is running in the main thread. If it is, then it's safe to export the results, if not, it's probably being run in parallel.
    if(threading.current_thread() == threading.main_thread()):
        return exportSimulation(index, potential, predictorT, predictorP, outputFile, posRecorder, forceRecorder, noiseRecorder, particleCount, duration, timestep, diffusion, memory, dataDelay, startBounds, activeForces, noise);
    return 0;

def exportSimulation(index, potential, predictorT=None, predictorP=None, outputFile="./result", posRecorder=None, forceRecorder=None, noiseRecorder=None, particleCount=None, duration=None, timestep=None, diffusion=1, memory=1, dataDelay=None, startBounds=None, activeForces=None, noise=None):
    # Export the results.
    print(str(index) + ":\tExporting results...");
    if(posRecorder):
        print(str(index) + ":\tGenerating prediction...");
        if not(predictorT):
            predictorT = ThermalDensityPredictor(potential);
        if not(predictorP):
            predictorP = DoubleWellPersistentDensityPredictor(potential);

        n = ((posRecorder.binMax - posRecorder.binMin) / posRecorder.dx) * 100;
        predictionT = predictorT.generateProfile(index, diffusion);
        predictionP = predictorP.generateProfile(index, memory, diffusion);
        posXY = Histogram(str(outputFile) + ".pos").interpolate();
        X = sp.linspace(posRecorder.binMin, posRecorder.binMax, n);
        ax = plt.gca();
        ax.set_xlabel("position");
        ax.set_ylabel("particle density");
        ax.plot(X, predictionT(X), "green");
        ax.plot(X, predictionP(X), "blue");
        ax.plot(posXY[0], posXY[1], "red");
        ax = ax.twinx();
        ax.set_ylabel("potential");
        ax.plot(X, potential(X), "black");

        black = patches.Patch(color="black", label="potential");
        green = patches.Patch(color="green", label="thermal");
        blue = patches.Patch(color="blue", label="persistent");
        red = patches.Patch(color="red", label="result");
        plt.legend(handles=[black, green, blue, red], loc=1);

        plt.savefig(str(outputFile) + "W.png", fmt=".png", dpi=200);
        plt.close();
    if(forceRecorder):
        pass;#TODO
    if(noiseRecorder):
        pass;#TODO

    return 0;
