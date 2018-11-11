
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
# Code for generating prediction curves. Initialize it with the external force, then generate predictions
# for specific scenarios by calling 'generateProfile'.

from scipy.special import erfi
import scipy.linalg
import itertools as itt
import copy;

class DensityPredictor:
    def __init__(self, potential):
        self.U_ = potential;
        self.dU_ = self.U_.deriv();
        self.d2U_ = self.dU_.deriv();

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
        self.q0 = sp.sqrt(tau/(2*sp.pi*diffusion))
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

        return lambda x: self.p(x, tau, diffusion, E, sp.array(bins), sp.array(vals))

    def p(self, x, tau, diffusion, E, bins, vals):
        a,b = vals[sp.digitize(x,bins)-1].T
        return self.d2U_(x)*self.q0*(a*E(self.dU_(x))+b)*sp.exp(-tau*self.dU_(x)**2/(2*diffusion));

#==================================================================================================================
# Utilities for recording and visualing collected data

class Histogram:
    def __init__(self, filePath):
        with open(filePath) as file:
            self.bins = np.array(list(map(float, next(file)[:-2].split(','))));
            self.times = [];
            self.data = [];
            for line in file:
                self.times.append(float(line[(line.index("t=") + 2):line.index(':')]));
                self.data.append(list(map(float, line[(line.index(':') + 1):-2].split(','))));
            self.times = np.array(self.times);
            self.data = np.array(self.data);

    def interpolate(self, frame=-1):
        X = (self.bins[1:] + self.bins[:-1]) / 2;
        normalizer = 0;
        lastPoint = self.data[frame][1]
        for point in self.data[frame][2:-1]:
            normalizer += min(lastPoint, point) + (abs(point - lastPoint) / 2);
            lastPoint = point;
        normalizer *= (self.bins[1] - self.bins[0]);
        Y = self.data[frame][1:-1] / normalizer;
        return (X, Y);

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

def viewHistogram(dataFile):
    histogram = Histogram(dataFile);
    animator = Animator(histogram);
    ani = animation.FuncAnimation(animator.fig, animator.animate, np.arange(1, len(histogram.data)), repeat=False);
    plt.show();
    plt.close();

#==================================================================================================================
# Utilities for creating functions.

'''
Class encapsulating a polynomial, it's intialized with coeffecients in ascending order (lowest order coeffecient
to highest order), then can be called like any normal function.
'''
class PolyFunc:
    def __init__(self, coeffecients):
        self.c = np.array(coeffecients);
        self.derivative = None;
        self.integral = None;

    def __len__(self):
        return len(self.c);

    def __str__(self):
        return ("\"poly " + str(self.c)[1:-1] + "\"");

    def __call__(self, x):
        return np.sum(np.multiply(self.c, np.power(x[:, None], np.arange(len(self)))), axis=1);

    def deriv(self):
        if(self.derivative == None):
            self.derivative = PolyFunc(np.multiply(self.c, np.arange(len(self)))[1:]);
        return self.derivative;

    def integ(self, C):
        if(self.integral == None):
            self.integral = PolyFunc(np.array([C]).append(np.multiply(self.c, np.reciprocal(np.arange(len.self) + 1))));
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
        self.bounds = bounds;
        self.derivative = None;
        self.integral = None;

    def __len__(self):
        return len(self.functions);

    def __getitem__(self, index):
        return self.functions[index];

    def __str__(self):
        s = "\"piece ";
        for i in range(len(self.bounds)):
            s += "\'" + str(self.functions[i]) + "\' " + str(self.bounds[i]) + " "
            if((self.directions == None) or not directions[i]):
                s += "0 ";
            else:
                s += "1 ";
        s += "\'" + str(self.functions[-1]) + "\'\"";
        return s;

    def __call__(self, x):
        for i in range(len(self.bounds)):
            if(x <= self.bounds[i]):
                if((x == self.bounds[i]) and (directions != None) and (directions[i])):
                    i += 1;
                break;
        else:
            i += 1;
        return self.functions[i](x);

    def deriv(self):
        if(self.derivative == None):
            derivatives = [func.derive() for func in self.functions];
            self.derivative = PieceFunc(derivatives, cp.deepcopy(self.bounds), cp.deepcopy(self.directions));
        return self.derivative;

    def integ(self, C):
        if(self.integral == None):
            integrals = [func.integ(C) for func in self.functions];
            self.integral = PieceFunc(integrals, cp.deepcopy(self.bounds), cp.deepcopy(self.directions));
        return self.integral;

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

        ab1 = ab.deriv();
        ab2 = ab1.deriv();
        ab3 = ab2.deriv();
        ab4 = ab3.deriv();
        aa = createSpline11111(points[0], ab(points[0]), ab1(points[0]), ab2(points[0]), ab3(points[0]), ab4(points[0]));

        de1 = de.deriv();
        de2 = de1.deriv();
        de3 = de2.deriv();
        de4 = de3.deriv();
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

    def deriv(self):
        return self.function.deriv();

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
# Utilities for creating histograms
#TODO WE NEED COMMENTS AND DOCUMENTATION FOR THIS!

class LinearHistogram:
    def __init__(self, binCount, minimum, maximum):
        self.binCount = binCount;
        self.binMin = minimum;
        self.binMax = maximum;
        self.width = (maximum - minimum) / binCount;

    def __str__(self):
        return ("\"linear " + str(self.binCount) + " " + str(self.binMin) + " " + str(self.binMax) + "\"");

class CustomHistogram:
    def __init__(self, bins):
        self.bins = bins;

    def __str__(self):
        return ("\"custom " + " ".join(map(str, self.bins)) + " \"");

#==================================================================================================================
#TODO DOWN
def runSimulation(index, potential, predictor=None, outputFile="./result", posRecorder=None, forceRecorder=None, noiseRecorder=None, particleCount=None, duration=None, timestep=None, diffusion=1, memory=1, dataDelay=None, startBounds=None, activeForces=None, noise=None):
    print(str(index) + ":\tInitializing...");

    # Create the command for running the simulation.
    command = str(os.path.abspath(os.path.join(__file__, "../simulate"))) + " " + str(potential);
    if(outputFile):
        command += " -of " + str(outputFile);
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

    if not(predictor):
        predictor = DensityPredictor(potential);

    # Create the output directory if it doesn't exist (C++ can't export to non-existant directories)
    dirIndex = max(outputFile.rfind('/'), outputFile.rfind('\\'));
    if((dirIndex > -1) and not os.path.exists(outputFile[:dirIndex])):
        os.makedirs(outputFile[:dirIndex]);

    # Run the simulation.
    print(str(index) + ":\tRunning Simulation...");
    returnVal = sproc.call(command);
    if(returnVal != 0):
        raise RuntimeError(str(index) + ":\tFailed to execute! Process returned: " + str(returnVal));

    # Export the results.
    print(str(index) + ":\tExporting results...");
    if(posRecorder):
        print(str(index) + ":\tGenerating prediction...");
        prediction = predictor.generateProfile(index, posRecorder.binMin, posRecorder.binMax, (posRecorder.binCount * 100), memory, diffusion);
        posXY = Histogram(str(outputFile) + ".pos").interpolate();
        X = sp.linspace(posRecorder.binMin, posRecorder.binMax, posRecorder.binCount * 100);
        ax = plt.gca();
        ax.set_xlabel("position");
        ax.set_ylabel("particle density");
        ax.plot(X, prediction(X), 'g')
        ax.plot(posXY[0], posXY[1], 'b');
        ax = ax.twinx();
        ax.set_ylabel("potential");
        ax.plot(X, potential(X), 'r');

        red = patches.Patch(color="red", label="potential");
        green = patches.Patch(color="green", label="prediction");
        blue = patches.Patch(color="blue", label="result");
        plt.legend(handles=[red, green, blue]);

        plt.savefig(str(outputFile) + "P.png", fmt=".png", dpi=400);
        plt.close();
    if(forceRecorder):
        pass;#TODO
    if(noiseRecorder):
        pass;#TODO
