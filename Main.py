
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib import path as path
from matplotlib import animation as animation
import numpy as np;
import subprocess as sproc;
import threading;

import scipy as sp
#from scipy.optimize import brentq
from scipy.special import erfi
import scipy.linalg
import itertools as itt
import matplotlib.pyplot as plt
import copy;

class FilyDensityCode:
    def __init__(self, coefficients):
        coefficients = list(coefficients);
        for counter in range(len(coefficients)):
            coefficients[counter] *= counter;
        coefficients = coefficients[1:];
        duTemp = copy.deepcopy(coefficients);
        self.dU_ = lambda x: polyFunction(duTemp, x);
        for counter in range(len(coefficients)):
            coefficients[counter] *= counter;
        coefficients = coefficients[1:];
        d2uTemp = copy.deepcopy(coefficients);
        self.d2U_ = lambda x: polyFunction(d2uTemp, x);

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
        q0 = sp.sqrt(tau/(2*sp.pi*diffusion))
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
        return self.d2U_(x)*(a*E(self.dU_(x))+b)*sp.exp(-tau*self.dU_(x)**2/(2*diffusion));

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

def polyFunction(coefficients, x):
    value = 0;
    for counter in range(len(coefficients)):
        value += coefficients[counter] * (x**counter);
    return value;

#TODO make this something other than JUST a linear histogram recorder
class Recorder:
    def __init__(self, type, min, max, count):
        self.type = type;
        self.binMin = min;
        self.binMax = max;
        self.binCount = count;

    def __str__(self):
        return self.type + ' ' + str(self.binMin) + ',' + str(self.binMax) + ',' + str(self.binCount);

#TODO make this something other than JUST a poly force
class Force:
    def __init__(self, coefficients):
        self.coefficients = coefficients;

    def __str__(self):
        return "poly" + ' ' + ','.join(map(str, self.coefficients));

    def act(self, x):
        value = 0;
        for counter in range(len(self.coefficients)):
            value += self.coefficients[counter] * (x**counter);
        return value;

def runSimulation(index, outputPath, force, predicter, posRecorder=None, forceRecorder=None, noiseRecorder=None, particleCount=50000, duration=60, timestep=0.05, diffusion=1, tau=1, dataDelay=20):
    print(str(index) + ":\tStarting...");
    #Create the command for running the simulation.
    command = "./simulate " + str(outputPath) + " -f " + str(force);
    if(posRecorder):
        command += " -pr " + str(posRecorder);
    if(forceRecorder):
        command += " -fr " + str(forceRecorder);
    if(noiseRecorder):
        command += " -nr " + str(noiseRecorder);
    command += " -p " + str(particleCount) + " -t " + str(duration) + " -dt "+ str(timestep) + " -d " + str(diffusion) + " -m " + str(tau) + " -dd " + str(dataDelay);
    print(command);
    #Run the simulation.
    print(str(index) + ":\tRunning Simulation...");
    returnVal = sproc.call(command.split());
    if(returnVal != 0):
        raise RuntimeError(str(tau) + ":\tFailed to run! Process returned: " + str(returnVal));

    print(str(index) + ":\tExporting results...");
    if(posRecorder):
        print(str(index) + ":\tGenerating prediction...");
        prediction = predicter.generateProfile(index, posRecorder.binMin, posRecorder.binMax, (posRecorder.binCount * 100), tau, diffusion);
        positionXY = Histogram(str(outputPath) + ".pos").interpolate();
        ax = plt.gca();
        X = sp.linspace(posRecorder.binMin, posRecorder.binMax, posRecorder.binCount * 10);
        ax.plot(X, force.act(X), 'g');
        ax = ax.twinx();
        ax.plot(X, prediction(X), 'b');
        ax.plot(positionXY[0], positionXY[1], 'r');
        plt.savefig(str(outputPath) + ".png",fmt='png',dpi=1000);
        plt.close();

runSimulation(0, "Results/test", Force([1, 0, 1]), FilyDensityCode([1,0,1]), Recorder("linear", -10, 10, 200));

#need this in the recorders binMin, binMax, binCount!!
#and also other stuff for the force too.
#def polySimB(coefficients, tau=1, diffusion=1, particles=50000, binCount=200, binMinP=-10, binMaxP=10):#, binMinF=-1, binMaxF=1

def runPolySim(coefficients, tau=1, diffusion=1, dt=0.005, particles=20000, duration=30, dataDelay=200, binCount=500, binMinP=-10, binMaxP=10, binMinF=-1, binMaxF=1, parallel=True):
    try:
        if(parallel):
            threads = [];
            for i in range(len(tau)):
                threads.append(threading.Thread(target=PolySimA, args=[coefficients, tau[i], diffusion, dt, particles, duration, dataDelay, binCount, binMinP, binMaxP, binMinF, binMaxF, str(i)]));
                threads[-1].start();
            for i in range(len(tau)):
                threads[i].join();
                polySimB(coefficients, tau[i], diffusion, particles, binCount, binMinP, binMaxP);
        else:
            for t in tau:
                PolySimA(coefficients, t, diffusion, dt, particles, duration, dataDelay, binCount, binMinP, binMaxP, binMinF, binMaxF);
                polySimB(coefficients, t, diffusion, particles, binCount, binMinP, binMaxP);
    except TypeError:
        PolySimA(coefficients, tau, diffusion, dt, particles, duration, dataDelay, binCount, binMinP, binMaxP, binMinF, binMaxF);
        polySimB(coefficients, tau, diffusion, particles, binCount, binMinP, binMaxP);
    print("Finished!");





