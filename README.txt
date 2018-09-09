First launch a python console, and 'import Main'.
There's really only two functions that are useful to you in here 'runPolySim' and 'viewHistogram'.

runPolySim first compiles and runs an active particle simulation in C++ with the specified parameters. In the process of doing so it'll generate some files, the 'simulate.exe' files can be deleted afterwards if you want, and there's also *.pos files with position data, and *.for files with active force data. Their names correspond to the value of tau used in that simulation. Next it generates the predicted distribution as per your code, and finally graphs the potential (green), prediction (blue), and results of the simulation (red) all together in a *.png file.
Here's the parameters of the function:
  - coefficients: Coeffecients of the potential polynomial from smallest to largest order. So the first element is the constant term, second is linear, etc...
  - tau:          Memory constant for the simulation (defaults to 1)
  - diffusion:    Diffusion constant for the simulation (defaults to 1)
  - dt:           Timestep for the simulation (defaults to 0.005)
  - particles:    Number of particles to simulate (defaults to 20000)
  - duration:     Duration for the simulation (defaults to 30)
  - dataDelay:    How many timesteps to wait in between recording data (defaults to 200)
  - binCount:     Number of bins to make for the histograms (defaults to 500)
  - binMinP:      The minimum position to record on the histogram (defaults to -10)
  - binMaxP:      The maximum position to record on the histogram (defaults to 10)
  - binMinF:      The minimum active force to record on the histogram (defaults to -1)
  - binMaxF:      The maximum active force to record on the histogram (defaults to 1)
  - parallel:     Whether to execute in parallel (defaults to True)

That's right! Instead of passing in just a single value for tau (which you definitely can still do), you can also pass in a collection that will either run sequentially or in parallel (if you have it set).
If you do run in parallel it's worth starting small, or keeping a process-manager ready, because this will murder your CPU when you run in parallel. It's also very hard to stop once it's started...
Example:
Main.runPolySim([0,0,1], [4,2,1,0.5]);
Will run 4 simulations with taus of 4, 2, 1, and 0.5 with V=x^2 as their potential.
If you find simulations are taking too long, you can decrease the number of particles, which decreases the accuracy and resolution of the result's graph, or increase the timestep which also decreases accuracy. Setting tau<dt can result in some bizarre behavior as well, so your timestep also sets a minimum tau you can use.



Then there's also the viewHistogram function, which loads an animation of the histograms over time, and only takes a single parameter of the data file to load. Both the *.pos and *.for can be loaded with this.

Example:
Main.viewHistogram("T=1.pos");
Will launch an animation showing how the positions changed over the course of the simulation with tau=1.