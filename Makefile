CC=g++
CFLAGS=-I.

simmake: Simulation.cpp Function.cpp Histogram.cpp HistogramRecorder.cpp
	$(CC) -o simulate Simulation.cpp Function.cpp Histogram.cpp HistogramRecorder.cpp $(CFLAGS)