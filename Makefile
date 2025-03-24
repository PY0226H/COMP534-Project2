# Adjust compiler and flags as necessary
CXX = icpc
CXXFLAGS = -O3 -fopenmp -std=c++11

TARGET = lu_omp

all: $(TARGET)

$(TARGET): lu_omp.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TARGET) *.o
