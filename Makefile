# Compiler and flags
CXX = icpc
CXXFLAGS = -O2 -fopenmp -std=c++11 -lrt -lnuma
DEBUGFLAGS = -O0 -g -fopenmp -std=c++11 -lrt -lnuma
NOWARN = -wd3180

# Executables
EXEC = lu-omp
EXEC_DEBUG = $(EXEC)-debug
EXEC_SERIAL = $(EXEC)-serial

# Problem sizes (override when calling make)
MATRIX_SIZE = 8000
CHECK_SIZE = 100
W := $(shell grep processor /proc/cpuinfo | wc -l)

# Intel Inspector tools (for race detection)
CHECKER = inspxe-cl -collect=ti3 -r check
VIEWER = inspxe-gui

# Targets to build everything
all: $(EXEC) $(EXEC_DEBUG) $(EXEC_SERIAL)

# Parallel optimized build
$(EXEC): $(EXEC).cpp
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(EXEC).cpp

# Parallel debug build
$(EXEC_DEBUG): $(EXEC).cpp
	$(CXX) $(DEBUGFLAGS) -o $(EXEC_DEBUG) $(EXEC).cpp

# Serial build (optional: same code with 1 thread)
$(EXEC_SERIAL): $(EXEC).cpp
	$(CXX) $(CXXFLAGS) $(NOWARN) -o $(EXEC_SERIAL) $(EXEC).cpp

# Run parallel executable with number of workers (W)
runp: $(EXEC)
	@echo "Running parallel version with $(W) workers..."
	./$(EXEC) $(MATRIX_SIZE) $(W)

# Run serial executable (always uses 1 thread)
runs: $(EXEC_SERIAL)
	@echo "Running serial version..."
	./$(EXEC_SERIAL) $(MATRIX_SIZE) 1

# Run parallel program with Intel Inspector thread checker
check: $(EXEC)
	@echo "Running Inspector to check for data races..."
	$(CHECKER) ./$(EXEC) $(CHECK_SIZE) $(W)

# View Inspector race detection results
view:
	@echo "Launching Inspector GUI..."
	$(VIEWER) check*/check*.inspxe

# Clean up builds and Inspector reports
clean:
	rm -rf $(EXEC) $(EXEC_DEBUG) $(EXEC_SERIAL) check*

# Convenience info
info:
	@echo "Executables:"
	@echo "  make runp  # Run parallel version"
	@echo "  make runs  # Run serial version"
	@echo "  make check # Run parallel version with Intel Inspector"
	@echo ""
	@echo "Matrix Size: $(MATRIX_SIZE)"
	@echo "Threads (W): $(W)"
