NOWARN=-wd3180
EXEC=lu-omp
OBJ =  $(EXEC) $(EXEC)-debug $(EXEC)-serial

MATRIX_SIZE=8000
MATRIX_CHECK_SIZE=100
W :=`grep processor /proc/cpuinfo | wc -l`

CHECKER=inspxe-cl -collect=ti3 -r check
VIEWER=inspxe-gui

# flags
OPT=-O2 -g
DEBUG=-O0 -g
OMP=-fopenmp

all: $(OBJ)

# build the debug parallel version of the program
$(EXEC)-debug: $(EXEC).cpp
	icpc $(DEBUG) $(OMP) -o $(EXEC)-debug $(EXEC).cpp -lrt -lnuma


# build the serial version of the program
$(EXEC)-serial: $(EXEC).cpp
	icpc $(OPT) $(NOWARN) -o $(EXEC)-serial $(EXEC).cpp -lrt -liomp5 -lnuma 

# build the optimized parallel version of the program
$(EXEC): $(EXEC).cpp
	icpc $(OPT) $(OMP) -o $(EXEC) $(EXEC).cpp -lrt -lnuma

#run the optimized program in parallel
runp: $(EXEC)
	@echo use make runp W=nworkers
	./$(EXEC) $(MATRIX_SIZE) $(W)

#run the serial version of your program
runs: $(EXEC)-serial
	@echo use make runs
	./$(EXEC)-serial $(MATRIX_SIZE) 1

#run the optimized program with thread checker
check: $(EXEC)
	@echo use make check W=nworkers
	$(CHECKER) ./$(EXEC) $(MATRIX_SIZE) $(W)

#view the thread checker result
view:
	$(VIEWER) check*/check*.inspxe


clean:
	/bin/rm -rf $(OBJ) check*
