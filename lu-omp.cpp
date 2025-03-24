#include <iostream>
#include <numa.h>
#include <omp.h>

int value = 0;

long fib(int n)
{
  if (n < 2) return n;
  else return fib(n-1) + fib(n-2);
}

void 
usage(const char *name)
{
	std::cout << "usage: " << name
                  << " matrix-size nworkers"
                  << std::endl;
 	exit(-1);
}


int
main(int argc, char **argv)
{

  const char *name = argv[0];

  if (argc < 3) usage(name);

  int matrix_size = atoi(argv[1]);

  int nworkers = atoi(argv[2]);

  std::cout << name << ": " 
            << matrix_size << " " << nworkers
            << std::endl;

  omp_set_num_threads(nworkers);

#pragma omp parallel
{
   int tid = omp_get_thread_num();
   void *mem = numa_alloc_local(4);
   int myN = 20 - tid;
   if (myN < 16) myN = 16;
   long res = fib(myN);
// #pragma omp critical
   value = tid; // data race
   printf("thread %d fib(%d) = %ld\n", tid, myN, res);
}
return 0;
}
