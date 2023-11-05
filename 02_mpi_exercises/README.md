# Message Passing Interface exercises

Run using:

    mpiexec -n 3 python script.py
    
replace the 3 with desired number of processes to run.

### Exercise 1

Generate a data set (matrix) in one process, distribute it to all processes, but make every process only read a portion of the data set.

### Exercise 2

Same as exercise 1, but using `MPI.COMM.scatter()`

### Exercise 3

Demonstration of the time required for communication between processes. Generating a random vector `x` in each process, and summing all of them under `sum` variable, using `MPI.COMM.Allreduce()`.