from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n = 36
x = np.random.random(n**2)
# print(f"Process {rank}, x={x}")

sum = np.empty(n**2)

comm.Barrier()
start_time = MPI.Wtime()
comm.Allreduce(x, [sum, MPI.DOUBLE], op=MPI.SUM)
end_time = MPI.Wtime()
print(f"Process {rank}, time taken = {end_time-start_time}")
# if rank == 0:
#     print(f"Sum={sum}")

# try 3, 6, 9, 12 processes to see how communication increases the allreduce time
# see also different n: 30, 60, ...

# try plotting a contour of n with processes (may be log linear)