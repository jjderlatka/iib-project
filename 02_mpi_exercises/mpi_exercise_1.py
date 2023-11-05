from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    M = np.random.random((12, 8))
    print(f"M={M}")
else:
    M = np.empty((12, 8))
comm.Bcast(M, root=0)

data_buffer = np.empty((4, 8))

for row_index in range(rank, 12, 3):
    # print(f"Process {rank} reading row {row_index}: {M[row_index]}")
    data_buffer[(row_index-rank)//3] = M[row_index]

print(f"Data read by process {rank}: {data_buffer}")