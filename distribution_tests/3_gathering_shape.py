from mpi4py import MPI
import numpy as np

def mpi_print(s):
    print(f"[Rank {MPI.COMM_WORLD.Get_rank()}]: {s}")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

N = 10
my_training_set = np.array_split(list(range(1, N+1)), MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]

data = []

for i in my_training_set:
    data.append(i*np.ones(3))

data = np.array(data)
mpi_print(data)

data = MPI.COMM_WORLD.gather(data, root=0)
mpi_print(data)

if rank == 0:
    data = np.concatenate(data, axis=0)
    mpi_print(data)
