from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

sendbuf = None
if rank == 0:
    sendbuf = np.empty((12, 8), dtype='i')
    sendbuf.T[:,:] = range(3)
recvbuf = np.empty(8, dtype='i')
comm.Scatter(sendbuf, recvbuf, root=0)