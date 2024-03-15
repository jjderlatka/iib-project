from mpi4py import MPI
import dolfinx

def mpi_print(s):
    print(f"[Rank {MPI.COMM_WORLD.Get_rank()}]: {s}")

gdim = 2

mpi_print('Bonjour')
mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD, 0, gdim=gdim)

mpi_print(f"Number of local cells: {mesh.topology.index_map(2).size_local}")
mpi_print(f"Number of global cells: {mesh.topology.index_map(2).size_global}")
