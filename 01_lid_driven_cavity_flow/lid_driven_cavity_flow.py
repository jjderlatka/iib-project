import gmsh

from dolfinx import io 
from mpi4py import MPI
from pathlib import Path

import ufl
from dolfinx import fem, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import numpy as np

from petsc4py import PETSc

from dolfinx.plot import vtk_mesh
import pyvista

# Creating the mesh
gmsh.initialize()

mesh_size = 0.04
gdim = 2 # dimension of the model

gmsh.model.add("parallelogram")

A = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)  # (x, y, z, mesh size)
B = gmsh.model.geo.addPoint(1, 0, 0, mesh_size)
C = gmsh.model.geo.addPoint(2, 1, 0, mesh_size)
D = gmsh.model.geo.addPoint(1, 1, 0, mesh_size)

AB = gmsh.model.geo.addLine(A, B)
BC = gmsh.model.geo.addLine(B, C)
CD = gmsh.model.geo.addLine(C, D)
DA = gmsh.model.geo.addLine(D, A)

loop = gmsh.model.geo.addCurveLoop([AB, BC, CD, DA])
gmsh.model.geo.addPlaneSurface([loop])

fluid_marker, lid_marker, wall_marker = 1, 2, 3
gmsh.model.addPhysicalGroup(2, [loop], fluid_marker) # doesn't work without this
gmsh.model.addPhysicalGroup(1, [CD], lid_marker)
gmsh.model.addPhysicalGroup(1, [AB, BC, DA], wall_marker)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(gdim)

# Importing the mesh to dolfinx
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
mesh, cell_markers, facet_markers = io.gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

# Function spaces
P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
UP = P2 * P1
W = fem.FunctionSpace(mesh, UP)
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()
# (u, p) = ufl.TrialFunctions(W)
w_trial = fem.Function(W)
(u, p) = ufl.split(w_trial)
w_test = ufl.TestFunction(W)
(v, q) = ufl.split(w_test)
fdim = mesh.topology.dim - 1

# Dirichlet boundary conditions

# Velocity
# No-slip walls
u_nonslip = (0,0)
bcu_walls = fem.dirichletbc(np.array(u_nonslip, dtype=PETSc.ScalarType), fem.locate_dofs_topological(V, fdim, facet_markers.find(wall_marker)), V)
# Driving lid
u_lid = (1,0)
bcu_lid = fem.dirichletbc(np.array(u_lid, dtype=PETSc.ScalarType), fem.locate_dofs_topological(V, fdim, facet_markers.find(lid_marker)), V)

# Pressure
def bottom_left(x):
    return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))

p_reference = PETSc.ScalarType(0)
bcp = fem.dirichletbc(p_reference, fem.locate_dofs_geometrical(Q, bottom_left), Q)

# Form

# bilinear parts
a_1 = ufl.inner(ufl.grad(v), ufl.grad(u)) * ufl.dx # not ufl.dot
a_2 = - p * ufl.div(v) * ufl.dx # not grad
a_3 = q * ufl.div(u) * ufl.dx # not grad
# non-linear part
b = ufl.inner(v, ufl.grad(u) * u) * ufl.dx
F = a_1 + a_2 + a_3 + b

# Nonlinear problem solution
problem = NonlinearProblem(F, w_trial, bcs=[bcu_walls, bcu_lid, bcp])

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(w_trial)
assert (converged)
print(f"Number of interations: {n:d}")

# # Saving the result
# results_folder = Path("results")
# results_folder.mkdir(exist_ok=True, parents=True)
# filename = results_folder / "lid_driven_cavity_flow"

# with io.VTXWriter(mesh.comm, filename.with_suffix(".bp"), [w_trial.sub(0)]) as vtx:
#     vtx.write(0.0)
# with io.XDMFFile(mesh.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_function(w_trial.sub(0))

# Plot
pyvista.start_xvfb()
pyvista.OFF_SCREEN = True

# Velocity
u_topology, u_cell_types, u_geometry = vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_vectors = np.zeros((u_geometry.shape[0], 3), dtype=np.float64)
u_vectors[:, :len(w_trial.sub(0).collapse())] = w_trial.sub(0).collapse().x.array.real.reshape((u_geometry.shape[0], len(w_trial.sub(0).collapse())))

mesh_grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(mesh_grid, show_edges=True)
u_plotter.add_arrows(u_grid.points, u_vectors, mag=4e-19)
u_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    u_plotter.screenshot("velocity.png")

# Pressure
p_topology, p_cell_types, p_geometry = vtk_mesh(Q)
p_grid = pyvista.UnstructuredGrid(p_topology, p_cell_types, p_geometry)
p_values = w_trial.sub(1).collapse().x.array.real

# mesh_grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))
p_grid.point_data["p"] = p_values
p_grid.set_active_scalars("p")
p_plotter = pyvista.Plotter()
p_plotter.add_mesh(p_grid, show_edges=True)
p_plotter.view_xy()

if not pyvista.OFF_SCREEN:
    p_plotter.show()
else:
    p_plotter.screenshot("pressure.png")
