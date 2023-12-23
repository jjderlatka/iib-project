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

# *** PARAMETERS ***
a = 1
b = 1
theta = np.pi / 4

nu = PETSc.ScalarType(1.) # NOTE Kinematic viscocity (say Water)
rho = PETSc.ScalarType(1.) # NOTE Density (say Water)

# Creating the mesh
gmsh.initialize()

mesh_size = 0.04
gdim = 2 # dimension of the model

gmsh.model.add("parallelogram")

A = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
B = gmsh.model.geo.addPoint(a, 0, 0, mesh_size)
C = gmsh.model.geo.addPoint(b * np.cos(theta) + a, b * np.sin(theta), 0, mesh_size)
D = gmsh.model.geo.addPoint(b * np.cos(theta), b * np.sin(theta), 0, mesh_size)

AB = gmsh.model.geo.addLine(A, B)
BC = gmsh.model.geo.addLine(B, C)
CD = gmsh.model.geo.addLine(C, D)
DA = gmsh.model.geo.addLine(D, A)

loop = gmsh.model.geo.addCurveLoop([AB, BC, CD, DA])
gmsh.model.geo.addPlaneSurface([loop])

fluid_marker, lid_marker, wall_marker = 1, 2, 3
gmsh.model.addPhysicalGroup(2, [loop], fluid_marker)
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

w_trial = fem.Function(W)
(u, p) = ufl.split(w_trial)
w_test = ufl.TestFunction(W)
(v, q) = ufl.split(w_test)

fdim = mesh.topology.dim - 1

# Dirichlet boundary conditions

# Velocity
# No-slip walls
u_nonslip = fem.Function(V)
u_nonslip.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
bcu_walls = fem.dirichletbc(u_nonslip, fem.locate_dofs_topological((W.sub(0), V), fdim, facet_markers.find(wall_marker)), W.sub(0))
# Driving lid
u_lid = fem.Function(V)
u_lid.interpolate(lambda x: (np.ones(x[0].shape), np.zeros(x[1].shape)))
bcu_lid = fem.dirichletbc(u_lid, fem.locate_dofs_topological((W.sub(0), V), fdim, facet_markers.find(lid_marker)), W.sub(0))

# Pressure
def bottom_left(x):
    return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))

p_reference = fem.Function(Q)
p_reference.interpolate(lambda x: np.zeros(x[0].shape))
bcp = fem.dirichletbc(p_reference, fem.locate_dofs_geometrical((W.sub(1), Q), bottom_left), W.sub(1))

# Form

# Bilinear parts
a_1 = ufl.inner(ufl.grad(v), nu * ufl.grad(u)) * ufl.dx
a_2 = - p * ufl.div(v) / rho * ufl.dx
a_3 = q * ufl.div(u) * ufl.dx

# Non-linear part
b = ufl.inner(v, ufl.grad(u) * u) * ufl.dx
F = a_1 + a_2 + a_3 + b

# Nonlinear problem assembly
problem = NonlinearProblem(F, w_trial, bcs=[bcu_walls, bcu_lid, bcp])


# Nonlinear problem solution
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "bcgs" # NOTE "preonly"
opts[f"{option_prefix}pc_type"] = "none" # NOTE "lu"
ksp.setFromOptions()

# TODO Try different solvers and change number of mesh points and number of processes

log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(w_trial)
assert (converged)
print(f"Number of interations: {n:d}")

# Saving the result
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)

V_interp = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
u_interp = fem.Function(V_interp)
u_expr = fem.Expression(w_trial.sub(0).collapse(), V_interp.element.interpolation_points())
u_interp.interpolate(u_expr)
 
filename = results_folder / "lid_driven_cavity_flow_velocity" # NOTE filename for velocity
 
with io.XDMFFile(mesh.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
     xdmf.write_mesh(mesh)
     xdmf.write_function(u_interp)

filename = results_folder / "lid_driven_cavity_flow_pressure" # NOTE filename for pressure

with io.XDMFFile(mesh.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
     xdmf.write_mesh(mesh)
     xdmf.write_function(w_trial.sub(1).collapse())