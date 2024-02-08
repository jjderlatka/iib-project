from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion
from dolfinx.nls.petsc import NewtonSolver

import rbnicsx.io, rbnicsx.backends
import dolfinx
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from itertools import product

class Parameters():
    def __init__(self, a=1, b=1, theta=np.pi / 2, nu=PETSc.ScalarType(1.), rho=PETSc.ScalarType(1.)):
        self.a = a
        self.b = b
        self.theta = theta
        self.nu = nu # NOTE Kinematic viscocity
        self.rho = rho # NOTE Density


    def __repr__(self):
        return f"a={self.a},b={self.b},theta={self.theta:.2f},nu={self.nu:.2f},rho={self.rho:.2f}"


    def matrix(self):
        return np.array([[self.a, self.b * np.cos(self.theta)],
                         [0,      self.b * np.sin(self.theta)]])
    

    def transform(self, x):
        return self.matrix() @ x[:gdim]


class ProblemOnDeformedDomain():
    def __init__(self, mesh, subdomains, boundaries, meshDeformationContext):
        # Mesh, Subdomians and Boundaries, Mesh deformation
        self._mesh = mesh
        self.gdim = self._mesh.geometry.dim
        self._subdomains = subdomains # NOTE cell_markers
        self._boundaries = boundaries # NOTE facet_markers
        self.meshDeformationContext = meshDeformationContext # TODO obsolete

        self._fluid_marker, self._lid_marker, self._wall_marker = 1, 2, 3

        # Function spaces
        P2 = ufl.VectorElement("Lagrange", self._mesh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", self._mesh.ufl_cell(), 1)
        UP = P2 * P1

        self._W = dolfinx.fem.FunctionSpace(self._mesh, UP)
        self._V, _ = self._W.sub(0).collapse()
        self._Q, _ = self._W.sub(1).collapse()


    def get_boundary_conditions(self):
        # Dirichlet boundary conditions

        fdim = self._mesh.topology.dim - 1

        # Velocity
        # No-slip walls
        u_nonslip = dolfinx.fem.Function(self._V)
        u_nonslip.interpolate(lambda x: (np.zeros_like(x[0]), np.zeros_like(x[1])))
        bcu_walls = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological((self._W.sub(0), self._V), fdim, self._boundaries.find(self._wall_marker)), self._W.sub(0))
        # Driving lid
        u_lid = dolfinx.fem.Function(self._V)
        u_lid.interpolate(lambda x: (np.ones(x[0].shape), np.zeros(x[1].shape)))
        bcu_lid = dolfinx.fem.dirichletbc(u_lid, dolfinx.fem.locate_dofs_topological((self._W.sub(0), self._V), fdim, self._boundaries.find(self._lid_marker)), self._W.sub(0))

        # Pressure
        def bottom_left(x):
            return np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0))

        p_reference = dolfinx.fem.Function(self._Q)
        p_reference.interpolate(lambda x: np.zeros(x[0].shape))
        bcp = dolfinx.fem.dirichletbc(p_reference, dolfinx.fem.locate_dofs_geometrical((self._W.sub(1), self._Q), bottom_left), self._W.sub(1))

        return [bcu_walls, bcu_lid, bcp]
    

    def get_problem_formulation(self, parameters):
        # Form
        w_trial = dolfinx.fem.Function(self._W) # TODO check ufl.TrialFunctions(self._W)
        w_test = ufl.TestFunction(self._W)
        (u, p) = ufl.split(w_trial)
        (v, q) = ufl.split(w_test)

        # Bilinear parts
        a_1 = ufl.inner(ufl.grad(v), parameters.nu * ufl.grad(u)) * ufl.dx
        a_2 = - p * ufl.div(v) / parameters.rho * ufl.dx
        a_3 = q * ufl.div(u) * ufl.dx

        # Non-linear part
        b = ufl.inner(v, ufl.grad(u) * u) * ufl.dx
        F = a_1 + a_2 + a_3 + b

        return F, w_trial
    

    def set_solver_options(self, solver):
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-6
        solver.report = True

        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"# NOTE "bcgs" "preonly"
        opts[f"{option_prefix}pc_type"] = "lu" # NOTE "none" "lu"
        ksp.setFromOptions()


    def interpolated_velocity(self, solution_u, parameters):
        with  HarmonicMeshMotion(self._mesh, 
                    self._boundaries, 
                    [self._wall_marker, self._lid_marker], 
                    [parameters.transform, parameters.transform], 
                    reset_reference=True, 
                    is_deformation=True):
            V_interp = dolfinx.fem.VectorFunctionSpace(self._mesh, ("Lagrange", 1))
            interpolated_u = dolfinx.fem.Function(V_interp)
            u_expr = dolfinx.fem.Expression(solution_u, V_interp.element.interpolation_points())
            interpolated_u.interpolate(u_expr)

            return interpolated_u
    

    def save_results(self, solution_vel, solution_p, parameters):
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True, parents=True)

        filename_pressure = results_folder / "lid_driven_cavity_flow_pressure"
        filename_velocity = results_folder / "lid_driven_cavity_flow_velocity"

        with  HarmonicMeshMotion(self._mesh, 
                    self._boundaries, 
                    [self._wall_marker, self._lid_marker], 
                    [parameters.transform, parameters.transform], 
                    reset_reference=True, 
                    is_deformation=True):
            with dolfinx.io.XDMFFile(self._mesh.comm, filename_velocity.with_suffix(".xdmf"), "w") as xdmf:
                xdmf.write_mesh(self._mesh)
                xdmf.write_function(solution_vel)

            with dolfinx.io.XDMFFile(self._mesh.comm, filename_pressure.with_suffix(".xdmf"), "w") as xdmf:
                xdmf.write_mesh(self._mesh)
                xdmf.write_function(solution_p)
    

    def solve(self, parameters=Parameters()):
         with  HarmonicMeshMotion(self._mesh, 
                    self._boundaries, 
                    [self._wall_marker, self._lid_marker], 
                    [parameters.transform, parameters.transform], 
                    reset_reference=True, 
                    is_deformation=False):
            F, w_trial = self.get_problem_formulation(parameters)
            
            bcs = self.get_boundary_conditions()

            # Nonlinear problem assembly
            problem = dolfinx.fem.petsc.NonlinearProblem(F, w_trial, bcs)

            # Nonlinear problem solution
            solver = NewtonSolver(MPI.COMM_WORLD, problem)
            self.set_solver_options(solver)

            # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

            n, converged = solver.solve(w_trial)

            solution_u = w_trial.sub(0).collapse()
            solution_p = w_trial.sub(1).collapse()

            return solution_u, solution_p


comm = MPI.COMM_WORLD
gdim = 2 # dimension of the model
gmsh_model_rank = 0

mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh.msh", comm,
                                    gmsh_model_rank, gdim=gdim)

# FEM solve
parameters = Parameters(theta=np.pi/6, a=2)
problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags, facet_tags,
                                             HarmonicMeshMotion)

solution_u, solution_p = problem_parametric.solve(parameters)
problem_parametric.save_results(problem_parametric.interpolated_velocity(solution_u, parameters), solution_p, parameters)


# POD

def generate_training_set(samples=[5, 5, 5]):
    training_set_0 = np.linspace(0.5, 2.5, samples[0])
    training_set_1 = np.linspace(0.5, 2.5, samples[1])
    training_set_2 = np.linspace(np.pi/2, np.pi/10, samples[2])
    training_set = np.array(list(product(training_set_0,
                                        training_set_1,
                                        training_set_2)))
    return training_set


training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

Nmax = 100 # the number of basis functions

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
snapshots_u_matrix = rbnicsx.backends.FunctionsList(problem_parametric._V)
snapshots_p_matrix = rbnicsx.backends.FunctionsList(problem_parametric._Q)

print("")

for (params_index, values) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(params_index+1), fill="#"))

    params = Parameters(*values)
    print("Parameter number ", (params_index+1), "of", training_set.shape[0])
    print("high fidelity solve for params =", params)
    snapshot_u, snapshot_p = problem_parametric.solve(params)

    print("update snapshots matrix")
    snapshots_u_matrix.append(snapshot_u)
    snapshots_p_matrix.append(snapshot_p)

    print("")
    
def inner_product_action(fun_j):
        def _(fun_i):
            return fun_i.vector.dot(fun_j.vector)
        return _

print(rbnicsx.io.TextLine("perform POD", fill="#"))
eigenvalues_u, modes_u, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_u_matrix,
                                    inner_product_action,
                                    N=Nmax, tol=1.e-6)

eigenvalues_p, modes_p, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_p_matrix,
                                    inner_product_action,
                                    N=Nmax, tol=1.e-6)

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))

def plot_eigenvalue_decay(ax, eigenvalues, modes, title):
    positive_eigenvalues = np.where(eigenvalues > 0., eigenvalues, np.nan)
    singular_values = np.sqrt(positive_eigenvalues)

    xint = list()
    yval = list()

    for x, y in enumerate(eigenvalues[:len(modes)]):
        yval.append(y)
        xint.append(x+1)

    ax.plot(xint, yval, "^-", color="tab:blue")
    ax.set_xlabel("Eigenvalue number", fontsize=18)
    ax.set_ylabel("Eigenvalue", fontsize=18)
    ax.set_xticks(xint)
    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.set_yscale("log")
    ax.set_title(f"{title}", fontsize=24)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
plot_eigenvalue_decay(ax1, eigenvalues_u, modes_u, "Velocity eigenvalues decay")
plot_eigenvalue_decay(ax2, eigenvalues_p, modes_p, "Pressure eigenvalues decay")
plt.tight_layout()
plt.savefig("eigenvalue_decay.png", dpi=300)
    

# POD Ends ###