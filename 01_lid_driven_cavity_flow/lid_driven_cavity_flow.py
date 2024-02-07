from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion
from dolfinx.nls.petsc import NewtonSolver

import dolfinx
import ufl

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from pathlib import Path

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
        opts[f"{option_prefix}ksp_type"] = "bcgs" # NOTE "preonly"
        opts[f"{option_prefix}pc_type"] = "none" # NOTE "lu"
        ksp.setFromOptions()


    def interpolate_velocity(self, w_trial):
        V_interp = dolfinx.fem.VectorFunctionSpace(self._mesh, ("Lagrange", 1))
        solution_u = dolfinx.fem.Function(V_interp)
        u_expr = dolfinx.fem.Expression(w_trial.sub(0).collapse(), V_interp.element.interpolation_points())
        solution_u.interpolate(u_expr)

        return solution_u
    

    def save_results(self, solution_vel, solution_p, parameters):
        results_folder = Path("results")
        results_folder.mkdir(exist_ok=True, parents=True)

        filename_pressure = results_folder / "lid_driven_cavity_flow_pressure"
        filename_velocity = results_folder / "lid_driven_cavity_flow_velocity"

        with  HarmonicMeshMotion(self._mesh, 
                    self._subdomains, 
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

            dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

            n, converged = solver.solve(w_trial)

            solution_u = self.interpolate_velocity(w_trial)
            solution_p = w_trial.sub(1).collapse()

            self.save_results(solution_u, solution_p, parameters)

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

problem_parametric.solve(parameters)
