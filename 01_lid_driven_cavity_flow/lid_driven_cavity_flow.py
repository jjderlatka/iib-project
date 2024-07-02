from lid_driven_cavity_flow_mesh import Parameters, fluid_marker, lid_marker, wall_marker

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion
from dolfinx.nls.petsc import NewtonSolver

import rbnicsx.io, rbnicsx.backends, rbnicsx.online
import dolfinx
import ufl

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from itertools import product


class ProblemOnDeformedDomain():
    def __init__(self, mesh, subdomains, boundaries, meshDeformationContext):
        # Mesh, Subdomians and Boundaries, Mesh deformation
        self._mesh = mesh
        self.gdim = self._mesh.geometry.dim
        self._subdomains = subdomains # NOTE cell_markers
        self._facet_tags = boundaries # NOTE facet_markers
        self.meshDeformationContext = meshDeformationContext

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
        bcu_walls = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological((self._W.sub(0), self._V), fdim, self._facet_tags.find(wall_marker)), self._W.sub(0))
        # Driving lid
        u_lid = dolfinx.fem.Function(self._V)
        u_lid.interpolate(lambda x: (np.ones(x[0].shape), np.zeros(x[1].shape)))
        bcu_lid = dolfinx.fem.dirichletbc(u_lid, dolfinx.fem.locate_dofs_topological((self._W.sub(0), self._V), fdim, self._facet_tags.find(lid_marker)), self._W.sub(0))

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
    

    def configure_solver(self, problem):
        # TODO ask about how exactly it would look like if this was mismatched, how is that used? How does the solving process in general look line, when the mesh is distributed?
        # Before I changed this, it consistently failed, but only on the 115 parameter, regardless of the number of processes.
        solver = NewtonSolver(self._mesh.comm, problem)

        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-6
        solver.report = True

        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"# NOTE "bcgs" "preonly"
        opts[f"{option_prefix}pc_type"] = "lu" # NOTE "none" "lu"
        ksp.setFromOptions()

        return solver
    

    def interpolated_velocity(self, solution_u):

        # TODO I tested this interpolation gives the same result with and without mesh deformation, but check with Nirav -> notebook answer
        V_interp = dolfinx.fem.VectorFunctionSpace(self._mesh, ("Lagrange", 1))
        interpolated_u = dolfinx.fem.Function(V_interp)
        u_expr = dolfinx.fem.Expression(solution_u, V_interp.element.interpolation_points())
        interpolated_u.interpolate(u_expr)
        
        return interpolated_u
    

    def save_results(self, parameters, solution_vel=None, solution_p=None, name_suffix=""):        
        results_folder = Path("results/01")
        results_folder.mkdir(exist_ok=True, parents=True)

        filename_pressure = results_folder / ( "lid_driven_cavity_flow_pressure" + name_suffix )
        filename_velocity = results_folder / ( "lid_driven_cavity_flow_velocity" + name_suffix )

        with  self.meshDeformationContext(self._mesh, 
                                        self._facet_tags, 
                                        [wall_marker, lid_marker], 
                                        [parameters.transform, parameters.transform], 
                                        reset_reference=True, 
                                        is_deformation=False):
        
            if solution_vel is not None:
                with dolfinx.io.XDMFFile(self._mesh.comm, filename_velocity.with_suffix(".xdmf"), "w") as xdmf:
                    xdmf.write_mesh(self._mesh)
                    xdmf.write_function(solution_vel)
            if solution_p is not None:
                with dolfinx.io.XDMFFile(self._mesh.comm, filename_pressure.with_suffix(".xdmf"), "w") as xdmf:
                    xdmf.write_mesh(self._mesh)
                    xdmf.write_function(solution_p)
    

    def solve(self, parameters=Parameters()):
        with  self.meshDeformationContext(self._mesh, 
                    self._facet_tags, 
                    [wall_marker, lid_marker], 
                    [parameters.transform, parameters.transform], 
                    reset_reference=True, 
                    is_deformation=False):
            F, w_trial = self.get_problem_formulation(parameters)
            
            bcs = self.get_boundary_conditions()

            # Nonlinear problem assembly
            problem = dolfinx.fem.petsc.NonlinearProblem(F, w_trial, bcs)

            # Nonlinear problem solution
            solver = self.configure_solver(problem)

            # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

            n, converged = solver.solve(w_trial)

            solution_u = w_trial.sub(0).collapse()
            solution_p = w_trial.sub(1).collapse()

            return solution_u, solution_p
        

    def get_dofs(self):
        # dofs * block size (for vector valued functions, vector dimensions in each dof)
        u_dofs = self._V.dofmap.index_map.size_global * self._V.dofmap.index_map_bs
        p_dofs = self._Q.dofmap.index_map.size_global * self._Q.dofmap.index_map_bs

        return u_dofs, p_dofs


class PODANNReducedProblem():
    def __init__(self, full_problem, function_space):
        self._full_problem = full_problem
        self._function_space = function_space
        self._basis_functions = rbnicsx.backends.FunctionsList(function_space)

        self.input_scaling_range = [-1., 1.]
        self.input_range = None
        self.output_scaling_range = [-1., 1.]
        self.output_range = None


    # TODO look at the difference between using different norms
    def _inner_product_action(self, fun_j):
        def _(fun_i):
            return fun_i.vector.dot(fun_j.vector)
        return _


    def compute_norm(self, function):
        return np.sqrt(self._inner_product_action(function)(function))
    

    def norm_error(self, reference_value, value):
        # TODO handle in a more elegant way
        absolute_error = dolfinx.fem.Function(self._function_space)
        absolute_error.x.array[:] = reference_value.x.array - value.x.array
        # absolute_error = value - reference_value
        return self.compute_norm(absolute_error)/self.compute_norm(reference_value)
    

    def norm_error_deformed_context(self, parameters, reference_value, value):
        with self._full_problem.meshDeformationContext(self._full_problem._mesh, 
                    self._full_problem._facet_tags, 
                    [wall_marker, lid_marker], 
                    [parameters.transform, parameters.transform], 
                    reset_reference=True, 
                    is_deformation=True):
            return self.norm_error(reference_value, value)


    def set_reduced_basis(self, functions):
        self._basis_functions.extend(functions)


    def rb_dimension(self):
        return len(self._basis_functions)
    

    def project_snapshot(self, solution, N=None):
        if N is None:
            N = self.rb_dimension()

        projected_snapshot = rbnicsx.online.create_vector(N)
        A = rbnicsx.backends.\
            project_matrix(self._inner_product_action,
                           self._basis_functions[:N]) 
        F = rbnicsx.backends.\
            project_vector(self._inner_product_action(solution),
                           self._basis_functions[:N])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot)
        return projected_snapshot


    def reconstruct_solution(self, reduced_solution):
        """Reconstruct a RB projection of a snapshot back to high fidelity space"""
        return self._basis_functions[:reduced_solution.size] * reduced_solution
    

    def set_scaling_range(self, parameters, solutions):
        self.input_range = np.stack((np.min(parameters, axis=0), np.max(parameters, axis=0)), axis=0)
        self.output_range = np.stack((np.min(solutions, axis=0), np.max(solutions, axis=0)), axis=0)