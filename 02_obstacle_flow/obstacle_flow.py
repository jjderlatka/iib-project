from obstacle_flow_mesh import Parameters, fluid_marker, inlet_marker, outlet_marker, wall_marker, obstacle_marker

import dolfinx
import dolfinx.fem.petsc
import ufl

from mpi4py import MPI
from petsc4py import PETSc

from tqdm import tqdm
import numpy as np

from pathlib import Path

class Problem():
    def __init__(self, mesh, subdomains, boundaries):
        self._mesh = mesh
        self.gdim = self._mesh.geometry.dim
        self._subdomains = subdomains # NOTE cell_markers
        self._boundaries = boundaries # NOTE facet_markers

        # Function spaces
        P2 = ufl.VectorElement("Lagrange", self._mesh.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", self._mesh.ufl_cell(), 1)
        UP = P2 * P1

        # self._W = dolfinx.fem.FunctionSpace(self._mesh, UP)
        # self._V, _ = self._W.sub(0).collapse()
        # self._Q, _ = self._W.sub(1).collapse()
        self._V = dolfinx.fem.FunctionSpace(mesh, P2)
        self._Q = dolfinx.fem.FunctionSpace(mesh, P1)


    def get_boundary_conditions(self, t):
        fdim = self._mesh.topology.dim - 1
       
        class InletVelocity():
            def __init__(self, t):
                self.t = t

            def __call__(self, x):
                values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
                values[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
                return values


        # Inlet
        u_inlet = dolfinx.fem.Function(self._V)
        inlet_velocity = InletVelocity(t)
        u_inlet.interpolate(inlet_velocity)
        bcu_inflow = dolfinx.fem.dirichletbc(u_inlet, dolfinx.fem.locate_dofs_topological(self._V, fdim, self._boundaries.find(inlet_marker)))
        # Walls
        u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
        bcu_walls = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(self._V, fdim, self._boundaries.find(wall_marker)), self._V)
        # Obstacle
        bcu_obstacle = dolfinx.fem.dirichletbc(u_nonslip, dolfinx.fem.locate_dofs_topological(self._V, fdim, self._boundaries.find(obstacle_marker)), self._V)
        bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
        # Outlet
        bcp_outlet = dolfinx.fem.dirichletbc(PETSc.ScalarType(0), dolfinx.fem.locate_dofs_topological(self._Q, fdim, self._boundaries.find(outlet_marker)), self._Q)
        bcp = [bcp_outlet]

        # TODO clean up modularisation of the code
        return bcu, bcp, u_inlet, inlet_velocity

    
    def solve(self, parameters):
        t = 0
        T = 8                         # Final time
        dt = 1 / 1600                 # Time step size
        num_steps = int(T / dt)
        # k = Constant(mesh, PETSc.ScalarType(dt)) # TODO
        k = PETSc.ScalarType(dt)

        bcu, bcp, u_inlet, inlet_velocity = self.get_boundary_conditions(t)
        
        # TODO go with someone over what is happening here and why this and not something else

        # w_trial = dolfinx.fem.Function(self._W) # TODO check ufl.TrialFunctions(self._W)
        # w_test = ufl.TestFunction(self._W)
        # (u, p) = ufl.split(w_trial)
        # (v, q) = ufl.split(w_test)
        u = ufl.TrialFunction(self._V)
        v = ufl.TestFunction(self._V)
        p = ufl.TrialFunction(self._Q)
        q = ufl.TestFunction(self._Q)
        
        u_ = dolfinx.fem.Function(self._V)
        u_.name = "u"
        u_s = dolfinx.fem.Function(self._V)
        u_n = dolfinx.fem.Function(self._V)
        u_n1 = dolfinx.fem.Function(self._V)
        
        p_ = dolfinx.fem.Function(self._Q)
        p_.name = "p"
        phi = dolfinx.fem.Function(self._Q)

        # First step
        f = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0, 0)))
        F1 = parameters.rho / k * ufl.dot(u - u_n, v) * ufl.dx
        F1 += ufl.inner(ufl.dot(1.5 * u_n - 0.5 * u_n1, 0.5 * ufl.nabla_grad(u + u_n)), v) * ufl.dx
        F1 += 0.5 * parameters.mu * ufl.inner(ufl.grad(u + u_n), ufl.grad(v)) * ufl.dx - ufl.dot(p_, ufl.div(v)) * ufl.dx
        F1 += ufl.dot(f, v) * ufl.dx
        a1 = dolfinx.fem.form(ufl.lhs(F1))
        L1 = dolfinx.fem.form(ufl.rhs(F1))
        A1 = dolfinx.fem.petsc.create_matrix(a1)
        b1 = dolfinx.fem.petsc.create_vector(L1)

        # Second step
        a2 = dolfinx.fem.form(ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx)
        L2 = dolfinx.fem.form(-parameters.rho / k * ufl.dot(ufl.div(u_s), q) * ufl.dx)
        A2 = dolfinx.fem.petsc.assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = dolfinx.fem.petsc.create_vector(L2)

        # Third step
        a3 = dolfinx.fem.form(parameters.rho * ufl.dot(u, v) * ufl.dx)
        L3 = dolfinx.fem.form(parameters.rho * ufl.dot(u_s, v) * ufl.dx - k * ufl.dot(ufl.nabla_grad(phi), v) * ufl.dx)
        A3 = dolfinx.fem.petsc.assemble_matrix(a3)
        A3.assemble()
        b3 = dolfinx.fem.petsc.create_vector(L3)

        # Solver for step 1
        solver1 = PETSc.KSP().create(MPI.COMM_WORLD)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        # Solver for step 2
        solver2 = PETSc.KSP().create(MPI.COMM_WORLD)
        solver2.setOperators(A2)
        solver2.setType(PETSc.KSP.Type.MINRES)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")

        # Solver for step 3
        solver3 = PETSc.KSP().create(MPI.COMM_WORLD)
        solver3.setOperators(A3)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

        folder = Path("results")
        folder.mkdir(exist_ok=True, parents=True)
        vtx_u = dolfinx.io.VTXWriter(mesh.comm, "dfg2D-3-u.bp", [u_], engine="BP4")
        vtx_p = dolfinx.io.VTXWriter(mesh.comm, "dfg2D-3-p.bp", [p_], engine="BP4")
        vtx_u.write(t)
        vtx_p.write(t)

        for i in tqdm(range(num_steps)):
            # Update current time step
            t += dt
            # Update inlet velocity
            inlet_velocity.t = t
            u_inlet.interpolate(inlet_velocity)

            # Step 1: Tentative velocity step
            A1.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(A1, a1, bcs=bcu)
            A1.assemble()
            with b1.localForm() as loc:
                loc.set(0)
            dolfinx.fem.petsc.assemble_vector(b1, L1)
            dolfinx.fem.petsc.apply_lifting(b1, [a1], [bcu])
            b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b1, bcu)
            solver1.solve(b1, u_s.vector)
            u_s.x.scatter_forward()

            # Step 2: Pressure corrrection step
            with b2.localForm() as loc:
                loc.set(0)
            dolfinx.fem.petsc.assemble_vector(b2, L2)
            dolfinx.fem.petsc.apply_lifting(b2, [a2], [bcp])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b2, bcp)
            solver2.solve(b2, phi.vector)
            phi.x.scatter_forward()

            p_.vector.axpy(1, phi.vector)
            p_.x.scatter_forward()

            # Step 3: Velocity correction step
            with b3.localForm() as loc:
                loc.set(0)
            dolfinx.fem.petsc.assemble_vector(b3, L3)
            b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            solver3.solve(b3, u_.vector)
            u_.x.scatter_forward()

            # Write solutions to file
            vtx_u.write(t)
            vtx_p.write(t)

            # Update variable with solution form this time step
            with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n, u_n1.vector.localForm() as loc_n1:
                loc_n.copy(loc_n1)
                loc_.copy(loc_n)

        vtx_u.close()
        vtx_p.close()

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    gdim = 2 # dimension of the model
    gmsh_model_rank = 0

    mesh, cell_tags, facet_tags = \
        dolfinx.io.gmshio.read_from_msh("obstacle_mesh.msh", comm,
                                        gmsh_model_rank, gdim=gdim)
    
    problem = Problem(mesh, cell_tags, facet_tags)
    problem.solve(Parameters())


# TODO questions:
#   1. elements
#   2. time discretization method
#       a. Why Crank-Nicholson? Is there a single best one?
#       b. What on earth happens in the Crank-Nicholson derivation