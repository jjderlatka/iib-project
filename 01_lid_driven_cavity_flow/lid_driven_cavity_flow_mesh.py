import gmsh
from petsc4py import PETSc

from math import pi
import numpy as np

fluid_marker, lid_marker, wall_marker = 1, 2, 3
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
    

    def transform(self, x, gdim=2):
        return self.matrix() @ x[:gdim]


def generate_mesh(parameters, file_name):
    gdim = 2 # dimension of the model

    # Creating the mesh
    gmsh.initialize()

    mesh_size = 0.05 # 0.04

    gmsh.model.add("parallelogram")

    A = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    B = gmsh.model.geo.addPoint(parameters.a, 0, 0, mesh_size)
    C = gmsh.model.geo.addPoint(parameters.b * np.cos(parameters.theta) + parameters.a, parameters.b * np.sin(parameters.theta), 0, mesh_size)
    D = gmsh.model.geo.addPoint(parameters.b * np.cos(parameters.theta), parameters.b * np.sin(parameters.theta), 0, mesh_size)

    AB = gmsh.model.geo.addLine(A, B)
    BC = gmsh.model.geo.addLine(B, C)
    CD = gmsh.model.geo.addLine(C, D)
    DA = gmsh.model.geo.addLine(D, A)

    loop = gmsh.model.geo.addCurveLoop([AB, BC, CD, DA])
    gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.addPhysicalGroup(2, [loop], fluid_marker)
    gmsh.model.addPhysicalGroup(1, [CD], lid_marker)
    gmsh.model.addPhysicalGroup(1, [AB, BC, DA], wall_marker)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(gdim)

    gmsh.write(file_name)


if __name__ == "__main__":
    parameters = Parameters()
    file_name = "mesh.msh"
    reference_mesh = generate_mesh(parameters, file_name)