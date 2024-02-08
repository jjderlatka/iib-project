import gmsh

from math import pi
import numpy as np


class Parameters():
    def __init__(self, a=1, b=1, theta=pi / 2):
        self.a = a
        self.b = b
        self.theta = theta


    def __repr__(self):
        return f"a={self.a},b={self.b},theta={self.theta:.2f}"


    def matrix(self):
        return np.array([[self.a, self.b * np.cos(self.theta)],
                         [0,      self.b * np.sin(self.theta)]])
    

    def transform(self, x):
        return self.matrix() @ x[:gdim]


gdim = 2 # dimension of the model
parameters = Parameters()

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

fluid_marker, lid_marker, wall_marker = 1, 2, 3
gmsh.model.addPhysicalGroup(2, [loop], fluid_marker)
gmsh.model.addPhysicalGroup(1, [CD], lid_marker)
gmsh.model.addPhysicalGroup(1, [AB, BC, DA], wall_marker)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(gdim)

gmsh.write("mesh.msh")
