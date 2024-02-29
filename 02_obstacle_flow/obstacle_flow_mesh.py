# https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html#mesh-generation

import gmsh

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

fluid_marker = 1
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5

class Parameters():
    def __init__(self, L=2.2, H=0.41, c_x = 0.2, c_y = 0.2, r = 0.05, mu=0.001, rho=1):
        self.L = L
        self.H = H
        self.c_x = c_x
        self.c_y = c_y
        self.r = r
        # self.mu = Constant(mesh, PETSc.ScalarType(0.001))  # TODO make into a mesh const
        # self.rho = Constant(mesh, PETSc.ScalarType(1))     # TODO
        self.mu = PETSc.ScalarType(mu)       # NOTE Dynamic viscosity
        self.rho = PETSc.ScalarType(rho)     # NOTE Density

def gerenate_mesh(parameters, file_name):
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, parameters.L, parameters.H, tag=1)
    obstacle = gmsh.model.occ.addDisk(parameters.c_x, parameters.c_y, 0, parameters.r, parameters.r)

    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

    inflow, outflow, walls, obstacle = [], [], [], []

    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, parameters.H / 2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [parameters.L, parameters.H / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [parameters.L / 2, parameters.H, 0]) or np.allclose(center_of_mass, [parameters.L / 2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])

    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

    # Create distance field from obstacle.
    # Add threshold of mesh sizes based on the distance field
    # LcMax -                  /--------
    #                      /
    # LcMin -o---------/
    #        |         |       |
    #       Point    DistMin DistMax
    res_min = parameters.r / 3

    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * parameters.H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", parameters.r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * parameters.H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(file_name)


if __name__ == "__main__":
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    parameters = Parameters()
    file_name = "obstacle_mesh.msh"

    gmsh.initialize()
    if mesh_comm.rank == model_rank:
        gerenate_mesh(parameters, file_name)