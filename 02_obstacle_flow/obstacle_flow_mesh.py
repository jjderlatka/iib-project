# https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html#mesh-generation

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

import dolfinx.io
from pathlib import Path

import gmsh

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

fluid_marker = 1
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
markers = [inlet_marker, outlet_marker, wall_marker, obstacle_marker]

class Parameters():
    def __init__(self, L=2.2, H=0.41, c_x = 0.2, c_y = 0.2, r = 0.05, nu=0.001, rho=1):
        self.L = L
        self.H = H
        self.c_x = c_x
        self.c_y = c_y
        self.r = r
        # self.mu = Constant(mesh, PETSc.ScalarType(0.001))  # TODO make into a mesh const
        # self.rho = Constant(mesh, PETSc.ScalarType(1))     # TODO
        self.nu = PETSc.ScalarType(nu)       # NOTE Dynamic viscosity
        self.rho = PETSc.ScalarType(rho)     # NOTE Density

    def __repr__(self):
        return f"(L={self.L},H={self.H},c_x={self.c_x},c_y={self.c_y},r={self.r},nu={self.nu:.3f},rho={self.rho:.3f})"

# NOTE all transformations assume a rectangle with lower left corner at (0, -H/2)
def transform(target, reference=Parameters()):
    """Return a list of transformations from reference to target 
    parameters for [inlet_marker, outlet_marker, wall_marker, 
    obstacle_marker] in that order. """
    def inlet_transform(x):
        dx = x[0] * 0.
        dy = (x[1]/reference.H) * (target.H - reference.H)
        return (dx, dy)
    
    def outlet_transform(x):
        dx = (x[0]/reference.L) * (target.L - reference.L)
        dy = (x[1]/reference.H) * (target.H - reference.H)
        return (dx, dy)
    
    def wall_transform(x):
        dx = (x[0]/reference.L) * (target.L - reference.L)
        dy = (x[1]/reference.H) * (target.H - reference.H)
        return (dx, dy)
    
    def obstacle_transform(x):
        absolute_target_c_x, absolute_target_c_y = target.c_x, - target.H/2 + target.c_y
        absolute_reference_c_x, absolute_reference_c_y = reference.c_x, - reference.H/2 + reference.c_y
        dx = (absolute_target_c_x - absolute_reference_c_x) + (target.r / reference.r - 1) * (x[0] - absolute_reference_c_x)
        dy = (absolute_target_c_y - absolute_reference_c_y) + (target.r / reference.r - 1) * (x[1] - absolute_reference_c_y)
        return (dx, dy)
    
    return [inlet_transform,
            outlet_transform,
            wall_transform,
            obstacle_transform]


def subdivide_transformation(target, steps=1, reference = Parameters()):
    """Linearly break down the transition from reference to target parameters
    into given number of steps."""
    L = np.linspace(reference.L, target.L, steps + 1)
    H = np.linspace(reference.H, target.H, steps + 1)
    C_X = np.linspace(reference.c_x, target.c_x, steps + 1)
    C_Y = np.linspace(reference.c_y, target.c_y, steps + 1)
    R = np.linspace(reference.r, target.r, steps + 1)

    steps_list = []
    for l, h, c_x, c_y, r in zip(L, H, C_X, C_Y, R):
        steps_list.append(Parameters(l, h, c_x, c_y, r))

    return steps_list


def transform_steps(target, steps=1, reference=Parameters()):
    """Return a function returning list of transformation functions
    for the ith step in a multistep reference to target deformation."""
    steps_list = subdivide_transformation(target, steps, reference)
    def transform_(i):
        assert(0 <= i and i < steps)
        return transform(steps_list[i+1], steps_list[i])
    return transform_


class CompoundedMeshDeformation:
    def __init__(self, meshDeformationContext, mesh, boundaries, bc_markers_list, bc_function_list,
                 reset_reference=False, is_deformation=True, steps=1):
        self.meshDeformationContext = meshDeformationContext
        self.mesh = mesh
        self.boundaries = boundaries
        self.bc_markers_list = bc_markers_list
        self.bc_function_list = bc_function_list
        self.reset_reference = reset_reference
        self.is_deformation = is_deformation
        
        self.steps = steps
        self.instances = []


    def __enter__(self):
        # Create and enter a new instance for each level of nesting
        for i in range(self.steps):
            instance = self.meshDeformationContext(self.mesh,
                                                   self.boundaries,
                                                   self.bc_markers_list,
                                                   self.bc_function_list(i),
                                                   self.reset_reference,
                                                   self.is_deformation)
            instance.__enter__()
            self.instances.append(instance)
        # Return the last instance entered
        return self.instances[-1]

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exit instances in reverse order to match the nesting
        while self.instances:
            instance = self.instances.pop()
            instance.__exit__(exc_type, exc_val, exc_tb)


def generate_mesh(parameters, file_name):
    gdim = 2
    
    rectangle = gmsh.model.occ.addRectangle(0, -parameters.H/2, 0, parameters.L, parameters.H, tag=1)
    obstacle = gmsh.model.occ.addDisk(parameters.c_x, -parameters.H/2 + parameters.c_y, 0, parameters.r, parameters.r)

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
        if np.allclose(center_of_mass, [0, 0, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [parameters.L, 0, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [parameters.L / 2, parameters.H/2, 0]) or np.allclose(center_of_mass, [parameters.L / 2, -parameters.H/2, 0]):
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


def mesh_xdmf():
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("obstacle_mesh.msh", MPI.COMM_WORLD, 0, gdim=2)

    results_folder = Path("results/02")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = results_folder / "obstacle_mesh"
    with dolfinx.io.XDMFFile(mesh.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)


def comparison():
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("obstacle_mesh.msh", MPI.COMM_WORLD, 0, gdim=2)

    results_folder = Path("results/02")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = results_folder / "obstacle_mesh_deformed"

    print("Attempting to deform the mesh")
    p = Parameters(H=1, r=0.1, c_x=0.4, c_y=0.5) # NOTE TODO when L is brought down sufficiently far (1.5), the mesh degenerates around the obstacle

    [p1, p2, p3] = subdivide_transformation(p, 2)

    with HarmonicMeshMotion(mesh, facet_tags, markers, transform(p3, p1), reset_reference=True, is_deformation=True):
        with dolfinx.io.XDMFFile(mesh.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
            xdmf.write_mesh(mesh)

    filename = results_folder / "obstacle_mesh_deformed_twice"
    with HarmonicMeshMotion(mesh, facet_tags, markers, transform(p2, p1), reset_reference=True, is_deformation=True):
        with HarmonicMeshMotion(mesh, facet_tags, markers, transform(p3, p2), reset_reference=True, is_deformation=True):
            with dolfinx.io.XDMFFile(mesh.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
                xdmf.write_mesh(mesh)


def deformed_mesh_xdmf():
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("obstacle_mesh.msh", MPI.COMM_WORLD, 0, gdim=2)

    results_folder = Path("results/02")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = results_folder / "obstacle_mesh_deformed"

    print("Attempting to deform the mesh")
    p = Parameters(H=1, r=0.1, c_x=0.4, c_y=0.5) # NOTE TODO when L is brought down sufficiently far (1.5), the mesh degenerates around the obstacle
    steps = 8
    with CompoundedMeshDeformation(HarmonicMeshMotion,
                                   mesh,
                                   facet_tags,
                                   markers,
                                   transform_steps(p, steps),
                                   reset_reference=True,
                                   is_deformation=True,
                                   steps=steps):
        with dolfinx.io.XDMFFile(mesh.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
            xdmf.write_mesh(mesh)



if __name__ == "__main__":
    parameters = Parameters()
    file_name = "obstacle_mesh.msh"

    gmsh.initialize()
    if MPI.COMM_WORLD.Get_rank() == 0:
        generate_mesh(parameters, file_name)
        mesh_xdmf()
        deformed_mesh_xdmf()
