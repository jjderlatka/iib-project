from lid_driven_cavity_flow_mesh import Parameters, fluid_marker, lid_marker, wall_marker
from lid_driven_cavity_flow import ProblemOnDeformedDomain, PODANNReducedProblem

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion
from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.train_validate_test.train_validate_test import train_nn, validate_nn

import dolfinx
import rbnicsx.io
from petsc4py import PETSc
import ufl

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from mpi4py import MPI

import pickle
from itertools import product
from pathlib import Path
from time import process_time_ns

import numpy as np


class Timer:
    def __init__(self):
        self.__start_time = process_time_ns()
        self.__timestamps = {}

    
    def timestamp(self, label):
        assert(label not in self.__timestamps)
        timestamp = process_time_ns() - self.__start_time
        self.__timestamps[label] = timestamp


    def get_timestamps(self):
        return self.__timestamps


class CustomDataset(Dataset):
    def __init__(self, parameters_list, solutions_list):
        assert(len(parameters_list) == len(solutions_list))
        self.parameters = parameters_list
        self.solutions = solutions_list


    def __len__(self):
        return len(self.parameters)


    def __getitem__(self, idx):
        return self.parameters[idx], self.solutions[idx]
    

def mpi_print(s):
    print(f"[Rank {MPI.COMM_WORLD.Get_rank()}]: {s}")


# TODO implementing __setstate__ and __getstate__ would let data to be pickled and therefore bcasted and gathered easily
def gather_functions_list(functions_list, root_rank=0):
    function_space = functions_list.function_space

    coefficients = []
    for func in functions_list:
        # TODO: could be removing copied functions from the original functions_list, but FunctionsList class doesn't have any method for removing other than erasing all
        coefficients.append(func.vector[:])
    
    coefficients = MPI.COMM_WORLD.gather(coefficients, root=root_rank)

    if MPI.COMM_WORLD.Get_rank() == root_rank:
        coefficients = np.concatenate(coefficients, axis=0)

        functions_list = rbnicsx.backends.FunctionsList(function_space)
        for (i, snapshot) in enumerate(coefficients):
            func = dolfinx.fem.Function(function_space)
            func.vector.setArray(snapshot)
            functions_list.append(func)

        return functions_list
    
    else:
        return None
    

def bcast_functions_list(functions_list, function_space, root_rank=0):
    coefficients = []

    if MPI.COMM_WORLD.Get_rank() == root_rank:
        assert(functions_list.function_space is function_space)
        for func in functions_list:
            coefficients.append(func.vector[:])

    coefficients = MPI.COMM_WORLD.bcast(coefficients, root=root_rank)

    functions_list = rbnicsx.backends.FunctionsList(function_space)
    for (i, snapshot) in enumerate(coefficients):
        func = dolfinx.fem.Function(function_space)
        func.vector.setArray(snapshot)
        functions_list.append(func)

    return functions_list


def generate_parameters_values_list(samples):
    training_set_0 = np.linspace(0.5, 2.5, samples[0])
    training_set_1 = np.linspace(0.5, 2.5, samples[1])
    training_set_2 = np.linspace(np.pi/2, np.pi/10, samples[2])
    training_set = np.array(list(product(training_set_0,
                                        training_set_1,
                                        training_set_2)))
    return training_set


def generate_parameters_list(parameteres_values):
    return np.array([Parameters(*vals) for vals in parameteres_values])


def generate_solutions_list(global_training_set):
    snapshots_u_matrix = rbnicsx.backends.FunctionsList(problem_parametric._V)
    snapshots_p_matrix = rbnicsx.backends.FunctionsList(problem_parametric._Q)

    # TODO discuss: what are the benefits or choosing every 10th vs continous sections of 10. Any caching benefit? Any way to use the similarity between solving similar solutions? Maybe iterative methods could use the solution to neighbouring parameter as the starting point?
    # TODO not necessarily helpful atm, as LU is used (am I right?) But maybe could switch to a different solver, slower for one solve, but faster for a range of them
    my_training_set = np.array_split(global_training_set, MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]
    
    for (params_index, params) in enumerate(my_training_set):
        print(rbnicsx.io.TextLine(f"{params_index+1} of {my_training_set.shape[0]}", fill="#"))
        print("High fidelity solve for params =", params)
        snapshot_u, snapshot_p = problem_parametric.solve(params)

        snapshots_u_matrix.append(snapshot_u)
        snapshots_p_matrix.append(snapshot_p)

    mpi_print(f"Rank {MPI.COMM_WORLD.Get_rank()} has {np.shape(snapshots_p_matrix)} snapshots.")
    
    snapshots_u_matrix = gather_functions_list(snapshots_u_matrix, 0)
    snapshots_p_matrix = gather_functions_list(snapshots_p_matrix, 0)

    return snapshots_u_matrix, snapshots_p_matrix


def proper_orthogonal_decomposition(reduced_problem, snapshots_matrix):
    # NOTE POD parallelization: correlation matrix assembly could be paralalleized
    # NOTE SLEPc eigensolve underlying POD could also be parallalized
    
    Nmax = 100 # the number of basis functions

    if MPI.COMM_WORLD.Get_rank() == 0:
        eigenvalues, modes, _ = \
            rbnicsx.backends.\
            proper_orthogonal_decomposition(snapshots_matrix,
                                            reduced_problem._inner_product_action,
                                            N=Nmax, tol=1.e-6)
        
    else:
        eigenvalues, modes = None, None

    eigenvalues, modes = MPI.COMM_WORLD.bcast(eigenvalues, root=0), bcast_functions_list(modes, reduced_problem._function_space, 0)

    reduced_problem.set_reduced_basis(modes)


def generate_rb_solutions_list(global_parameters_set):
    my_parameters_set = np.array_split(global_parameters_set, MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]

    rb_snapshots_u_matrix = np.empty((len(my_parameters_set), reduced_problem_u.rb_dimension()))
    rb_snapshots_p_matrix = np.empty((len(my_parameters_set), reduced_problem_p.rb_dimension()))

    for (params_index, params) in enumerate(my_parameters_set):
        print(rbnicsx.io.TextLine(f"{params_index+1} of {len(my_parameters_set)}", fill="#"))
        print("High fidelity solve for params =", params)
        snapshot_u, snapshot_p = problem_parametric.solve(params)

        rb_snapshot_u = reduced_problem_u.project_snapshot(snapshot_u)
        rb_snapshot_p = reduced_problem_p.project_snapshot(snapshot_p)

        rb_snapshots_u_matrix[params_index, :] = rb_snapshot_u
        rb_snapshots_p_matrix[params_index, :] = rb_snapshot_p

    rb_snapshots_u_matrix = MPI.COMM_WORLD.allgather(rb_snapshots_u_matrix)
    rb_snapshots_p_matrix = MPI.COMM_WORLD.allgather(rb_snapshots_p_matrix)
    
    rb_snapshots_u_matrix = np.concatenate(rb_snapshots_u_matrix, axis=0)
    rb_snapshots_p_matrix = np.concatenate(rb_snapshots_p_matrix, axis=0)

    return rb_snapshots_u_matrix, rb_snapshots_p_matrix


def prepare_test_and_training_sets(parameters_list, solutions_u, solutions_p):
    num_training_samples = int(0.7 * len(parameters_list))
    num_validation_samples = len(parameters_list) - num_training_samples

    paramteres_values_list = np.array([p.to_numpy() for p in parameters_list])

    input_training_set = paramteres_values_list[:num_training_samples, :]
    solutions_u_training_set = solutions_u[:num_training_samples, :]
    solutions_p_training_set = solutions_p[:num_training_samples, :]
    training_dataset_u = CustomDataset(input_training_set, solutions_u_training_set)
    training_dataset_p = CustomDataset(input_training_set, solutions_p_training_set)

    input_validation_set = paramteres_values_list[num_training_samples:, :]
    solutions_u_validation_set = solutions_u[num_training_samples:, :]
    solutions_p_validation_set = solutions_p[num_training_samples:, :]
    validation_dataset_u = CustomDataset(input_validation_set, solutions_u_validation_set)
    validation_dataset_p = CustomDataset(input_validation_set, solutions_p_validation_set)

    return training_dataset_u, training_dataset_p, validation_dataset_u, validation_dataset_p


def save_all_timestamps(timer, root_rank=0):
    timestamps = timer.get_timestamps()
    timestamps = MPI.COMM_WORLD.gather(timestamps, root_rank)
    
    if MPI.COMM_WORLD.Get_rank() == root_rank:
        results_folder = Path("results")
        file_path = results_folder / 'timing.pkl'

        if file_path.exists():
            with open(file_path, 'rb') as f:
                all_results = pickle.load(f)
        else:
            all_results = {}
        
        all_results[MPI.COMM_WORLD.Get_size()] = timestamps
        
        with open(file_path, 'wb') as f:
            pickle.dump(all_results, f)


def train_NN(training_dataset, validation_dataset):
    # u only to begin with
    
    device = "cpu"
    batch_size = 64

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    assert(training_dataset[0])
    input_size = len(training_dataset[0][0])
    output_size = len(training_dataset[0][1])
    print(f"NN {input_size=}, {output_size=}")

    model.double() # TODO remove? Convert the entire model to Double (or would have to convert input and outputs to floats (they're now doubles))
    print(model)

    # TODO investigate the loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # TODO plot the evolution with number of epochs
    # (plot the loss function evolution, but also prediction made with NN trained up to given epoch)
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # TODO make a pull request removing reduced_problem argument from these functions
        train_nn(None, train_dataloader, model, loss_fn, optimizer)
        validate_nn(None, test_dataloader, model, loss_fn)
    print("Model trained!")

    return model


def save_preview(preview_parameter, model, problem_parametric, reduced_problem_u, reduced_problem_p):
    # Infer solution
    X = torch.tensor(preview_parameter.to_numpy())
    rb_pred = model(X)
    rb_pred_vec = PETSc.Vec().createWithArray(rb_pred.detach().numpy(), comm=MPI.COMM_SELF)

    # Full solution
    fem_u, fem_p = problem_parametric.solve(preview_parameter)
    problem_parametric.save_results(preview_parameter, problem_parametric.interpolated_velocity(fem_u), fem_p, name_suffix="_fem")

    # Reduced basis projection of full solution
    rb_snapshot_u = reduced_problem_u.project_snapshot(fem_u)
    rb_snapshot_p = reduced_problem_p.project_snapshot(fem_p)
    reconstructed_u = reduced_problem_u.reconstruct_solution(rb_snapshot_u)
    reconstructed_p = reduced_problem_p.reconstruct_solution(rb_snapshot_p)
    problem_parametric.save_results(preview_parameter, problem_parametric.interpolated_velocity(reconstructed_u), reconstructed_p, name_suffix="_rb")
    # problem_parametric.save_results(p, problem_parametric.interpolated_velocity(fem_u), fem_p, name_suffix="_rb")

    # NN solution
    pred_u = reduced_problem_u.reconstruct_solution(rb_pred_vec) 
    interpolated_pred = problem_parametric.interpolated_velocity(pred_u)
    problem_parametric.save_results(p, interpolated_pred, name_suffix="_pred")

    # Plot difference
    u_diff = problem_parametric.interpolated_velocity(pred_u - reconstructed_u)
    problem_parametric.save_results(p, u_diff, name_suffix="_diff")

    # TODO relative difference

    # Divergence
    divergence_space = dolfinx.fem.FunctionSpace(problem_parametric._mesh, ufl.FiniteElement("DG", problem_parametric._mesh.ufl_cell(), 1))
    divergence_plot_space = dolfinx.fem.FunctionSpace(problem_parametric._mesh, ufl.FiniteElement("CG", problem_parametric._mesh.ufl_cell(), 1))

    # Plot divergence of the original one
    fem_u_div_expr = dolfinx.fem.Expression(ufl.div(fem_u), divergence_space.element.interpolation_points())
    fem_u_div = dolfinx.fem.Function(divergence_plot_space)
    fem_u_div.interpolate(fem_u_div_expr)
    problem_parametric.save_results(p, solution_vel=fem_u_div, name_suffix="_div_fem")

    # Plot divergence of the reduced basis projection
    rb_u_div_expr = dolfinx.fem.Expression(ufl.div(reconstructed_u), divergence_space.element.interpolation_points())
    rb_u_div = dolfinx.fem.Function(divergence_plot_space)
    rb_u_div.interpolate(rb_u_div_expr)
    problem_parametric.save_results(p, solution_vel=rb_u_div, name_suffix="_div_rb")

    # Plot divergence of the NN solution
    pred_u_div_expr = dolfinx.fem.Expression(ufl.div(pred_u), divergence_space.element.interpolation_points())
    pred_u_div = dolfinx.fem.Function(divergence_plot_space)
    pred_u_div.interpolate(pred_u_div_expr)
    problem_parametric.save_results(p, solution_vel=pred_u_div, name_suffix="_div_pred")

    # Plot the difference in divergence
    div_diff_expr = dolfinx.fem.Expression(ufl.div(pred_u) - ufl.div(reconstructed_u), divergence_space.element.interpolation_points())
    div_diff = dolfinx.fem.Function(divergence_plot_space)
    div_diff.interpolate(div_diff_expr)
    problem_parametric.save_results(p, div_diff, name_suffix="_div_diff")

if __name__ == "__main__":
    timer = Timer()
    
    np.random.seed(0) # TODO temporary

    # Load mesh
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("mesh.msh", MPI.COMM_SELF, 0, gdim=2)
    mpi_print(f"Number of local cells: {mesh.topology.index_map(2).size_local}")
    mpi_print(f"Number of global cells: {mesh.topology.index_map(2).size_global}")
    timer.timestamp("Mesh loaded") # TODO is it a problem that I'm timing i/o operations? -> difference between process time and all time


    # Set up the problem
    problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags, facet_tags, HarmonicMeshMotion)
    
    reduced_problem_u = PODANNReducedProblem(problem_parametric, problem_parametric._V)
    reduced_problem_p = PODANNReducedProblem(problem_parametric, problem_parametric._Q)

    # POD
    print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))

    # TODO ask: why waste time broadcasting this set to every rank, if each can generate it itself in very few operations -> in case the sampling is not deterministic, but random
    global_training_set = generate_parameters_list(generate_parameters_values_list([7, 7, 7]))
    snapshots_u_matrix, snapshots_p_matrix = generate_solutions_list(global_training_set)
    mpi_print(f"Matrix built on rank {MPI.COMM_WORLD.Get_rank()} has {np.shape(snapshots_p_matrix)} snapshots.")
    timer.timestamp("POD training set calculated")

    print(rbnicsx.io.TextLine("perform POD", fill="#"))
    proper_orthogonal_decomposition(reduced_problem_u, snapshots_u_matrix)
    proper_orthogonal_decomposition(reduced_problem_p, snapshots_p_matrix)
    timer.timestamp("Reduced basis calculated")

    # NN
    # Generate training data
    paramteres_list = generate_parameters_list(generate_parameters_values_list([7, 7, 7]))
    np.random.shuffle(paramteres_list)
    solutions_list_u, solutions_list_p = generate_rb_solutions_list(paramteres_list)
    training_dataset_u, training_dataset_p, validation_dataset_u, validation_dataset_p =\
        prepare_test_and_training_sets(paramteres_list, solutions_list_u, solutions_list_p)
    timer.timestamp("NN dataset calculated")

    if MPI.COMM_WORLD.Get_rank() == 0:

        model = train_NN(training_dataset_u, validation_dataset_u)
        timer.timestamp("NN trained")

        timer.timestamp("Error analysis completed")

        p = Parameters(1, 2)
        save_preview(p, model, problem_parametric, reduced_problem_u, reduced_problem_p)
    
    save_all_timestamps(timer, 0)