from lid_driven_cavity_flow_mesh import Parameters, fluid_marker, lid_marker, wall_marker
from lid_driven_cavity_flow import ProblemOnDeformedDomain, PODANNReducedProblem

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion
from dlrbnicsx.dataset.custom_partitioned_dataset import CustomPartitionedDataset
from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.train_validate_test.train_validate_test_distributed import train_nn, validate_nn, online_nn
from dlrbnicsx.interface.wrappers import model_synchronise, init_cpu_process_group

import dolfinx
import rbnicsx.io
from petsc4py import PETSc
import ufl

from scipy.stats import qmc

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from mpi4py import MPI

import copy
import pickle
from itertools import product
from pathlib import Path
from time import process_time_ns, perf_counter_ns

import numpy as np


class Timer:
    def __init__(self):
        self.__timer = perf_counter_ns
        self.__start_time = self.__timer()
        self.__timestamps = {}

    
    def timestamp(self, label):
        assert(label not in self.__timestamps)
        timestamp = self.__timer() - self.__start_time
        self.__timestamps[label] = timestamp


    def get_timestamps(self):
        return self.__timestamps
    

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


def generate_parameters_values_list_random(ranges, n):
    l_bounds = [r[0] for r in ranges]
    u_bounds = [r[1] for r in ranges]
    
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n)
    scaled_samples = qmc.scale(sample, l_bounds, u_bounds)
    
    return scaled_samples


def generate_parameters_list(ranges, number_of_samples):
    samples = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        samples = generate_parameters_values_list_random(ranges, number_of_samples)
        with open('results/samples.npy', 'wb') as f:
            np.save(f, np.array(ranges))
            np.save(f, np.array(samples))
    samples = MPI.COMM_WORLD.bcast(samples, root=0)

    return np.array([Parameters(*sample) for sample in samples])


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


def temp_generate_solutions_list(global_training_set):
    # TODO discuss: what are the benefits or choosing every 10th vs continous sections of 10. Any caching benefit? Any way to use the similarity between solving similar solutions? Maybe iterative methods could use the solution to neighbouring parameter as the starting point?
    # TODO not necessarily helpful atm, as LU is used (am I right?) But maybe could switch to a different solver, slower for one solve, but faster for a range of them
    solutions_u = []
    solutions_p = []
    
    my_training_set = np.array_split(global_training_set, MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]
    
    for (params_index, params) in enumerate(my_training_set):
        print(rbnicsx.io.TextLine(f"{params_index+1} of {my_training_set.shape[0]}", fill="#"))
        print("High fidelity solve for params =", params)
        snapshot_u, snapshot_p = problem_parametric.solve(params)

        solutions_u.append(snapshot_u)
        solutions_p.append(snapshot_p)

    mpi_print(f"Rank {MPI.COMM_WORLD.Get_rank()} has {np.shape(solutions_p)} snapshots.")

    coefficients_u = []
    coefficients_p = []
    for sol_u, sol_p in zip(solutions_u, solutions_p):
        # TODO: could be removing copied functions from the original functions_list, but FunctionsList class doesn't have any method for removing other than erasing all
        coefficients_u.append(sol_u.vector[:])
        coefficients_p.append(sol_p.vector[:])
    
    coefficients_u = MPI.COMM_WORLD.allgather(coefficients_u)
    coefficients_p = MPI.COMM_WORLD.allgather(coefficients_p)

    coefficients_u = np.concatenate(coefficients_u, axis=0)
    coefficients_p = np.concatenate(coefficients_p, axis=0)
    
    solutions_u, solutions_p = [], []
    for (coefs_u, coefs_p) in zip(coefficients_u, coefficients_p):
        sol_u, sol_p = dolfinx.fem.Function(problem_parametric._V), dolfinx.fem.Function(problem_parametric._Q)
        sol_u.vector.setArray(coefs_u)
        sol_p.vector.setArray(coefs_p)
        solutions_u.append(sol_u)
        solutions_p.append(sol_p)

    return solutions_u, solutions_p


def temp_generate_functions_list(solutions, function_space):
    solutions_matrix = rbnicsx.backends.FunctionsList(function_space)
    for solution in solutions:
        solutions_matrix.append(solution)

    return solutions_matrix


def temp_generate_rb_solutions_list(solutions, reduced_problem):
    # TODO for larger problems it may make sense to parallelize this projection. Here communication cost almost certainly higher than the cost of all projections combined
    rb_snapshots_matrix = np.empty((len(solutions), reduced_problem.rb_dimension()))

    for i, solution in enumerate(solutions):
        rb_snapshots_matrix[i, :] = reduced_problem.project_snapshot(solution)

    return rb_snapshots_matrix


def shuffle(paramteres_list, rb_solutions_list_u, rb_solutions_list_p):
    if MPI.COMM_WORLD.Get_rank() == 0:
        permutation = list(range(len(paramteres_list)))
        np.random.shuffle(permutation)
    else:
        permutation = None
    permutation = MPI.COMM_WORLD.bcast(permutation, root=0)
    
    paramteres_list = [paramteres_list[i] for i in permutation]
    rb_solutions_list_u = [rb_solutions_list_u[i] for i in permutation]
    rb_solutions_list_p = [rb_solutions_list_p[i] for i in permutation]

    return paramteres_list, rb_solutions_list_u, rb_solutions_list_p


def proper_orthogonal_decomposition(reduced_problem, snapshots_matrix, Nmax):
    # NOTE POD parallelization: correlation matrix assembly could be paralalleized
    # NOTE SLEPc eigensolve underlying POD could also be parallalized

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

    return eigenvalues, modes


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

    # TODO possible to not gather all solutions, as each NN training processes only uses its subset of training data
    rb_snapshots_u_matrix = MPI.COMM_WORLD.allgather(rb_snapshots_u_matrix)
    rb_snapshots_p_matrix = MPI.COMM_WORLD.allgather(rb_snapshots_p_matrix)
    
    rb_snapshots_u_matrix = np.concatenate(rb_snapshots_u_matrix, axis=0)
    rb_snapshots_p_matrix = np.concatenate(rb_snapshots_p_matrix, axis=0)

    return rb_snapshots_u_matrix, rb_snapshots_p_matrix


def prepare_test_and_training_sets(parameters_list, solutions, num_training_samples, reduced_problem):
    paramteres_values_list = np.array([p.to_numpy() for p in parameters_list])

    reduced_problem.set_scaling_range(paramteres_values_list, solutions)

    # TODO disable transforms in CustomDataset sourcecode and compare results
    # Training set
    input_training_set = paramteres_values_list[:num_training_samples, :]
    solutions_training_set = np.array(solutions)[:num_training_samples, :]
    my_training_indices = np.array_split(range(len(input_training_set)), MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]
    
    # TODO seems wasteful for each instance of CustomPartitionedDataset to be keeping entire dataset and only using a subset of it
    training_dataset = CustomPartitionedDataset(reduced_problem,
                                     input_training_set,
                                     solutions_training_set,
                                     my_training_indices,
                                     verbose=False)

    # Validation set
    input_validation_set = paramteres_values_list[num_training_samples:, :]
    solutions_validation_set = np.array(solutions)[num_training_samples:, :]
    my_validation_indices = np.array_split(range(len(input_validation_set)), MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]
    validation_dataset = CustomPartitionedDataset(reduced_problem,
                                       input_validation_set,
                                       solutions_validation_set,
                                       my_validation_indices,
                                       verbose=False)

    return training_dataset, validation_dataset


def train_NN(training_dataset, validation_dataset, error_analyser, test_parameters_list, solutions, reduced_problem, name):
    # u only to begin with
    
    device = "cpu"
    batch_size = 64

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    assert(training_dataset[0])
    input_size = len(training_dataset[0][0])
    output_size = len(training_dataset[0][1])
    print(f"NN {input_size=}, {output_size=}")

    model = HiddenLayersNet(input_size, [30, 30], output_size, Tanh()).to(device)
    # model.double() # TODO remove? Convert the entire model to Double (or would have to convert input and outputs to floats (they're now doubles)) -> DLRBniCSx conerts everything to floats internally
    model_synchronise(model, verbose=False)

    # TODO investigate the loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) # torch.optim.SGD(model.parameters(), lr=1)
    

    # TODO plot the evolution with number of epochs
    # (plot the loss function evolution, but also prediction made with NN trained up to given epoch) 
    epochs = 4000

    norm_error_deformed_evolution = []
    norm_error_evolution = []

    lowest_validation_score = np.inf
    best_parameters = copy.deepcopy(model.state_dict())
    current_increase_length = 0
    max_increase_length = 100

    for t in range(epochs):
        report = (t % 50 == 1)
        if report:
            print(f"Epoch {t+1}\n-------------------------------")
        
        # TODO make a pull request removing reduced_problem argument from these functions
        train_nn(None, train_dataloader, model, loss_fn, optimizer, report)
        validation_score = validate_nn(None, test_dataloader, model, loss_fn, report)

        if validation_score < lowest_validation_score:
            lowest_validation_score = validation_score
            best_parameters = copy.deepcopy(model.state_dict())
            current_increase_length = 0
        else:
            current_increase_length += 1
            if current_increase_length >= max_increase_length:
                print(f"Early stopping triggered. Restoring the best model parameters from epoch {t - current_increase_length + 1}")
                model.load_state_dict(best_parameters)
                break

        if report:
            norm_error_deformed, norm_error = error_analyser(test_parameters_list, model, solutions, reduced_problem)
            norm_error_deformed_evolution.append(norm_error_deformed)
            norm_error_evolution.append(norm_error)

    with open(f'results/error_evolution_{name}.npy', 'wb') as f:
        np.save(f, norm_error_deformed_evolution)
        np.save(f, norm_error_evolution)
    
    print("Model trained!")

    return model


def error_analysis_distr(global_test_parameters, model_u, model_p, global_u_solutions, global_p_solutions, reduced_problem_u, reduced_problem_p):
    my_indices = np.array_split(range(len(global_test_parameters)), MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]

    norm_error_u_sum, norm_error_deformed_u_sum = 0, 0
    norm_error_p_sum, norm_error_deformed_p_sum = 0, 0

    for p_index in my_indices:
        p = global_test_parameters[p_index]
        rb_pred_u = online_nn(reduced_problem_u, None, p.to_numpy(), model_u, reduced_problem_u.rb_dimension())
        rb_pred_p = online_nn(reduced_problem_p, None, p.to_numpy(), model_p, reduced_problem_p.rb_dimension())

        pred_u = reduced_problem_u.reconstruct_solution(rb_pred_u)
        pred_p = reduced_problem_p.reconstruct_solution(rb_pred_p)

        norm_error_deformed_u_sum += reduced_problem_u.norm_error_deformed_context(p, global_u_solutions[p_index], pred_u)
        norm_error_u_sum += reduced_problem_u.norm_error(global_u_solutions[p_index], pred_u)

        norm_error_deformed_p_sum += reduced_problem_p.norm_error_deformed_context(p, global_p_solutions[p_index], pred_p)
        norm_error_p_sum += reduced_problem_p.norm_error(global_p_solutions[p_index], pred_p)
    
    norm_error_deformed_u_sum = MPI.COMM_WORLD.allreduce(norm_error_u_sum, MPI.SUM)
    norm_error_u_sum = MPI.COMM_WORLD.allreduce(norm_error_u_sum, MPI.SUM)

    norm_error_deformed_p_sum = MPI.COMM_WORLD.allreduce(norm_error_p_sum, MPI.SUM)
    norm_error_p_sum = MPI.COMM_WORLD.allreduce(norm_error_p_sum, MPI.SUM)
    
    return norm_error_deformed_u_sum / len(global_test_parameters), norm_error_u_sum / len(global_test_parameters), norm_error_deformed_p_sum / len(global_test_parameters), norm_error_p_sum / len(global_test_parameters)


def error_analysis_distr_u(global_test_parameters, model_u, global_u_solutions, reduced_problem_u):
    my_indices = np.array_split(range(len(global_test_parameters)), MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]

    norm_error_u_sum, norm_error_deformed_u_sum = 0, 0

    for p_index in my_indices:
        p = global_test_parameters[p_index]
        rb_pred_u = online_nn(reduced_problem_u, None, p.to_numpy(), model_u, reduced_problem_u.rb_dimension())

        pred_u = reduced_problem_u.reconstruct_solution(rb_pred_u)

        norm_error_deformed_u_sum += reduced_problem_u.norm_error_deformed_context(p, global_u_solutions[p_index], pred_u)
        norm_error_u_sum += reduced_problem_u.norm_error(global_u_solutions[p_index], pred_u)

    norm_error_deformed_u_sum = MPI.COMM_WORLD.allreduce(norm_error_u_sum, MPI.SUM)
    norm_error_u_sum = MPI.COMM_WORLD.allreduce(norm_error_u_sum, MPI.SUM)
    
    return norm_error_deformed_u_sum / len(global_test_parameters), norm_error_u_sum / len(global_test_parameters)


def error_analysis_distr_p(global_test_parameters, model_p, global_p_solutions, reduced_problem_p):
    my_indices = np.array_split(range(len(global_test_parameters)), MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]

    norm_error_p_sum, norm_error_deformed_p_sum = 0, 0

    for p_index in my_indices:
        p = global_test_parameters[p_index]
        rb_pred_p = online_nn(reduced_problem_p, None, p.to_numpy(), model_p, reduced_problem_p.rb_dimension())

        pred_p = reduced_problem_p.reconstruct_solution(rb_pred_p)

        norm_error_deformed_p_sum += reduced_problem_p.norm_error_deformed_context(p, global_p_solutions[p_index], pred_p)
        norm_error_p_sum += reduced_problem_p.norm_error(global_p_solutions[p_index], pred_p)

    norm_error_deformed_p_sum = MPI.COMM_WORLD.allreduce(norm_error_p_sum, MPI.SUM)
    norm_error_p_sum = MPI.COMM_WORLD.allreduce(norm_error_p_sum, MPI.SUM)
    
    return norm_error_deformed_p_sum / len(global_test_parameters), norm_error_p_sum / len(global_test_parameters)


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


def save_preview(preview_parameter, model_u, model_p, problem_parametric, reduced_problem_u, reduced_problem_p, path="results/"):
    # Infer solution
    # X = torch.tensor(preview_parameter.to_numpy()).to(torch.float32)
    # rb_pred = model(X)
    # rb_pred_vec = PETSc.Vec().createWithArray(rb_pred.detach().numpy(), comm=MPI.COMM_SELF)
    rb_pred_u = online_nn(reduced_problem_u, None, preview_parameter.to_numpy(), model_u, reduced_problem_u.rb_dimension())
    rb_pred_p = online_nn(reduced_problem_p, None, preview_parameter.to_numpy(), model_p, reduced_problem_p.rb_dimension())

    # Full solution
    fem_u, fem_p = problem_parametric.solve(preview_parameter)
    problem_parametric.save_results(preview_parameter, problem_parametric.interpolated_velocity(fem_u), fem_p, name_suffix="_fem", path=path)

    # Reduced basis projection of full solution
    rb_snapshot_u = reduced_problem_u.project_snapshot(fem_u)
    rb_snapshot_p = reduced_problem_p.project_snapshot(fem_p)
    reconstructed_u = reduced_problem_u.reconstruct_solution(rb_snapshot_u)
    reconstructed_p = reduced_problem_p.reconstruct_solution(rb_snapshot_p)
    problem_parametric.save_results(preview_parameter, problem_parametric.interpolated_velocity(reconstructed_u), reconstructed_p, name_suffix="_rb", path=path)
    # problem_parametric.save_results(p, problem_parametric.interpolated_velocity(fem_u), fem_p, name_suffix="_rb")

    # NN solution
    pred_u = reduced_problem_u.reconstruct_solution(rb_pred_u) 
    pred_p = reduced_problem_p.reconstruct_solution(rb_pred_p) 
    interpolated_pred = problem_parametric.interpolated_velocity(pred_u)
    problem_parametric.save_results(p, interpolated_pred, pred_p, name_suffix="_pred", path=path)

    # Plot difference
    u_diff = problem_parametric.interpolated_velocity(pred_u - reconstructed_u)
    p_diff = dolfinx.fem.Function(pred_p.function_space)
    p_diff.vector.setArray(pred_p.vector.getArray() - reconstructed_p.vector.getArray())
    problem_parametric.save_results(p, u_diff, p_diff, name_suffix="_diff", path=path)

    # Divergence
    divergence_space = dolfinx.fem.FunctionSpace(problem_parametric._mesh, ufl.FiniteElement("DG", problem_parametric._mesh.ufl_cell(), 1))
    divergence_plot_space = dolfinx.fem.FunctionSpace(problem_parametric._mesh, ufl.FiniteElement("CG", problem_parametric._mesh.ufl_cell(), 1))

    # Plot divergence of the original one
    fem_u_div_expr = dolfinx.fem.Expression(ufl.div(fem_u), divergence_space.element.interpolation_points())
    fem_u_div = dolfinx.fem.Function(divergence_plot_space)
    fem_u_div.interpolate(fem_u_div_expr)
    problem_parametric.save_results(p, solution_vel=fem_u_div, name_suffix="_div_fem", path=path)

    # Plot divergence of the reduced basis projection
    rb_u_div_expr = dolfinx.fem.Expression(ufl.div(reconstructed_u), divergence_space.element.interpolation_points())
    rb_u_div = dolfinx.fem.Function(divergence_plot_space)
    rb_u_div.interpolate(rb_u_div_expr)
    problem_parametric.save_results(p, solution_vel=rb_u_div, name_suffix="_div_rb", path=path)

    # Plot divergence of the NN solution
    pred_u_div_expr = dolfinx.fem.Expression(ufl.div(pred_u), divergence_space.element.interpolation_points())
    pred_u_div = dolfinx.fem.Function(divergence_plot_space)
    pred_u_div.interpolate(pred_u_div_expr)
    problem_parametric.save_results(p, solution_vel=pred_u_div, name_suffix="_div_pred", path=path)

    # Plot the difference in divergence
    div_diff_expr = dolfinx.fem.Expression(ufl.div(pred_u) - ufl.div(reconstructed_u), divergence_space.element.interpolation_points())
    div_diff = dolfinx.fem.Function(divergence_plot_space)
    div_diff.interpolate(div_diff_expr)
    problem_parametric.save_results(p, div_diff, name_suffix="_div_diff", path=path)

    # Plot the first two most significant RB mode projected into the full basis
    problem_parametric.save_results(Parameters(), problem_parametric.interpolated_velocity(reduced_problem_u._basis_functions[0]), reduced_problem_p._basis_functions[0], name_suffix="_rb_1st_mode", path=path)
    problem_parametric.save_results(Parameters(), problem_parametric.interpolated_velocity(reduced_problem_u._basis_functions[1]), reduced_problem_p._basis_functions[1], name_suffix="_rb_2nd_mode", path=path)


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

    range_0 = (0.5, 2.5)
    range_1 = (0.5, 2.5)
    range_2 = (np.pi/10, 9*np.pi/10)
    ranges = [range_0, range_1, range_2]
    number_of_samples = 300
    Nmax = 100 # Max number of POD basis functions
    reuse_samples = True
    

    global_training_set = generate_parameters_list(ranges, number_of_samples)

    # snapshots_u_matrix, snapshots_p_matrix = generate_solutions_list(global_training_set)
    snapshots_u_list, snapshots_p_list = temp_generate_solutions_list(global_training_set)
    snapshots_u_matrix, snapshots_p_matrix = temp_generate_functions_list(snapshots_u_list, problem_parametric._V), temp_generate_functions_list(snapshots_p_list, problem_parametric._Q)
    mpi_print(f"Matrix built on rank {MPI.COMM_WORLD.Get_rank()} has {np.shape(snapshots_p_matrix)} snapshots.")
    timer.timestamp("POD training set calculated")

    print(rbnicsx.io.TextLine("perform POD", fill="#"))
    e_u, m_u = proper_orthogonal_decomposition(reduced_problem_u, snapshots_u_matrix, Nmax)
    e_p, m_p = proper_orthogonal_decomposition(reduced_problem_p, snapshots_p_matrix, Nmax)
    if MPI.COMM_WORLD.rank == 0:
        with open('results/eigen.npy', 'wb') as f:
            np.save(f, [e_u, e_p])
            np.save(f, [len(m_u), len(m_p)])
    timer.timestamp("Reduced basis calculated")

    # TODO bug
    # lorem, ipsum = temp_generate_solutions_list(global_training_set) # doesnt work, crashes on the error estimation set generatino
    # temp_generate_solutions_list(global_training_set) # works
    
    # NN
    # Generate training data
    if reuse_samples == False: # works for N=150, 200 samples but not for N=250, 300 samples
        paramteres_list = generate_parameters_list(ranges, number_of_samples)
        snapshots_u_list, snapshots_p_list = temp_generate_solutions_list(paramteres_list)
    else: # works
        paramteres_list = global_training_set # legacy naming, should be the same variable


    rb_solutions_list_u = temp_generate_rb_solutions_list(snapshots_u_list, reduced_problem_u)
    rb_solutions_list_p = temp_generate_rb_solutions_list(snapshots_p_list, reduced_problem_p)

    paramteres_list, rb_solutions_list_u, rb_solutions_list_p = shuffle(paramteres_list, rb_solutions_list_u, rb_solutions_list_p) 

    init_cpu_process_group(MPI.COMM_WORLD)

    num_training_samples = int(0.7 * len(paramteres_list))
    training_dataset_u, validation_dataset_u =\
        prepare_test_and_training_sets(paramteres_list, rb_solutions_list_u, num_training_samples, reduced_problem_u)
    training_dataset_p, validation_dataset_p =\
        prepare_test_and_training_sets(paramteres_list, rb_solutions_list_p, num_training_samples, reduced_problem_p)
    # timer.timestamp("NN dataset calculated")

    test_parameters_list = generate_parameters_list(ranges, 27)
    solutions_u, solutions_p = temp_generate_solutions_list(test_parameters_list)

    model_u = train_NN(training_dataset_u, validation_dataset_u, error_analysis_distr_u, test_parameters_list, solutions_u, reduced_problem_u, "velocity")
    model_p = train_NN(training_dataset_p, validation_dataset_p, error_analysis_distr_p, test_parameters_list, solutions_p, reduced_problem_p, "pressure")
    timer.timestamp("NN trained")

    norm_error_deformed_u, norm_error_u = error_analysis_distr_u(test_parameters_list, model_u, solutions_u, reduced_problem_u)
    norm_error_deformed_p, norm_error_p = error_analysis_distr_p(test_parameters_list, model_p, solutions_p, reduced_problem_p)
    mpi_print(f"N = {number_of_samples}, error_u= {norm_error_deformed_u}, error_p= {norm_error_deformed_p}")
    mpi_print(f"Velocity RB dimension = {reduced_problem_u.rb_dimension()} Pressure RB dimension = {reduced_problem_p.rb_dimension()}")

    timer.timestamp("Error analysis completed")

    if MPI.COMM_WORLD.Get_rank() == 0:
        # p = Parameters(1, 2, np.pi/6)
        p = Parameters(1, 1, np.pi/2)
        save_preview(p, model_u, model_p, problem_parametric, reduced_problem_u, reduced_problem_p, path="results/ref/")
        
        p = Parameters(0.75, 2, 2*np.pi/3)
        save_preview(p, model_u, model_p, problem_parametric, reduced_problem_u, reduced_problem_p, path="results/left/")

        p = Parameters(2, 0.75, np.pi/6)
        save_preview(p, model_u, model_p, problem_parametric, reduced_problem_u, reduced_problem_p, path="results/right/")

        test_parameters_list = generate_parameters_list(ranges, 100)
        FEM_times = []
        POD_ANN_times = []

        for i, params in enumerate(test_parameters_list):
            print(i)
            start_time = process_time_ns()
            problem_parametric.solve(params)
            end_time = process_time_ns()
            FEM_times.append(end_time-start_time)

            start_time = process_time_ns()
            online_nn(reduced_problem_u, None, params.to_numpy(), model_u, reduced_problem_u.rb_dimension())
            online_nn(reduced_problem_p, None, params.to_numpy(), model_p, reduced_problem_p.rb_dimension())
            end_time = process_time_ns()
            POD_ANN_times.append(end_time-start_time)

        with open('results/online_time.npy', 'wb') as f:
            np.save(f, FEM_times)
            np.save(f, POD_ANN_times)
    
    save_all_timestamps(timer, 0)