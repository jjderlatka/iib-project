from lid_driven_cavity_flow_mesh import Parameters, fluid_marker, lid_marker, wall_marker
from lid_driven_cavity_flow import ProblemOnDeformedDomain, PODANNReducedProblem

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion
from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh
from dlrbnicsx.train_validate_test.train_validate_test import train_nn, validate_nn

import dolfinx
import rbnicsx.io
from petsc4py import PETSc

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


# TODO implementing __setstate__ and __getstate__ would data to be pickled and therefore bcasted and gathered easily
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


def generate_rb_solutions_list(parameters_set):
    my_parameters_set = np.array_split(parameters_set, MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]

    rb_snapshots_u_matrix = np.empty((len(my_parameters_set), reduced_problem_u.rb_dimension()))
    rb_snapshots_p_matrix = np.empty((len(my_parameters_set), reduced_problem_p.rb_dimension()))

    for (params_index, params) in enumerate(my_parameters_set):
        print(rbnicsx.io.TextLine(f"{params_index+1} of {len(my_parameters_set)}", fill="#"))
        print("high fidelity solve for params =", params)
        snapshot_u, snapshot_p = problem_parametric.solve(params)

        rb_snapshot_u = reduced_problem_u.project_snapshot(snapshot_u)
        rb_snapshot_p = reduced_problem_p.project_snapshot(snapshot_p)

        rb_snapshots_u_matrix[params_index, :] = rb_snapshot_u
        rb_snapshots_p_matrix[params_index, :] = rb_snapshot_p

    print("")

    rb_snapshots_u_matrix = MPI.COMM_WORLD.gather(rb_snapshots_u_matrix, 0)
    rb_snapshots_p_matrix = MPI.COMM_WORLD.gather(rb_snapshots_p_matrix, 0)
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        rb_snapshots_u_matrix = np.concatenate(rb_snapshots_u_matrix, axis=0)
        rb_snapshots_p_matrix = np.concatenate(rb_snapshots_p_matrix, axis=0)

    rb_snapshots_u_matrix = MPI.COMM_WORLD.bcast(rb_snapshots_u_matrix, 0)
    rb_snapshots_p_matrix = MPI.COMM_WORLD.bcast(rb_snapshots_p_matrix, 0)

    return rb_snapshots_u_matrix, rb_snapshots_p_matrix


def prepare_test_and_training_sets(samples):
    paramteres_list_ = generate_parameters_list(generate_parameters_values_list(samples))
    
    np.random.shuffle(paramteres_list_)
    solutions_list_u, solutions_list_p = generate_rb_solutions_list(paramteres_list_)

    num_training_samples = int(0.7 * len(paramteres_list_))
    num_validation_samples = len(paramteres_list_) - num_training_samples

    paramteres_list = np.array([p.to_numpy() for p in paramteres_list_])

    input_training_set = paramteres_list[:num_training_samples, :]
    solutions_u_training_set = solutions_list_u[:num_training_samples, :]
    solutions_p_training_set = solutions_list_p[:num_training_samples, :]
    training_dataset_u = CustomDataset(input_training_set, solutions_u_training_set)
    training_dataset_p = CustomDataset(input_training_set, solutions_p_training_set)

    input_validation_set = paramteres_list[num_training_samples:, :]
    solutions_u_validation_set = solutions_list_u[num_training_samples:, :]
    solutions_p_validation_set = solutions_list_p[num_training_samples:, :]
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


if __name__ == "__main__":
    timer = Timer()
    
    # TODO probably remove
    np.random.seed(0)

    comm = MPI.COMM_WORLD
    gdim = 2 # dimension of the model

    mesh, cell_tags, facet_tags = \
        dolfinx.io.gmshio.read_from_msh("mesh.msh", MPI.COMM_SELF, 0, gdim=gdim)
    
    timer.timestamp("Mesh loaded")

    mpi_print(f"Number of local cells: {mesh.topology.index_map(2).size_local}")
    mpi_print(f"Number of global cells: {mesh.topology.index_map(2).size_global}")

    problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags, facet_tags,
                                                HarmonicMeshMotion)

    # POD

    # TODO ask: why waste time broadcasting this set to every rank, if each can generate it itself in very few operations
    global_training_set = generate_parameters_list(generate_parameters_values_list([6, 6, 6]))

    Nmax = 100 # the number of basis functions

    print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))

    # TODO All below could be enclosed in a loop
    print("set up snapshots matrix")
    snapshots_u_matrix = rbnicsx.backends.FunctionsList(problem_parametric._V)
    snapshots_p_matrix = rbnicsx.backends.FunctionsList(problem_parametric._Q)

    # TODO discuss: what are the benefits or choosing every 10th vs continous sections of 10. Any caching benefit? Any way to use the similarity between solving similar solutions? Maybe iterative methods could use the solution to neighbouring parameter as the starting point?
    # TODO not necessarily helpful atm, as LU is used (am I right?) But maybe could switch to a different solver, slower for one solve, but faster for a range of them

    # TODO ask: why do dolfinx and rbnicsx have the two level interface? Each function calls its backend version
    my_training_set = np.array_split(global_training_set, MPI.COMM_WORLD.Get_size())[MPI.COMM_WORLD.Get_rank()]
    # mpi_print(my_training_set)
    # TODO priority first: make a function for solving a for many parameters, with the distributed option, in the original class
    
    for (params_index, params) in enumerate(my_training_set):
        print(rbnicsx.io.TextLine(f"{params_index+1} of {my_training_set.shape[0]}", fill="#"))
        print("high fidelity solve for params =", params)
        snapshot_u, snapshot_p = problem_parametric.solve(params)

        snapshots_u_matrix.append(snapshot_u)
        snapshots_p_matrix.append(snapshot_p)

    mpi_print(f"Rank {MPI.COMM_WORLD.Get_rank()} has {np.shape(snapshots_p_matrix)} snapshots.")

    snapshots_u_matrix = gather_functions_list(snapshots_u_matrix, 0)
    snapshots_p_matrix = gather_functions_list(snapshots_p_matrix, 0)

    timer.timestamp("POD training set calculated")

    mpi_print(f"Matrix built on rank {MPI.COMM_WORLD.Get_rank()} has {np.shape(snapshots_u_matrix)} snapshots.")
    mpi_print(f"Matrix built on rank {MPI.COMM_WORLD.Get_rank()} has {np.shape(snapshots_p_matrix)} snapshots.")

    print("set up reduced problem")
    reduced_problem_u = PODANNReducedProblem(problem_parametric, problem_parametric._V)
    reduced_problem_p = PODANNReducedProblem(problem_parametric, problem_parametric._Q)

    # NOTE POD parallelization
    # correlation matrix assembly could be paralalleized

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(rbnicsx.io.TextLine("perform POD", fill="#"))
        eigenvalues_u, modes_u, _ = \
            rbnicsx.backends.\
            proper_orthogonal_decomposition(snapshots_u_matrix,
                                            reduced_problem_u._inner_product_action,
                                            N=Nmax, tol=1.e-6)

        eigenvalues_p, modes_p, _ = \
            rbnicsx.backends.\
            proper_orthogonal_decomposition(snapshots_p_matrix,
                                            reduced_problem_p._inner_product_action,
                                            N=Nmax, tol=1.e-6)
        
    else:
        eigenvalues_u, modes_u = None, None
        eigenvalues_p, modes_p = None, None

    eigenvalues_u, modes_u = MPI.COMM_WORLD.bcast(eigenvalues_u, root=0), bcast_functions_list(modes_u, problem_parametric._V, 0)
    eigenvalues_p, modes_p = MPI.COMM_WORLD.bcast(eigenvalues_p, root=0), bcast_functions_list(modes_p, problem_parametric._Q, 0)

    reduced_problem_u.set_reduced_basis(modes_u)
    reduced_problem_p.set_reduced_basis(modes_p)

    timer.timestamp("Reduced basis calculated")

    # Generate training data
    training_dataset_u, training_dataset_p, validation_dataset_u, validation_dataset_p = prepare_test_and_training_sets([6, 6, 6])

    timer.timestamp("NN dataset calculated")

    if MPI.COMM_WORLD.Get_rank() == 0:

        device = "cpu"

        # u only to begin with
        batch_size = 64 # TODO increase
        train_dataloader = DataLoader(training_dataset_u, batch_size=batch_size)
        test_dataloader = DataLoader(validation_dataset_u, batch_size=batch_size)

        input_size = len(Parameters())
        output_size = reduced_problem_u.rb_dimension()
        print(f"NN {input_size=}, {output_size=}")

        model = HiddenLayersNet(input_size, [512, 512], output_size, Tanh()).to(device)
        # TODO remove 
        model.double() # Convert the entire model to Double (or would have to convert input and outputs to floats (they're now doubles))
        print(model)

        # TODO investigate the loss function
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        ## NN training
        # TODO plot the evolution with number of epochs
        epochs = 10
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            # TODO make a pull request removing reduced_problem argument from these functions
            train_nn(None, train_dataloader, model, loss_fn, optimizer)
            validate_nn(None, test_dataloader, model, loss_fn)
        print("Done!")

        timer.timestamp("NN trained")

        # Infer solution and save it
        p = Parameters(1.21, 1.37, np.pi/4*1.01)
        X = torch.tensor(p.to_numpy())
        rb_pred = model(X)
        # TODO: ask why it didn't work without specyfing the communicator? How did it know
        rb_pred_vec = PETSc.Vec().createWithArray(rb_pred.detach().numpy(), comm=MPI.COMM_SELF)

        # TODO make a pull request adding the option to supply solution to error analysis function
        with  HarmonicMeshMotion(mesh, 
                                    facet_tags, 
                                    [wall_marker, lid_marker], 
                                    [p.transform, p.transform], 
                                    reset_reference=True, 
                                    is_deformation=False) as mesh_class:
            print(f"Projecting the prediction to full basis")

            full_order_pred = reduced_problem_u.reconstruct_solution(rb_pred_vec)

            interpolated_pred = problem_parametric.interpolated_velocity(full_order_pred)

            print(f"Saving the result")
            results_folder = Path("results")
            results_folder.mkdir(exist_ok=True, parents=True)
            filename_velocity = results_folder / "lid_driven_cavity_flow_velocity"
            with dolfinx.io.XDMFFile(mesh.comm, filename_velocity.with_suffix(".xdmf"), "w") as xdmf:
                xdmf.write_mesh(mesh)
                xdmf.write_function(interpolated_pred)

        timer.timestamp("Error analysis completed")
    
    save_all_timestamps(timer, 0)