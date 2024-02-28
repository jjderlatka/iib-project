from lid_driven_cavity_flow_mesh import Parameters, fluid_marker, lid_marker, wall_marker
from lid_driven_cavity_flow import ProblemOnDeformedDomain, PODANNReducedProblem

from mdfenicsx.mesh_motion_classes import HarmonicMeshMotion

import dolfinx
import rbnicsx.io
from petsc4py import PETSc

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from mpi4py import MPI

from itertools import product
from pathlib import Path

import numpy as np

comm = MPI.COMM_WORLD
gdim = 2 # dimension of the model
gmsh_model_rank = 0

mesh, cell_tags, facet_tags = \
    dolfinx.io.gmshio.read_from_msh("mesh.msh", comm,
                                    gmsh_model_rank, gdim=gdim)

problem_parametric = ProblemOnDeformedDomain(mesh, cell_tags, facet_tags,
                                             HarmonicMeshMotion)

# 1. Generate Reduced basis with POD
# 2. Generate training data (parameters -> Reduced basis solution)
# 3. Train the NN on the data


# POD

def generate_parameters_values_list(samples=[5, 5, 5]):
    training_set_0 = np.linspace(0.5, 2.5, samples[0])
    training_set_1 = np.linspace(0.5, 2.5, samples[1])
    training_set_2 = np.linspace(np.pi/2, np.pi/10, samples[2])
    training_set = np.array(list(product(training_set_0,
                                        training_set_1,
                                        training_set_2)))
    return training_set


def generate_parameters_list(parameteres_values):
    return np.array([Parameters(*vals) for vals in parameteres_values])


training_set = rbnicsx.io.on_rank_zero(mesh.comm, lambda x=[5, 5, 5]: generate_parameters_list(generate_parameters_values_list(x)))

Nmax = 100 # the number of basis functions

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
snapshots_u_matrix = rbnicsx.backends.FunctionsList(problem_parametric._V)
snapshots_p_matrix = rbnicsx.backends.FunctionsList(problem_parametric._Q)

print("")

for (params_index, params) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(params_index+1), fill="#"))

    print("Parameter number ", (params_index+1), "of", training_set.shape[0])
    print("high fidelity solve for params =", params)
    snapshot_u, snapshot_p = problem_parametric.solve(params)

    print("update snapshots matrix")
    snapshots_u_matrix.append(snapshot_u)
    snapshots_p_matrix.append(snapshot_p)

    print("")

print("set up reduced problem")
reduced_problem_u = PODANNReducedProblem(problem_parametric, problem_parametric._V)
reduced_problem_p = PODANNReducedProblem(problem_parametric, problem_parametric._Q)

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

reduced_problem_u.set_reduced_basis(modes_u)
reduced_problem_p.set_reduced_basis(modes_p)
print("")

# Generate training data
class CustomDataset(Dataset):
    def __init__(self, parameters_list, solutions_list):
        assert(len(parameters_list) == len(solutions_list))
        self.parameters = parameters_list
        self.solutions = solutions_list


    def __len__(self):
        return len(self.parameters)


    def __getitem__(self, idx):
        return self.parameters[idx], self.solutions[idx]


def generate_rb_solutions_list(parameters_set):
    rb_snapshots_u_matrix = np.empty((len(parameters_set), reduced_problem_u.rb_dimension()))
    rb_snapshots_p_matrix = np.empty((len(parameters_set), reduced_problem_p.rb_dimension()))

    for (params_index, params) in enumerate(parameters_set):
        print(rbnicsx.io.TextLine(str(params_index+1), fill="#"))

        print("Parameter number ", (params_index+1), "of", len(parameters_set))
        print("high fidelity solve for params =", params)
        snapshot_u, snapshot_p = problem_parametric.solve(params)

        rb_snapshot_u = reduced_problem_u.project_snapshot(snapshot_u)
        rb_snapshot_p = reduced_problem_p.project_snapshot(snapshot_p)

        rb_snapshots_u_matrix[params_index, :] = rb_snapshot_u
        rb_snapshots_p_matrix[params_index, :] = rb_snapshot_p

    print("")
    return rb_snapshots_u_matrix, rb_snapshots_p_matrix


def prepare_test_and_training_sets():
    paramteres_list_ = generate_parameters_list(generate_parameters_values_list([7, 7, 7]))
    np.random.shuffle(paramteres_list_)
    solutions_list_u, solutions_list_p = generate_rb_solutions_list(paramteres_list_)
    print(f"{np.shape(solutions_list_u)=}")

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


training_dataset_u, training_dataset_p, validation_dataset_u, validation_dataset_p = prepare_test_and_training_sets()

device = "cpu"

# u only to begin with
batch_size = 64 # TODO increase
train_dataloader = DataLoader(training_dataset_u, batch_size=batch_size)
test_dataloader = DataLoader(validation_dataset_u, batch_size=batch_size)

input_size = len(Parameters())
output_size = reduced_problem_u.rb_dimension()
print(f"NN {input_size=}, {output_size=}")

# ToDo parametrize the architecture as a list of layer sizes, so its easier to optimize
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )


    def forward(self, params):
        logits = self.linear_relu_stack(params)
        return logits


model = NeuralNetwork().to(device)
# TODO remove 
model.double() # Convert the entire model to Double (or would have to convert input and outputs to floats (they're now doubles))
print(model)

# TODO investigate the loss function
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

## NN training
# TODO plot the evolution with number of epochs
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# Infer solution and save it
p = Parameters(1.21, 1.37, np.pi/4*1.01)
X = torch.tensor(p.to_numpy())
rb_pred = model(X)
rb_pred_vec = PETSc.Vec().createWithArray(rb_pred.detach().numpy())

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
    with dolfinx.io.XDMFFile(comm, filename_velocity.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(interpolated_pred)

print("Done")