# Example usage
# python main.py --m 2 --n 5 --epochs 20000 --max_loss_plot 0.06
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg') # on MacOSX
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker
from copy import deepcopy
import argparse
import os
import re

from toymodels import ToyModelsDataset, ToyModelsNet
from slt import LearningMachine

# TODO
#   - Plot test loss

def parse_commandline():
    parser = argparse.ArgumentParser(description="SLT Toy Model")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=512)
    parser.add_argument("--m", help="Number of hidden dimensions", type=int, default=2)
    parser.add_argument("--n", help="Number of input dimensions", type=int, default=5)
    parser.add_argument("--epochs", help="epochs", type=int, default=6000)
    parser.add_argument("--sgld_chains", help="Number of SGLD chains to average over during posterior estimates", type=int, default=5)
    parser.add_argument("--init_polygon", help="Initial weight matrix", type=int, default=None)
    parser.add_argument("--init_noise_scale", help="Initial weight matrix noise scale", type=float, default=0.1)
    parser.add_argument("--lr", help="Initial learning rate", type=float, default=1e-3)
    parser.add_argument("--hatlambdas", help="Number of hatlambdas to compute", type=int, default=20)
    parser.add_argument("--gpu", help="Use GPU, off by default", action="store_true")
    parser.add_argument("--truth_gamma", help="Related to std for true distribution", type=int, default=10)
    parser.add_argument("--max_loss_plot", help="Maximum on y axis for loss plot", type=float, default=None)
    parser.add_argument("--save_plot", help="Save the resulting plot", action="store_true")
    parser.add_argument(
        "--outputdir",
        help="Path to output directory. Create if not exist.",
        type=str,
        default=None,
    )
    return parser

def main(args):
    # Hyperparameters
    m, n = args.m, args.n
    num_epochs = args.epochs
    init_polygon = args.init_polygon
    lr_init = args.lr
    truth_gamma = args.truth_gamma # 1/sqrt(truth_gamma) is the std of the true distribution q(y|x)

    num_covariance_checkpoints = 40
    steps_per_epoch = 128
    num_plots = 5
    first_snapshot_epoch = 200
    assert num_epochs > first_snapshot_epoch, "Number of epochs too small"
    plot_interval = (num_epochs - first_snapshot_epoch) // (num_plots - 1)
    covariance_interval = (num_epochs - first_snapshot_epoch) // (num_covariance_checkpoints - 1)
    smoothing_window = num_epochs // 100

    print(f"SLT Toy Model m={m},n={n}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Device: {device}")
    
    # The training set is allocated at the beginning using the custom dataset
    total_train = steps_per_epoch * num_epochs
    trainset = ToyModelsDataset(total_train, n, truth_gamma)
    #testset = ToyModelsDataset(total_train // 6, n, truth_gamma)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    trainloader_batched = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    lr = lr_init
    criterion = nn.MSELoss()

    # The main model for training
    model = ToyModelsNet(n, m, init_config=init_polygon, noise_scale=args.init_noise_scale, use_optimal=True)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # The learning machine, for computing free energy and lambda
    machine = LearningMachine(model, trainloader_batched, criterion, optimizer, device, truth_gamma, args.sgld_chains)
    
    W_history = []
    b_history = []
    loss_history = []
    snapshot_epoch = []
    stat_epochs = []
    lfe_history = []
    energy_history = []
    hatlambda_history = []
    dims_per_feature = []
    model_state_history = []
    testloss_history = []

    def dim_per_feature(W):
        out = W.size(0) / (torch.linalg.matrix_norm(W) ** 2)
        return out
    
    # Training loop
    dataiter = iter(trainloader)
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            data = next(dataiter)
            x = data[0].to(device)
            
            # Forward pass
            output = model(x)
            loss = criterion(output, x)

            # (see slt.py)
            # L'_k(w) = truth_gamma/2 1/k \sum_{i=1}^k || y - ReLU(W^TWx + b ) ||^2
            loss = loss * truth_gamma / 2

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Save W matrix at regular intervals
        if (epoch - first_snapshot_epoch + 1) % plot_interval == 0:
            snapshot_epoch.append(epoch+1)
            W_history.append(model.W.cpu().detach().clone())
            b_history.append(model.b.cpu().detach().clone())
        
        if (epoch - first_snapshot_epoch + 1) % covariance_interval == 0:
            model_state_history.append({
                "epoch": epoch,
                "W": model.W.cpu().detach().clone(),
                "b": model.b.cpu().detach().clone()
            })

        if epoch > first_snapshot_epoch and (epoch + 1) % (num_epochs // args.hatlambdas) == 0:
            stat_epochs.append(epoch+1)
            dims_per_feature.append(dim_per_feature(model.W).cpu().detach().clone())

            energy = machine.compute_energy()
            lfe = machine.compute_local_free_energy()
            hatlambda = (lfe - energy) / np.log(total_train)

            energy_history.append(energy.cpu().detach().clone())
            lfe_history.append(lfe.cpu().detach().clone())
            hatlambda_history.append(hatlambda.cpu().detach().clone())

        epoch_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.5f}')

        loss_history.append(epoch_loss)

    ####################
    # Compute covariance matrices
    #

    num_trajectories = 200
    num_covariant_chain_steps = 100
    covariance_epochs = []    
    covariance_maxeigenvalues = []
    covariance_maxeigenvectors = []

    def add_gaussian_noise_to_state(model, std=0.01):
        state_dict = model.state_dict()
        for _, param in state_dict.items():
            noise = torch.randn_like(param) * std
            param.add_(noise)  # This adds the noise in-place

        model.load_state_dict(state_dict)
        return model

    print("Computing covariance matrices")
    for _, saved_state in enumerate(model_state_history):
        covariance_epochs.append(saved_state["epoch"])
        X_list = []

        for i in range(num_trajectories):
            torch.manual_seed(i)
            torch.cuda.manual_seed_all(i)
            
            # Create a new model and optimizer instance
            new_model = ToyModelsNet(n, m, init_config=init_polygon, noise_scale=args.init_noise_scale, use_optimal=True)
            new_model.W.data = saved_state["W"]
            new_model.b.data = saved_state["b"]
            #new_model = add_gaussian_noise_to_state(new_model, std=0.01)
            new_model.to(device)
            new_optimizer = optim.SGD(new_model.parameters(), lr=lr)

            # Create an independent training set        
            new_trainset = ToyModelsDataset(num_covariant_chain_steps, n, truth_gamma)
            new_trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=1, shuffle=True)
            new_dataiter = iter(new_trainloader)

            for _ in range(num_covariant_chain_steps):
                data = next(new_dataiter)
                x = data[0].to(device)
                
                # Forward pass
                output = new_model(x)
                loss = criterion(output, x)
                loss = loss * truth_gamma / 2

                # Backward pass
                new_optimizer.zero_grad()
                loss.backward()
                new_optimizer.step()

            W_flat = torch.flatten(new_model.W)
            combined_vector = torch.cat((W_flat, new_model.b))
            X_list.append(combined_vector.cpu().detach().clone())

        X = torch.stack(X_list)
        mean_vector = torch.mean(X, dim=0)
        X_centered = X - mean_vector
        cov_matrix = X_centered.t().mm(X_centered) / (X.size(0) - 1)
        
        eigenresult = torch.linalg.eig(cov_matrix)
        eigenvalues = eigenresult.eigenvalues.real
        max_eigenvalue, idx = torch.max(eigenvalues, dim=0)
        max_eigenvector = eigenresult.eigenvectors[:, idx].real

        #print("Eigenvalues:", eigenvalues)  # The eigenvalues are in the first column of the returned tensor
        print("Max eigenvalue:", max_eigenvalue)
        covariance_maxeigenvalues.append(max_eigenvalue)
        covariance_maxeigenvectors.append(max_eigenvector)

    ####################
    # Plotting
    
    num_plots = len(W_history)

    plot_rows = 5
    plot_ratios = [2,1,1,2,2]

    plot_height = 55
    
    def find_closest_index(target, numbers):
        return min(enumerate(numbers), key=lambda x: abs(x[1] - target))[0]
    
    gs = gridspec.GridSpec(plot_rows, num_plots, height_ratios=plot_ratios)
    fig = plt.figure(figsize=(35, plot_height))
    fig.suptitle(f"Toy models (n={n}, m={m}, init_polygon={args.init_polygon or 'None'})", fontsize=10)
    if m == 2:
        axes1 = [fig.add_subplot(gs[0, i]) for i in range(num_plots)]
    elif m == 3:
        axes1 = [fig.add_subplot(gs[0,i], projection='3d') for i in range(num_plots)]

    axes2 = [fig.add_subplot(gs[1, i]) for i in range(num_plots)]

    axes3 = [fig.add_subplot(gs[2, i]) for i in range(num_plots)]

    for i, W in enumerate(W_history):
        # Plot the eigenvectors of the covariance matrix
        nearest_cov_epoch = find_closest_index(snapshot_epoch[i],covariance_epochs)
        eigenvector = covariance_maxeigenvectors[nearest_cov_epoch]
        V = eigenvector[:W.numel()].reshape(W.shape).cpu()
                
        for j in range(n):
            column_vector = W.cpu()[:, j].numpy()
            V_column_vector = V[:, j].numpy()

            # Plot the arrow
            if m == 2:
                # 2D
                axes1[i].quiver(0, 0, column_vector[0], column_vector[1], angles='xy', scale_units='xy', scale=1, label=f'Column {j+1}')
                axes1[i].quiver(column_vector[0], column_vector[1], V_column_vector[0], V_column_vector[1], angles='xy', scale_units='xy', scale=1, color='red')
            elif m == 3:
                # 3D
                axes1[i].quiver(0, 0, 0, column_vector[0], column_vector[1], column_vector[2], 
                        color=plt.cm.jet(j/n), linewidth=1.5, label=f'Column {j+1}')
                axes1[i].quiver(column_vector[0], column_vector[1], column_vector[2], V_column_vector[0], V_column_vector[1], V_column_vector[2], color='red')
        

        axes1[i].set_title(f'Epoch {snapshot_epoch[i]}')
        if m == 2:
            axes1[i].set_xlim(-1.5, 1.5)
            axes1[i].set_ylim(-1.5, 1.5)
        elif m == 3:
            axes1[i].set_xlim(-1, 1)
            axes1[i].set_ylim(-1, 1)
            axes1[i].set_zlim(-1, 1)
            axes1[i].view_init(elev=20, azim=45)
        axes1[i].set_aspect('equal')

        if i != 0:
            axes1[i].set_xticklabels([])
            axes1[i].set_yticklabels([])

            if m == 3:
                axes1[i].set_zticklabels([])

        column_norms = torch.norm(W.cpu(), dim=0).numpy()

        # Plot the distribution of column norms
        axes2[i].hist(column_norms, bins=np.linspace(0, 2, num=21), alpha=0.75, range=(0,2))
        if i == 0:
            axes2[i].set_ylabel('W')
            axes2[i].tick_params(axis='x', labelsize=8)
        else:
            axes2[i].set_xticklabels([])
            axes2[i].set_yticklabels([])
        
        axes2[i].set_xlim(0, 2)
        axes2[i].set_xticks(np.arange(0, 2.1, 0.5))
        axes2[i].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for i, b in enumerate(b_history):
        biases = b.cpu().numpy()

        # Plot the distribution of biases
        axes3[i].hist(biases, bins=np.linspace(-2, 0.5, num=21), alpha=0.75, range=(-2,0.5))
        if i == 0:
            axes3[i].set_ylabel('b')
            axes3[i].tick_params(axis='x', labelsize=8)
        else:
            axes3[i].set_xticklabels([])
            axes3[i].set_yticklabels([])
        
        axes3[i].set_xlim(-2, 0.5)
        axes3[i].set_xticks(np.arange(-2, 0.6, 0.5))
        axes3[i].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Set up the subplot for the loss function
    axes4 = fig.add_subplot(gs[3, :])
    rolling_loss = np.convolve(loss_history, np.ones(smoothing_window)/smoothing_window, mode='valid')
    rolling_variance = np.array([np.var(loss_history[i - smoothing_window//2 : i + smoothing_window//2]) 
                                for i in range(smoothing_window//2, len(loss_history) - smoothing_window//2 + 1)])
    upper_band = rolling_loss + np.sqrt(rolling_variance)
    lower_band = rolling_loss - np.sqrt(rolling_variance)

    loss_plot_range = range(smoothing_window//2, len(rolling_loss)+smoothing_window//2)
    axes4.plot(loss_plot_range, rolling_loss, label="training loss")
    axes4.fill_between(loss_plot_range, lower_band, upper_band, color='gray', alpha=0.1)

    #axes4_frob = axes4.twinx()
    #axes4_frob.plot(stat_epochs, dims_per_feature, color='g', marker='o', alpha=0.3, label="Dims per feature")
    #axes4_frob.set_ylabel('Dims per feature')
    axes4_eigen = axes4.twinx()
    axes4_eigen.plot(covariance_epochs, covariance_maxeigenvalues, color='g', marker='o', alpha=0.3, label="Max cov eigen")
    axes4_eigen.set_ylabel('Max cov eigen')
    axes4_eigen.set_yscale('log')

    axes4.scatter(snapshot_epoch, [loss_history[i - 1] for i in snapshot_epoch], color='r', marker='o')
    axes4.set_xticklabels([])
    axes4.set_ylabel('Losses')

    if not args.max_loss_plot:
        rolling_loss_max = energy_history[0] / total_train + 0.04
    else:
        rolling_loss_max = args.max_loss_plot

    axes4.set_ylim([np.min(rolling_loss) - 0.02, rolling_loss_max])
    axes4.set_xlim([0, max(snapshot_epoch) + 20])
    axes4.grid(axis='y', alpha=0.3)
    axes4.legend(loc='upper right')
    
    # Set up the subplot for lfe, energy and hatlambda
    axes5 = fig.add_subplot(gs[4, :])

    axes5.plot(stat_epochs, lfe_history, "--o", label="lfe")
    axes5.plot(stat_epochs, energy_history, color='g', marker='o', label="energy")
    axes5.legend(loc='lower left')
    axes5_hatlambda = axes5.twinx()
    axes5_hatlambda.plot(stat_epochs, hatlambda_history, color='r', marker='x', alpha=0.3, label="hatlambda")
    axes5_hatlambda.legend(loc='lower right')

    axes5.set_xlabel('Epoch')
    axes5.set_ylabel('Energies')
    axes5.set_xlim([0, max(snapshot_epoch) + 20])
    axes5_hatlambda.set_ylabel('Hat lambda')
    axes5.grid(axis='y', alpha=0.3)

    def _get_save_filepath(outputdir):
        max_num = 0
        name = f"m{m}n{n}"
        for filename in os.listdir(outputdir):
            match = re.match(name + r'_(\d+)\.png', filename)
            if match is not None:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

        filepath = os.path.join(outputdir, name + f"_{max_num + 1}")
        print(f"Filepath constructed: {filepath}")
        return filepath

    if args.save_plot and args.outputdir:
        outputdir = os.path.join(args.outputdir, f"m{m}n{n}")
        os.makedirs(outputdir, exist_ok=True)
        fig.savefig(_get_save_filepath(outputdir))

    plt.show()

    #if args.save_model:
    #    torch.save(net.state_dict(), _get_save_filepath("model.pth"))

if __name__ == "__main__":
    args = parse_commandline().parse_args()
    main(args)