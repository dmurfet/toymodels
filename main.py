# Example usage
# python main.py --m 2 --n 5 --epochs 20000 --max_loss_plot 0.06
# python main.py --m 2 --n 8 --epochs 2000 --init_polygon 2 --detect_transitions
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
from scipy.signal import find_peaks, peak_prominences
from sklearn.decomposition import PCA
import re

from toymodels import ToyModelsDataset, ToyModelsNet
from slt import LearningMachine

torch.set_printoptions(precision=4)

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
    parser.add_argument("--detect_transitions", help="Whether to run the transition detection", action="store_true")
    parser.add_argument("--prominence_cutoff", help="Prominence cutoff for bifurcation detection", type=float, default=1e-5)
    parser.add_argument("--covariance_checkpoints", help="Number of points at which to compute covariance matrices", type=int, default=120)
    parser.add_argument(
        "--outputdir",
        help="Path to output directory. Create if not exist.",
        type=str,
        default=None,
    )
    return parser

# Remark: prominence cutoff defaulted to 5e-4 for maximum eigenvalues, now 1e-5 for covariane derivative

def main(args):
    # Hyperparameters
    m, n = args.m, args.n
    num_epochs = args.epochs
    init_polygon = args.init_polygon
    prominence_cutoff = args.prominence_cutoff

    lr_init = args.lr
    truth_gamma = args.truth_gamma # 1/sqrt(truth_gamma) is the std of the true distribution q(y|x)

    num_covariance_checkpoints = args.covariance_checkpoints
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

    # Historical note 6-9-23: originally our code had a bug, which meant the covariance matrix was computed
    # from the following set of samples: starting at the saved W and saved b, it did the following 200 times
    # in a row: take 100 SGD steps and record the weight (without restarting).

    num_trajectories = 2
    num_covariant_chain_steps = 2 * covariance_interval * steps_per_epoch # 200 * 100
    covariance_epochs = []
    covariance_maxeigenvalues = []
    covariance_maxeigenvectors = []
    covariance_secondmaxeigenvectors = []
    covariance_matrices = []
    covariance_meanvectors = []
    covariance_eigenratios = []

    print("")
    print("Computing covariance matrices")
    
    def sgd_trajectory(initial_W, initial_b, num_steps, thinning_factor=100):
        # Create a new model and optimizer instance
        new_model = ToyModelsNet(n, m, init_config=init_polygon, noise_scale=args.init_noise_scale, use_optimal=True)
        new_model.W.data = initial_W.clone().detach()
        new_model.b.data = initial_b.clone().detach()
        new_model.to(device)
        new_optimizer = optim.SGD(new_model.parameters(), lr=lr)

        # Create an independent training set        
        new_trainset = ToyModelsDataset(num_steps, n, truth_gamma)
        new_trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=1, shuffle=True)
        new_dataiter = iter(new_trainloader)

        trajectory = []

        for i in range(num_steps):
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
            
            if i > 0 and i % thinning_factor == 0:
                W_flat = torch.flatten(new_model.W)
                combined_vector = torch.cat((W_flat, new_model.b))
                trajectory.append(combined_vector.cpu().detach().clone())

        return trajectory

    for i, _ in enumerate(model_state_history):
        this_state = model_state_history[i]
        if i == 0:
            previous_state = this_state
        else:
            previous_state = model_state_history[i-1]

        covariance_epochs.append(this_state["epoch"])

        # We run SGD trajectories starting at the previous checkpoint and running roughly long
        # enough to have made it to the next checkpoint. Recall that the number of steps between
        # checkpoints is covariance_interval * steps_per_epoch

        X_list = []
        for i in range(num_trajectories):   
            torch.manual_seed(i)
            torch.cuda.manual_seed_all(i)
            trajectory = sgd_trajectory(previous_state["W"],previous_state["b"],num_covariant_chain_steps)
            X_list += trajectory

        X = torch.stack(X_list)
        mean_vector = torch.mean(X, dim=0)
        X_centered = X - mean_vector
        cov_matrix = X_centered.t().mm(X_centered) / (X.size(0) - 1) # C = E[ (w - w^*)^T (w - w^*) ]
        
        eigenresult = torch.linalg.eig(cov_matrix)
        eigenvalues = eigenresult.eigenvalues.real
        max_eigenvalue, idx = torch.max(eigenvalues, dim=0)
        max_eigenvector = eigenresult.eigenvectors[:, idx].real
        min_eigenvalue, _ = torch.min(eigenvalues, dim=0)

        sorted_eigenvalues, sorted_indices = torch.sort(torch.abs(eigenvalues), descending=True)
        second_largest_eigenvector = eigenresult.eigenvectors[:, sorted_indices[1]].real

        eigen_ratio = sorted_eigenvalues[0] / sorted_eigenvalues[1]

        print("Max eigenvalue: {:.5e}   Min eigenvalue: {:.5e}   Eigenratio: {:.5e}".format(max_eigenvalue.item(), min_eigenvalue.item(),eigen_ratio.item()))

        covariance_maxeigenvalues.append(max_eigenvalue)
        covariance_maxeigenvectors.append(max_eigenvector)
        covariance_secondmaxeigenvectors.append(second_largest_eigenvector)
        covariance_matrices.append(cov_matrix)
        covariance_meanvectors.append(mean_vector)
        covariance_eigenratios.append(eigen_ratio)

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
    fig.suptitle(f"Toy models (n={n}, m={m}, init_polygon={args.init_polygon or 'None'}{'' if not args.detect_transitions else ', prominence=' + str(args.prominence_cutoff)})", fontsize=10)
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
    axes4_eigen.plot(covariance_epochs, covariance_maxeigenvalues, color='g', marker='o', alpha=0.3, label="Max eigenvalue")
    axes4_eigen.set_ylabel('Max eigenvalue')
    axes4_eigen.set_yscale('log')

    axes4.scatter(snapshot_epoch, [loss_history[i - 1] for i in snapshot_epoch], color='r', marker='o')
    axes4.set_xticklabels([])
    axes4.set_ylabel('Loss')

    if not args.max_loss_plot:
        rolling_loss_max = energy_history[0] / total_train + 0.04
    else:
        rolling_loss_max = args.max_loss_plot

    axes4.set_ylim([np.min(rolling_loss) - 0.02, rolling_loss_max])
    axes4.set_xlim([0, max(snapshot_epoch) + 20])
    axes4.grid(axis='y', alpha=0.3)
    axes4.legend(loc='upper right')
    
    if not args.detect_transitions:
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
    else:
        # Look for places where the covariant matrix is changing relatively little,
        # as this is a precondition for a bifurcation (stationary point of the
        # Ornstein-Uhlenbeck process).
        
        # Uncomment to use peaks in maximum eigenvalues of C as the target
        #peaks, _ = find_peaks(covariance_maxeigenvalues)
        #prominences, _, _ = peak_prominences(covariance_maxeigenvalues, peaks)
        covariance_deltas = [torch.norm(covariance_matrices[i] - covariance_matrices[i-1],p='fro') for i in range(1,len(covariance_epochs))]

        # Look for local _minima_ in the deltas between subsequent covariance matrices
        # as this should detect "corners"
        neg_covariance_deltas = [-a for a in covariance_deltas]
        peaks, _ = find_peaks(neg_covariance_deltas)
        prominences, _, _ = peak_prominences(neg_covariance_deltas, peaks)
        print(peaks)
        print(prominences)

        # Remove insignificant peaks (the + 1 since we cut off the beginning of covariance_deltas)
        peaks = [p + 1 for (p,prom) in zip(peaks,prominences) if prom > prominence_cutoff]
        data_for_peaks = []
        lya_window_size = round(args.covariance_checkpoints / 4)

        if len(peaks) > 0:
            axes5 = fig.add_subplot(gs[4, :])
            axes5.plot(loss_plot_range, rolling_loss)
            axes5.fill_between(loss_plot_range, lower_band, upper_band, color='gray', alpha=0.1)
            axes5.set_ylim([np.min(rolling_loss) - 0.02, rolling_loss_max])
            axes5.set_xlim([0, max(snapshot_epoch) + 20])
            axes5.grid(axis='y', alpha=0.3)
            axes5.set_ylabel('Lyapunov')

            # For each peak, plot a Lyapunov function
            for i, peak in enumerate(peaks):
                # Put a vertical line in the plot at this transition
                axes5.axvline(x=covariance_epochs[peak], color='blue', linestyle='--')

                C = covariance_matrices[peak]
                lyapunov = []

                start_index = max(0, peak-lya_window_size)
                end_index = min(len(model_state_history),peak+lya_window_size) # was + 1
                weight_vectors_for_this_peak = []

                for _, saved_state in enumerate(model_state_history[start_index:end_index]):
                    # NOTE really we should add the value of the loss at the averaged weights, we're hoping the rolling loss is close
                    W = saved_state["W"]
                    b = saved_state["b"]
                    w_vec = torch.cat((torch.flatten(W), b)) - covariance_meanvectors[peak]
                    weight_vectors_for_this_peak.append(w_vec)
                    l = np.dot(w_vec, np.dot(C, w_vec)) + rolling_loss[covariance_epochs[peak]]
                    lyapunov.append(l)
                
                data_for_peaks.append({"weights": weight_vectors_for_this_peak, "peak": peak})
                #axes5.plot(covariance_epochs[start_index:end_index], lyapunov, label="transition " + str(i))
            
            axes5_covdelta = axes5.twinx()
            axes5_covdelta.plot(covariance_epochs[1:], covariance_deltas, color='b', marker='o', alpha=0.3, label="Cov delta")
            axes5_covdelta.set_ylabel('Cov delta')
            axes5_covdelta.set_yscale('log')

            axes5.legend(loc='upper right')

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

    if args.detect_transitions:
        def plot_weight_vectors_pca(weight_vectors, max_eigenvector, second_maxeigenvector, C, num_transition, eigenratio, trajectories_start, meanvector, num_sgd_steps):
            weight_vectors = [w.detach().cpu().numpy() for w in weight_vectors]
            max_eigenvector = max_eigenvector.detach().cpu().numpy()

            # Compute a number of SGD trajectories
            all_trajectories = []
            weights_and_trajectories = []
            weights_and_trajectories += weight_vectors

            for j in range(5):
                torch.manual_seed(j)
                torch.cuda.manual_seed_all(j)
                trajectories = sgd_trajectory(trajectories_start["W"],trajectories_start["b"],num_sgd_steps)
                t_list = [(t - meanvector).numpy() for t in trajectories]
                weights_and_trajectories += t_list
                all_trajectories.append(np.array(t_list))

            pca = PCA(n_components=2)
            pca.fit_transform(weights_and_trajectories)
            variance_explained = pca.explained_variance_ratio_
            
            # Plot the projected data of the original SGD trajectory
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            fig.suptitle(f"PCA projection of development, transition={num_transition}, leading eigenvalue ratio={eigenratio:.5e}")
            
            projected_data = pca.transform(weight_vectors)
            x_min, x_max = projected_data[:, 0].min(), projected_data[:, 0].max()
            y_min, y_max = projected_data[:, 1].min(), projected_data[:, 1].max()
            colormap = plt.cm.get_cmap('viridis', 256)

            for _, trajectory in enumerate(all_trajectories):
                projected_trajectory = pca.transform(trajectory)
                
                colors = [colormap(i) for i in np.linspace(0, 1, len(trajectory))]
                for i, (x,y) in enumerate(projected_trajectory):
                    axes[0].scatter(x, y, color=colors[i])

                x_min = min(x_min, projected_trajectory[:,0].min())
                x_max = max(x_max, projected_trajectory[:,0].max())
                y_min = min(y_min, projected_trajectory[:,1].min())
                y_max = max(y_max, projected_trajectory[:,1].max())
            
            # Scatter plot the true trajectory
            colormap = plt.cm.get_cmap('Greys', 256)
            colors = [colormap(i) for i in np.linspace(0, 1, len(weight_vectors))]
            axes[0].scatter(0, 0, color='black', s=50)
            for i, (x,y) in enumerate(projected_data):
                axes[0].scatter(x, y, color=colors[i], edgecolors='black', linewidths=1)

            # Draw a line in the direction of the two eigenvectors of the covariance matrix
            x_delta = x_max - x_min
            y_delta = y_max - y_min
            x_min -= 0.05 * x_delta
            x_max += 0.05 * x_delta
            y_min -= 0.05 * y_delta
            y_max += 0.05 * y_delta

            vector_x, vector_y = pca.transform(max_eigenvector.reshape(1, -1)).ravel()
            if vector_x == 0:
                axes[0].plot([0, 0], [y_min, y_max], 'k-', linewidth=1, label='Principal eigenspace')
            else:
                slope = vector_y / vector_x
                y1 = slope * x_min
                y2 = slope * x_max
                axes[0].plot([x_min, x_max], [y1, y2], 'k-', linewidth=1, label='Principal eigenspace')

            vector_x, vector_y = pca.transform(second_maxeigenvector.reshape(1, -1)).ravel()
            if vector_x == 0:
                axes[0].plot([0, 0], [y_min, y_max], 'k--', linewidth=1, label='Secondary eigenspace')
            else:
                slope = vector_y / vector_x
                y1 = slope * x_min
                y2 = slope * x_max
                axes[0].plot([x_min, x_max], [y1, y2], 'k--', linewidth=1, label='Secondary eigenspace')
            
            # Create a grid for the density plot
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

            # Compute function values for the grid
            Z = np.zeros(xx.shape)
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    u = xx[i, j] * pca.components_[0] + yy[i, j] * pca.components_[1]
                    Z[i, j] = np.dot(u.T, np.dot(C, u))

            # Plot the density map
            Z = (Z - Z.min()) / (Z.max() - Z.min()) # rescale
            im = axes[0].imshow(Z, interpolation='bilinear', origin='lower',
               extent=(x_min, x_max, y_min, y_max), cmap='Blues', alpha=0.75, aspect='auto')
            
            # Lyapunov values are from 0 (white) to 1 (blue)
            # Development time is from dark purple to yellow

            axes[0].set_xlabel('PC 1')
            axes[0].set_ylabel('PC 2')
            axes[0].legend(loc='upper right')

            # Plot variance explained
            axes[1].bar(np.arange(len(variance_explained)), variance_explained, alpha=0.7, align='center')
            axes[1].set_ylabel('Proportion of Variance')
            axes[1].set_xlabel('PC')
            axes[1].set_xticks(np.arange(len(variance_explained)))

        for i, peak in enumerate(peaks):
            C = covariance_matrices[peak]
            eigenvector = covariance_maxeigenvectors[peak]
            second_eigenvector = covariance_secondmaxeigenvectors[peak]
            meanvector = covariance_meanvectors[peak]
            eigenratio = covariance_eigenratios[peak]
            sgd_length = 30 # 10
            start_index = max(0, peak-sgd_length)
            saved_state = model_state_history[start_index] #model_state_history[peak]
            num_sgd_steps = 2 * sgd_length * covariance_interval * steps_per_epoch

            plot_weight_vectors_pca(data_for_peaks[i]["weights"], eigenvector, second_eigenvector, C, i, eigenratio, saved_state, meanvector, num_sgd_steps)

    plt.show()

    #if args.save_model:
    #    torch.save(net.state_dict(), _get_save_filepath("model.pth"))

if __name__ == "__main__":
    args = parse_commandline().parse_args()
    main(args)