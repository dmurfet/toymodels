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

from toymodels import ToyModelsDataset, ToyModelsNet
from slt import LearningMachine

def parse_commandline():
    parser = argparse.ArgumentParser(description="SLT Toy Model")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=512)
    parser.add_argument("--m", help="Number of hidden dimensions", type=int, default=2)
    parser.add_argument("--n", help="Number of input dimensions", type=int, default=5)
    parser.add_argument("--epochs", help="epochs", type=int, default=6000)
    parser.add_argument("--show", help="plt.show() if specified.", action="store_true")
    parser.add_argument("--sgld_chains", help="Number of SGLD chains to average over during posterior estimates", type=int, default=5)
    parser.add_argument("--init_polygon", help="Initial weight matrix", type=int, default=2)
    parser.add_argument("--lr", help="Initial learning rate", type=float, default=1e-3)
    parser.add_argument("--polygon_stats", help="Prints only energy and entropy stats for polygons", action="store_true")
    parser.add_argument("--max_stat_batches", help="When giving polygon stats, range of batches", type=int, default=10)
    parser.add_argument("--hatlambdas", help="Number of hatlambdas to compute", type=int, default=20)
    parser.add_argument("--gpu", help="Use GPU, off by default", action="store_true")
    parser.add_argument("--truth_gamma", help="Related to std for true distribution", type=int, default=1)
    return parser

def main(args):
    # Hyperparameters
    m, n = args.m, args.n
    num_epochs = args.epochs
    init_polygon = args.init_polygon
    lr_init = args.lr
    truth_gamma = args.truth_gamma # 1/sqrt(truth_gamma) is the std of the true distribution q(y|x)
    
    steps_per_epoch = 128
    decay_factor = 1
    num_plots = 5
    first_snapshot_epoch = 250
    plot_interval = (num_epochs - first_snapshot_epoch) // (num_plots - 1)
    decay_interval = 2 * plot_interval
    smoothing_window = num_epochs // 100
    show_lfe = True

    print(f"SLT Toy Model m={m},n={n}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Device: {device}")
    
    # The training set is allocated at the beginning using the custom dataset
    total_train = steps_per_epoch * num_epochs
    trainset = ToyModelsDataset(total_train, n, truth_gamma)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    trainloader_batched = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    lr = lr_init
    criterion = nn.MSELoss()

    # The main model for training
    model = ToyModelsNet(n, m, init_config=init_polygon, noise_scale=0.1)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # The learning machine, for computing free energy and lambda
    machine = LearningMachine(model, trainloader_batched, criterion, optimizer, device, truth_gamma, args.sgld_chains)
    
    W_history = []
    b_history = []
    loss_history = []
    snapshot_epoch = []
    lfe_epochs = []
    lfe_history = []
    energy_history = []
    hatlambda_history = []

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

        if show_lfe and epoch > first_snapshot_epoch and (epoch + 1) % (num_epochs // args.hatlambdas) == 0:
            lfe_epochs.append(epoch+1)
            energy = machine.compute_energy()
            lfe = machine.compute_local_free_energy()
            hatlambda = (lfe - energy) / np.log(total_train)

            energy_history.append(energy.cpu().detach().clone())
            lfe_history.append(lfe.cpu().detach().clone())
            hatlambda_history.append(hatlambda.cpu().detach().clone())

        epoch_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.5f}')

        loss_history.append(epoch_loss)

        if (epoch - first_snapshot_epoch + 1) % decay_interval == 0:
            lr = lr * decay_factor
            if decay_factor != 1:
                print(f'New learning rate: {lr:.5f}')

    ####################
    # Plotting
    
    num_plots = len(W_history)

    # Plot columns of W matrices
    if show_lfe:
        plot_rows = 5
        plot_ratios = [2,1,1,2,2]
        plot_height = 55
    else:
        plot_rows = 4
        plot_ratios = [2,1,1,2]
        plot_height = 45

    gs = gridspec.GridSpec(plot_rows, num_plots, height_ratios=plot_ratios)
    fig = plt.figure(figsize=(35, plot_height))
    fig.suptitle(f"Toy models (n={n}, m={m}, lr={lr_init})", fontsize=10)
    if m == 2:
        axes1 = [fig.add_subplot(gs[0, i]) for i in range(num_plots)]
    elif m == 3:
        axes1 = [fig.add_subplot(gs[0,i], projection='3d') for i in range(num_plots)]

    axes2 = [fig.add_subplot(gs[1, i]) for i in range(num_plots)]
    axes3 = [fig.add_subplot(gs[2, i]) for i in range(num_plots)]

    for i, W in enumerate(W_history):
        for j in range(n):
            column_vector = W.cpu()[:, j].numpy()

            # Plot the arrow
            if m == 2:
                # 2D
                axes1[i].quiver(0, 0, column_vector[0], column_vector[1], angles='xy', scale_units='xy', scale=1, label=f'Column {j+1}')
            elif m == 3:
                # 3D
                axes1[i].quiver(0, 0, 0, column_vector[0], column_vector[1], column_vector[2], 
                        color=plt.cm.jet(j/n), linewidth=1.5, label=f'Column {j+1}')
        

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
        
        axes2[i].set_xlim(0, 2)
        axes2[i].set_xticks(np.arange(0, 2.1, 0.5))
        axes2[i].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    for i, b in enumerate(b_history):
        biases = b.cpu().numpy()

        # Plot the distribution of column norms
        axes3[i].hist(biases, bins=np.linspace(-2, 0.5, num=21), alpha=0.75, range=(-2,0.5))
        if i == 0:
            axes3[i].set_ylabel('b')
            axes3[i].tick_params(axis='x', labelsize=8)
        else:
            axes3[i].set_xticklabels([])
        
        axes3[i].set_xlim(-2, 0.5)
        axes3[i].set_xticks(np.arange(-2, 0.6, 0.5))
        axes3[i].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Calculate frame potential for each W in W_history
    def fp(W):
        W = W / torch.norm(W, dim=0, keepdim=True)
        out = torch.matmul(W.T, W)
        out = torch.norm(out, p='fro').item() ** 2 - (n**2)/m
        return out

    fp_loss = [fp(W) for W in W_history]

    # Set up the subplot for the loss function
    axes4 = fig.add_subplot(gs[3, :])
    rolling_loss = np.convolve(loss_history, np.ones(smoothing_window)/smoothing_window, mode='valid')
    rolling_variance = np.array([np.var(loss_history[i - smoothing_window//2 : i + smoothing_window//2]) 
                                for i in range(smoothing_window//2, len(loss_history) - smoothing_window//2 + 1)])
    upper_band = rolling_loss + np.sqrt(rolling_variance)
    lower_band = rolling_loss - np.sqrt(rolling_variance)

    loss_plot_range = range(smoothing_window//2, len(rolling_loss)+smoothing_window//2)
    axes4.plot(loss_plot_range, rolling_loss, label="loss")
    axes4.fill_between(loss_plot_range, lower_band, upper_band, color='gray', alpha=0.1)

    axes4_frob = axes4.twinx()
    axes4_frob.plot(snapshot_epoch, fp_loss, color='g', marker='o', alpha=0.3, label="FP")

    axes4.scatter(snapshot_epoch, [loss_history[i - 1] for i in snapshot_epoch], color='r', marker='o')
    axes4.set_xticklabels([])
    axes4.set_ylabel('Loss (batch=1)')
    axes4_frob.set_ylabel('FP (green)')
    rolling_loss_max = energy_history[0] / total_train + 0.01
    axes4.set_ylim([0, rolling_loss_max])
    axes4.set_xlim([0, max(snapshot_epoch) + 20])
    axes4.grid(axis='y', alpha=0.3)
    axes4.legend(loc='lower left')

    # Set up the subplot for lfe, energy and hatlambda
    if show_lfe:
        axes5 = fig.add_subplot(gs[4, :])

        axes5.plot(lfe_epochs, lfe_history, "--o", label="lfe")
        axes5.plot(lfe_epochs, energy_history, color='g', marker='o', label="energy")
        axes5.legend(loc='lower left')
        axes5_hatlambda = axes5.twinx()
        axes5_hatlambda.plot(lfe_epochs, hatlambda_history, color='r', marker='x', alpha=0.3)

        axes5.set_xlabel('Epoch')
        axes5.set_ylabel('Energies')
        axes5.set_xlim([0, max(snapshot_epoch) + 20])
        axes5_hatlambda.set_ylabel('Hat lambda')
        axes5.grid(axis='y', alpha=0.3)

    plt.show()

if __name__ == "__main__":
    args = parse_commandline().parse_args()
    main(args)