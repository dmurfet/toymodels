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
import os
import argparse

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
    parser.add_argument("--hatlambdas", help="Number of hatlambdas to compute", type=int, default=20)
    return parser

class ToyModelsDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, n):
        self.num_samples = num_samples
        self.n = n
        self.data = []
        self.targets = []

        for _ in range(num_samples):
            a = torch.randint(1, self.n + 1, (1,)).item()
            lambda_val = torch.rand(1).item()

            # Create input vector x
            x = torch.zeros(self.n)
            x[a - 1] = lambda_val

            self.data.append(x)
            self.targets.append(x.clone())  # In this case, the target is the same as the input

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
class ToyModelsNet(nn.Module):
    def __init__(self, in_features, out_features, init_config=None, noise_scale=1):
        super(ToyModelsNet, self).__init__()
        
        if init_config:
            assert in_features >= init_config, "Not enough columns in W"

            # Populate W with a sequence of columns, first being (1,0,...)
            # and then the remainder being rotated by 2 * pi * i / n for i = 0, ... , k - 1
            # and then zeros, where k = init_config
            self.b = nn.Parameter(torch.zeros(in_features) + noise_scale * torch.randn(in_features))
            W = torch.zeros((out_features, in_features))
            W[0, 0] = 1

            theta = 2 * np.pi / init_config
            rotation_matrix = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float)
            
            rotated_col = W[:2,0]

            for i in range(1, in_features):
                if i < init_config:
                    rotated_col = torch.matmul(rotation_matrix, rotated_col)
                    W[:2, i] = rotated_col
                    W[:, i] += noise_scale * torch.randn_like(W[:, i])

            self.W = nn.Parameter(W)
        else:
            # Random W and b
            self.W = nn.Parameter(noise_scale * torch.randn(out_features, in_features))
            self.b = nn.Parameter(noise_scale * torch.randn(in_features))

    def forward(self, x):
        # compute ReLU(W^TWx + b)
        #out = torch.matmul(out, x.unsqueeze(-1)) + self.b.unsqueeze(-1)
        out = torch.matmul(self.W.T, self.W)
        out = torch.bmm(out.unsqueeze(0).repeat(x.shape[0], 1, 1), x.unsqueeze(-1)).squeeze(-1)
        out += self.b.unsqueeze(0).repeat(x.shape[0], 1)
        out = torch.relu(out)
        return out

def main(args):
    # Hyperparameters
    m, n = args.m, args.n
    num_epochs = args.epochs
    init_polygon = args.init_polygon
    num_sgld_chains = args.sgld_chains 
    lr_init = args.lr

    steps_per_epoch = 128
    decay_factor = 1
    num_plots = 5
    first_snapshot_epoch = 250
    plot_interval = (num_epochs - first_snapshot_epoch) // (num_plots - 1)
    decay_interval = 2 * plot_interval
    smoothing_window = num_epochs // 100
    show_lfe = True

    print(f"SLT Toy Model m={m},n={n}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def compute_energy(trainloader, model, criterion):
        # this is nL_n
        energies = []
        with torch.no_grad():
            for data in trainloader:
                x = data[0].to(device)
                outputs = model(x)
                loss = criterion(outputs, x)
                energies.append(loss * args.batch_size)
        return sum(energies)
    
    def compute_local_free_energy(trainloader, model, criterion, optimizer, sgld_chains=num_sgld_chains):
        def closure():
            dataiter = iter(trainloader)
            data = next(dataiter)
            x = data[0].to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, x)
            loss.backward()
            return loss

        # Store the current parameter, since we will be doing SGLD trajectories
        # starting here, and will need to reset        
        param_groups = []
        for group in optimizer.param_groups:
            group_copy = dict()
            group_copy["params"] = deepcopy(group["params"])
            param_groups.append(group_copy)
        
        num_iter = 100 # default 100
        gamma = 1000 # default 1000
        epsilon = 0.1 / 10000 # default 0.1 / 10000
        Lms = []

        def reset_chain():
            # Reset the optimiser's parameter to what it was before SGLD
            for group_index, group in enumerate(optimizer.param_groups):
                for param_index, p in enumerate(group["params"]):
                    w = param_groups[group_index]["params"][param_index]
                    p.data.copy_(w.data)

        for _ in range(sgld_chains):
            # This outer loop is over SGLD chains, starting at the current optimiser.param
            for _ in range(num_iter):
                with torch.enable_grad():
                    # call a minibatch loss backward so that we have gradient of average minibatch loss with respect to w'
                    loss = closure()

                for group_index, group in enumerate(optimizer.param_groups):
                    for param_index, w_prime in enumerate(group["params"]):
                        w = param_groups[group_index]["params"][param_index]
                        # by using the scaling below of w_prime.grad, we are ensuring the SGLD is for full batch posterior at inverse temp 1/log n
                        dx_prime = -w_prime.grad.data / np.log(total_train) * total_train  #
                        dx_prime.add_(w_prime.data - w.data, alpha=-gamma)
                        w_prime.data.add_(dx_prime, alpha=epsilon / 2)
                        gaussian_noise = torch.empty_like(w_prime)
                        gaussian_noise.normal_()
                        w_prime.data.add_(gaussian_noise, alpha=(np.sqrt(epsilon)))

                Lms.append(loss)
                # record average of nL_n(x_l') for l = 1,...,L by using n* minbatch L_m as surrogate
            
            reset_chain()

        local_free_energy = total_train * sum(Lms) / len(Lms)

        reset_chain()

        return local_free_energy
    
    # The training set is allocated at the beginning using the custom dataset
    total_train = steps_per_epoch * num_epochs
    trainset = ToyModelsDataset(total_train, n)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    trainloader_batched = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    lr = lr_init
    criterion = nn.MSELoss()

    if args.polygon_stats:
        # Print the energies of polygons with current trainset
        for k in range(n+1):
            polygon_model = ToyModelsNet(n, m, init_config=k, noise_scale=0)
            polygon_model.to(device)
            energy = compute_energy(trainloader_batched, polygon_model, criterion)
            avg_energy = energy / total_train
            optimizer = optim.SGD(polygon_model.parameters(), lr=lr)
            lfe = compute_local_free_energy(trainloader_batched, polygon_model, criterion, optimizer)
            hatlambda = (lfe - energy) / np.log(total_train)

            print(f"[{k}-gon] energy per sample: {avg_energy:.6f}, hatlambda: {hatlambda:.6f}")
        return

    # The main model for training
    model = ToyModelsNet(n, m, init_config=init_polygon, noise_scale=0.1)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

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
            energy = compute_energy(trainloader_batched, model, criterion)
            lfe = compute_local_free_energy(trainloader_batched, model, criterion, optimizer)
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