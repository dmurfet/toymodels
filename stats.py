# Example usage:
#
# python stats.py --m 2 --n 5 --epochs 2000 --max_stat_batches 20 --truth_gamma 2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('TkAgg') # on MacOSX
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from itertools import cycle

from toymodels import ToyModelsDataset, ToyModelsNet
from slt import LearningMachine

# TODO
#
#   - Collect statistics at actual local minima
#   - Automatic mapping of phase transitions

def parse_commandline():
    parser = argparse.ArgumentParser(description="SLT Toy Model")
    parser.add_argument("--batch_size", help="Batch size", type=int, default=512)
    parser.add_argument("--m", help="Number of hidden dimensions", type=int, default=2)
    parser.add_argument("--n", help="Number of input dimensions", type=int, default=5)
    parser.add_argument("--epochs", help="epochs", type=int, default=6000)
    parser.add_argument("--lr", help="Initial learning rate", type=float, default=1e-3)
    parser.add_argument("--sgld_chains", help="Number of SGLD chains to average over during posterior estimates", type=int, default=200)
    parser.add_argument("--max_stat_batches", help="When giving polygon stats, range of batches", type=int, default=10)
    parser.add_argument("--gpu", help="Use GPU, off by default", action="store_true")
    parser.add_argument("--truth_gamma", help="Related to std for true distribution", type=int, default=10)
    return parser

def show_lfe_hatlambda_vs_numbatches(args):
    m, n = args.m, args.n
    num_epochs = args.epochs
    truth_gamma = args.truth_gamma # 1/sqrt(truth_gamma) is the std of the true distribution q(y|x)
    steps_per_epoch = 128
    
    print(f"SLT Toy Model m={m},n={n}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Device: {device}")
    
    # The training set is allocated at the beginning using the custom dataset
    total_train = steps_per_epoch * num_epochs
    trainset = ToyModelsDataset(total_train, n, truth_gamma)
    trainloader_batched = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    lr = args.lr
    criterion = nn.MSELoss()

    lfes = []
    hatlambdas = []
    batch_sizes = range(1,args.max_stat_batches+1)

    # Print the energies of polygons with current trainset
    for k in range(2, n+1):
        polygon_model = ToyModelsNet(n, m, init_config=k, noise_scale=0)
        polygon_model.to(device)
        optimizer = optim.SGD(polygon_model.parameters(), lr=lr)

        # Burn-in to get a nearby critical point (ideally)
        dataiter = iter(trainloader_batched)
        for _ in range(100):
            data = next(dataiter)
            x = data[0].to(device)
            
            # Forward pass
            output = polygon_model(x)
            loss = criterion(output, x)

            loss = loss * truth_gamma / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        machine = LearningMachine(polygon_model, trainloader_batched, criterion, optimizer, device, truth_gamma, args.sgld_chains)
    
        energy = machine.compute_energy()
        avg_energy = energy / total_train

        hatlambdas_per_polygon = []
        lfes_per_polygon = []
        
        for num_batches in batch_sizes:
            lfe = machine.compute_local_free_energy(num_batches=num_batches)
            hatlambda = (lfe - energy) / np.log(total_train)
            lfes_per_polygon.append(lfe.cpu().detach().clone())
            hatlambdas_per_polygon.append(hatlambda.cpu().detach().clone())

        hatlambdas.append(hatlambdas_per_polygon)
        lfes.append(lfes_per_polygon)

        print(f"[{k}-gon] energy per sample: {avg_energy:.4f}, hatlambda: {hatlambda:.4f}")

    if len(batch_sizes) > 1:
        fig = plt.figure(figsize=(25, 20))
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        
        for k in range(2, n+1):
            ax1.plot(batch_sizes, lfes[k-2], "--o", label=f"{k}-gon")
            ax2.plot(batch_sizes, hatlambdas[k-2], "--o", label=f"{k}-gon")

        ax1.set_xticks(batch_sizes)
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylabel('Local free energy')

        ax2.set_xticks(batch_sizes)
        ax2.set_xlabel("Number of batches")
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylabel('Hat lambda')

        plt.suptitle(f"Sensitivity to num_batches, epochs={num_epochs}, m={m}, n={n}, truth_gamma={truth_gamma}")
        plt.show()

# Plot hatlambdas vs N (dataset size)
#
# Suppose w^* is near an optimum w_0 of L_n, so that with beta = 1/logn
#
#   E^beta_w[ NL_N(w) ] = NL_N(w_0) + lambda logN
# 
# Then also
#
#   E^beta_w[ NL_N(w) ] - NL_N(w^*) = NL_N(w_0) - NL_N(w^*) + lambda logn
#   1/logN( E^beta_w[ NL'_N(w) ] - NL'_N(w^*) ) = N/logN[ L'_N(w_0) - L'_N(w^*) ] + lambda
#
# We assume that for N sufficiently large, L_N(w_0) - L_N(w^*) is constant, noting that
# we do not know L_N(w_0). Note the left hand side is hatlambda(N). So this shows that
# if we fit hatlambda(N) against N to a function of the form a N/logN + b we expect that
# b will be the RLCT.

def main(args):
    m, n = args.m, args.n
    truth_gamma = args.truth_gamma # 1/sqrt(truth_gamma) is the std of the true distribution q(y|x)
    steps_per_epoch = 128
    lr = args.lr
    steps_per_epoch = 128

    print(f"SLT Toy Model stats m={m},n={n}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"  [Device: {device}]")
    criterion = nn.MSELoss()

    epochs = [6000, 8000]
    #epochs = [6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

    print("  [Generating dataset]")
    total_trainset = ToyModelsDataset(steps_per_epoch * max(epochs), n, truth_gamma)
    
    hatlambdas = []
    energies = []
    lfes = []
    
    min_polygon = 4
    for k in range(min_polygon, n+1):
        print(f"{k}-gon")
        polygon_model = ToyModelsNet(n, m, init_config=k, noise_scale=0.01, use_optimal=True)
        polygon_model.to(device)
        optimizer = optim.SGD(polygon_model.parameters(), lr=lr)
        
        # Burn-in to get a nearby critical point (ideally)
        print(f"  [Burning in]")
        trainloader_tune = torch.utils.data.DataLoader(total_trainset, batch_size=1, shuffle=True)    
        dataiter = iter(trainloader_tune)
        num_tune_epochs = 3000
        pretune_loss = None
        tune_lr = lr
        for i in range(num_tune_epochs):
            epoch_loss = 0.0
            total_gradient_norm = 0.0

            # Reduce the learning rate
            if (i+1) % (num_tune_epochs // 3) == 0:
                tune_lr = tune_lr * 0.5
                print(f"    [reducing lr {tune_lr}]")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = tune_lr

            for _ in range(steps_per_epoch):
                data = next(dataiter)
                x = data[0].to(device)
                
                # Forward pass
                output = polygon_model(x)
                loss = criterion(output, x)

                loss = loss * truth_gamma / 2

                optimizer.zero_grad()
                loss.backward()

                # Compute gradient norm
                for param in polygon_model.parameters():
                    if param.grad is not None:
                        total_gradient_norm += param.grad.data.norm(2).item()

                optimizer.step()
                epoch_loss += loss.item()
            
            avg_gradient_norm = total_gradient_norm / steps_per_epoch
    
            if i == 0:
                pretune_loss = epoch_loss / steps_per_epoch
        
        print(f"    [final grad norm {avg_gradient_norm:.4f}]")
        print(f"    [loss decrease {pretune_loss - epoch_loss / steps_per_epoch:.4f}]")

        # Reset the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        hatlambdas_per = []
        energies_per = []
        lfes_per = []

        for i, num_epochs in enumerate(epochs):
            # TODO: Consider taking a random subset?
            subset_indices = list(range(steps_per_epoch * num_epochs))
            trainset = Subset(total_trainset, subset_indices)
            trainloader_batched = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)    
            
            machine = LearningMachine(polygon_model, trainloader_batched, criterion, optimizer, device, truth_gamma, args.sgld_chains)
        
            # N = steps_per_epoch * num_epochs
            energy = machine.compute_energy() # NL'_N            
            lfe = machine.compute_local_free_energy() # N E_w^\beta[ L'_M ] ~ N E_w^beta[ L'_N ]
            hatlambda = (lfe - energy) / np.log(steps_per_epoch * num_epochs)

            hatlambdas_per.append(hatlambda.cpu().detach().clone())
            energies_per.append(energy.cpu().detach().clone())
            lfes_per.append(lfe.cpu().detach().clone())

            energy_per_sample = energy / (steps_per_epoch * num_epochs)
            print(f"  [{num_epochs}] energy per sample: {energy_per_sample:.6f}, hatlambda: {hatlambda:.6f}")

        hatlambdas.append(hatlambdas_per)
        energies.append(energies_per)
        lfes.append(lfes_per)

    fig = plt.figure(figsize=(20, 25))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    
    def func(x, a, b):
        return a * x / np.log(x) + b
    
    #def func_lin(x, a, b):
    #    return a * x + b
    def func_lfe(x, a, b):
        return a * x + b * np.log(x)
    
    def fef(N, E, rlct):
        return N * E + rlct * np.log(N)

    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    b_values = {}
    dataset_sizes = [epoch * steps_per_epoch for epoch in epochs]
    intermediate_epochs = np.linspace(min(epochs), max(epochs), 1000)  # 1000 is the number of points
    
    print("\n")
    for k in range(min_polygon, n+1):
        hatlambda_data = hatlambdas[k-min_polygon]
        lfe_data = lfes[k-min_polygon]

        # Fit a function a x/logx + b to the hatlambda data points
        popt, _ = curve_fit(func, dataset_sizes, hatlambda_data)
        popt_lfe, _ = curve_fit(func_lfe, dataset_sizes, lfe_data)
        #popt_lin, _ = curve_fit(func_lin, dataset_sizes, hatlambda_data)

        predicted = [func(dsize, *popt) for dsize in dataset_sizes]
        r2 = r2_score(hatlambda_data, predicted)

        predicted_lfe = [func_lfe(dsize, *popt_lfe) for dsize in dataset_sizes]
        r2_lfe = r2_score(lfe_data, predicted_lfe)

        # We have now fitted the LHS
        #
        #   1/logN( E^beta_w[ NL_N(w) ] - NL_N(w^*) ) = N/logN * a + b
        #
        # which means that
        #
        #   a ~ L'_N(w_0) - L'_N(w^*)
        #   b ~ lambda
        #
        # where w^* is our chosen parameter (the init_config) and w_0 is the
        # hypothesised nearby local optimum. We can therefore compute#
        #
        #   L'_N(w_0) = a + L'_N(w^*)
        #
        # We are assuming L'_N(w_0) - L'_N(w^*) is roughly constant in N, and
        # we take the latest energy in training for L'_N(w^*)

        final_N = epochs[-1] * steps_per_epoch
        optimum_energy_per_sample = popt[0] + 1/final_N * energies[k-min_polygon][-1] # NL'_N(w_0)
        estimated_rlct = popt[1]

        b_values[k] = popt[1]

        predicted = [func(epoch * steps_per_epoch, *popt) for epoch in intermediate_epochs]
        predicted_lfes = [fef(epoch * steps_per_epoch, optimum_energy_per_sample, estimated_rlct) for epoch in intermediate_epochs]

        color = next(colors)
        ax1.plot(epochs, hatlambda_data, 'o', color=color, label=f"{k}-gon")
        ax1.plot(intermediate_epochs, predicted, "--", color=color)
        ax2.plot(epochs, lfe_data, 'o', color=color, label=f"{k}-gon")
        ax2.plot(intermediate_epochs, predicted_lfes, "--", color=color)

        print(f"{k}-gon")
        print(f"  hatlambda slope = {popt[0]:.4f},  lambda = {popt[1]:.4f},     R2 = {r2:.4f},     optimum per-sample energy = {optimum_energy_per_sample:.5f}")
        print(f"  lfe fitted E = {popt_lfe[0]:.6f}, lambda = {popt_lfe[1]:.6f}, R2 = {r2_lfe:.6f}")
        print(f"  avg hatlamda={sum(hatlambda_data)/len(hatlambda_data):.4f}")

    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')
    ax1.set_xticks(epochs)
    ax2.set_xticks(epochs)
    ax1.set_ylabel('Hat lambda')
    ax2.set_ylabel('Free energies')
    ax1.set_xlabel('')
    ax2.set_xlabel("Number of epochs (= N/128)")
    ax1.grid(axis='y', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle(f"Hatlambda against dataset size, m={m}, n={n}, sgld_chains={args.sgld_chains}, truth_gamma={truth_gamma}, single dataset")

    # Create a table of b values
    #ax1.table(cellText=list(b_values.items()), colLabels=['k', 'lambda'], cellLoc = 'center', loc='bottom')

    # Adjust the position of the axes to make room for the table
    #ax1.set_position([0.2, 0.2, 0.5, 0.7])

    plt.show()

if __name__ == "__main__":
    args = parse_commandline().parse_args()
    main(args)