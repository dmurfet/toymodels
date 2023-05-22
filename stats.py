# Example usage:
#
# python stats.py --m 2 --n 5 --epochs 2000 --max_stat_batches 20 --truth_gamma 2

import torch
import torch.nn as nn
import torch.optim as optim
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
    parser.add_argument("--sgld_chains", help="Number of SGLD chains to average over during posterior estimates", type=int, default=5)
    parser.add_argument("--max_stat_batches", help="When giving polygon stats, range of batches", type=int, default=10)
    parser.add_argument("--gpu", help="Use GPU, off by default", action="store_true")
    parser.add_argument("--truth_gamma", help="Related to std for true distribution", type=int, default=10)
    parser.add_argument("--no_bias", help="Use no bias in the model", action="store_true")
    return parser

def show_lfe_hatlambda_vs_numbatches(args):
    m, n = args.m, args.n
    num_epochs = args.epochs
    truth_gamma = args.truth_gamma # 1/sqrt(truth_gamma) is the std of the true distribution q(y|x)
    steps_per_epoch = 128
    no_bias = args.no_bias
    
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
        polygon_model = ToyModelsNet(n, m, init_config=k, noise_scale=0, use_bias=not no_bias)
        polygon_model.to(device)
        optimizer = optim.SGD(polygon_model.parameters(), lr=lr)

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

        print(f"[{k}-gon] energy per sample: {avg_energy:.6f}, hatlambda: {hatlambda:.6f}")

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
    num_batches = 20 # number of batches to use in computing L_m in lfe
    lr = args.lr
    no_bias = args.no_bias

    print(f"SLT Toy Model stats m={m},n={n}{', No bias' if no_bias else ''}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Device: {device}")
    criterion = nn.MSELoss()

    #epochs = [10000, 12000, 14000, 16000, 18000, 20000]
    epochs = [6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]

    print("Generating datasets:")
    trainsets = []
    for num_epochs in epochs:
        print(f"   num_epochs={num_epochs}")
        trainset = ToyModelsDataset(steps_per_epoch * num_epochs, n, truth_gamma)
        trainsets.append(trainset)

    hatlambdas = []
    energies = []
    lfes = []
    
    for k in range(2, n+1):
        print(f"{k}-gon")
        polygon_model = ToyModelsNet(n, m, init_config=k, noise_scale=0, use_optimal=True, use_bias=not no_bias)
        polygon_model.to(device)
        optimizer = optim.SGD(polygon_model.parameters(), lr=lr)
        
        hatlambdas_per = []
        energies_per = []
        lfes_per = []

        for i, num_epochs in enumerate(epochs):
            trainset = trainsets[i]
            trainloader_batched = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)    
            
            machine = LearningMachine(polygon_model, trainloader_batched, criterion, optimizer, device, truth_gamma, args.sgld_chains)
        
            # N = steps_per_epoch * num_epochs
            energy = machine.compute_energy() # NL'_N            
            lfe = machine.compute_local_free_energy(num_batches=num_batches) # N E_w^\beta[ L'_M ] ~ N E_w^beta[ L'_N ]
            hatlambda = (lfe - energy) / np.log(steps_per_epoch * num_epochs)

            hatlambdas_per.append(hatlambda.cpu().detach().clone())
            energies_per.append(energy.cpu().detach().clone())
            lfes_per.append(lfe.cpu().detach().clone())

            energy_per_sample = energy / (steps_per_epoch * num_epochs)
            print(f"  [{k}-gon/num_epochs {num_epochs}] energy per sample: {energy_per_sample:.6f}, hatlambda: {hatlambda:.6f}")

        hatlambdas.append(hatlambdas_per)
        energies.append(energies_per)
        lfes.append(lfes_per)

    fig = plt.figure(figsize=(20, 25))
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    
    def func(x, a, b):
        return a * x / np.log(x) + b
    
    def func_lin(x, a, b):
        return a * x + b
    
    def fef(N, E, rlct):
        return N * E + rlct * np.log(N)

    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    b_values = {}
    dataset_sizes = [epoch * steps_per_epoch for epoch in epochs]
    intermediate_epochs = np.linspace(min(epochs), max(epochs), 1000)  # 1000 is the number of points
    
    print("\n")
    for k in range(2, n+1):
        # Fit a function a x/logx + b to these data points
        popt, _ = curve_fit(func, dataset_sizes, hatlambdas[k-2])
        popt_lin, _ = curve_fit(func_lin, dataset_sizes, hatlambdas[k-2])

        predicted = [func(dsize, *popt) for dsize in dataset_sizes]
        r2 = r2_score(hatlambdas[k-2], predicted)

        predicted_lin = [func_lin(dsize, *popt_lin) for dsize in dataset_sizes]
        r2_lin = r2_score(hatlambdas[k-2], predicted_lin)

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
        optimum_energy_per_sample = popt[0] + 1/final_N * energies[k-2][-1] # NL'_N(w_0)
        estimated_rlct = popt[1]

        b_values[k] = popt[1]

        predicted = [func(epoch * steps_per_epoch, *popt) for epoch in intermediate_epochs]
        predicted_lfes = [fef(epoch * steps_per_epoch, optimum_energy_per_sample, estimated_rlct) for epoch in intermediate_epochs]

        color = next(colors)
        ax1.plot(epochs, hatlambdas[k-2], 'o', color=color, label=f"{k}-gon")
        ax1.plot(intermediate_epochs, predicted, "--", color=color)
        ax2.plot(epochs, lfes[k-2], 'o', color=color, label=f"{k}-gon")
        ax2.plot(intermediate_epochs, predicted_lfes, "--", color=color)

        print(f"{k}-gon, fitted a = {popt[0]:.6f}, b = {popt[1]:.6f}, R2 = {r2:.6f}, optimum per-sample energy = {optimum_energy_per_sample:.6f}")
        print(f"  linear fitted a = {popt[0]:.6f}, b = {popt[1]:.6f}, R2 = {r2_lin:.6f}")

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

    plt.suptitle(f"Hatlambda against dataset size, m={m}, n={n}, num_batches={num_batches}, truth_gamma={truth_gamma}")

    # Create a table of b values
    #ax1.table(cellText=list(b_values.items()), colLabels=['k', 'lambda'], cellLoc = 'center', loc='bottom')

    # Adjust the position of the axes to make room for the table
    #ax1.set_position([0.2, 0.2, 0.5, 0.7])

    plt.show()

if __name__ == "__main__":
    args = parse_commandline().parse_args()
    main(args)