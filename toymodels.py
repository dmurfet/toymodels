import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class ToyModelsDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, n, gamma):
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

            gaussian_noise = torch.empty_like(x)
            gaussian_noise.normal_(mean=0,std=1/np.sqrt(gamma))
            y = x + gaussian_noise

            self.data.append(x)
            self.targets.append(y)

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