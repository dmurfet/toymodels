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
    def __init__(self, in_features, out_features, init_config=None, noise_scale=1, use_optimal=False, use_bias=True):
        super(ToyModelsNet, self).__init__()

        self.use_bias = use_bias

        if init_config:
            assert in_features >= init_config, "Not enough columns in W"

            # Populate W with a sequence of columns, first being (1,0,...)
            # and then the remainder being rotated by 2 * pi * i / n for i = 0, ... , k - 1
            # and then zeros, where k = init_config
            length = 1
            bias = 0

            # If asked, we use the optimal length for some configurations (see SLT29)
            # NOTE: we only use these optimal configurations for the "top" k-gon
            if use_optimal and in_features == init_config and out_features == 2 and (init_config >= 5) and (init_config <= 7):
                if init_config == 5:
                    length = 1.1138
                    bias = -0.204094
                elif init_config == 6:
                    length = 1.09155
                    bias = -0.271845
                elif init_config == 7:
                    length = np.sqrt( 1 / (1 + 2 * np.cos(2 * np.pi/init_config) ** 2) )
                    
                print(f"Using length {length:.6f}")

            b = torch.ones(in_features) * bias + noise_scale * torch.randn(in_features)
            W = torch.zeros((out_features, in_features))
            W[0, 0] = length

            theta = 2 * np.pi / init_config
            rotation_matrix = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float)
            
            rotated_col = W[:2,0]

            for i in range(1, in_features):
                if i < init_config:
                    rotated_col = torch.matmul(rotation_matrix, rotated_col)
                    W[:2, i] = rotated_col
                    W[:, i] += noise_scale * torch.randn_like(W[:, i])

            self.W = nn.Parameter(W)

            if self.use_bias:
                self.b = nn.Parameter(b)
            else:
                self.b = torch.zeros(in_features)
        else:
            # Random W and b
            self.W = nn.Parameter(noise_scale * torch.randn(out_features, in_features))
            if self.use_bias:
                self.b = nn.Parameter(noise_scale * torch.randn(in_features))
            else:
                self.b = torch.zeros(in_features)

    def forward(self, x):
        # compute ReLU(W^TWx + b)
        #out = torch.matmul(out, x.unsqueeze(-1)) + self.b.unsqueeze(-1)
        out = torch.matmul(self.W.T, self.W)
        out = torch.bmm(out.unsqueeze(0).repeat(x.shape[0], 1, 1), x.unsqueeze(-1)).squeeze(-1)
        out += self.b.unsqueeze(0).repeat(x.shape[0], 1)
        out = torch.relu(out)
        return out