import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import numpy.linalg as la
from tqdm import tqdm
import os
import pdb

# Simple model over n variables
# Enumerating over 2^n assignments
# Note: forward method does not return log probability
class Simple(nn.Module):
    def __init__(self, n, x=None):
        super().__init__()
        m = 1 << n

        self.n = n
        self.m = m

        W = None
        if x is None:
            W = torch.randn(m - 1)
        else:
            W = torch.zeros(m - 1)

            cnt = 0.0
            count = {}
            for mask in range(0, m):
                count[mask] = 0.0
            for i in range(0, x.shape[0]):
                mask = 0
                for j in range(0, n):
                    mask += x[i, j].item() << j
                count[mask] += 1.0
                if mask != 0:
                    cnt += 1.0
            for mask in range(1, m):
                count[mask] += 1.0 / (m - 1)

            for k in count:
                count[k] /= cnt + 1.0

            for mask in range(1, m):
                W[mask - 1] = torch.log(torch.Tensor([count[mask]]))

        self.W = nn.Parameter(W, requires_grad=True)

    def forward(self, x):
        n = self.n
        y = []
        for i in range(0, n):
            y.append(x[:, i] << i)
        y = torch.sum(torch.stack(y, -1), -1)

        w = nn.functional.softmax(self.W, dim=0)
        w = torch.cat((torch.zeros(1).to(x.device), w), -1)
        y = w[y]

        return y

class SimplePGC(nn.Module):
    def __init__(self, n, partition, x=None):
        super().__init__()

        self.n = n
        self.partition = partition
        self.dpp_size = len(partition)

        dpp_size = self.dpp_size

        B = torch.randn(dpp_size, dpp_size)
        B_norm = torch.norm(B, dim=0)
        for i in range(0, dpp_size):
            B[:,i] /= B_norm[i]
        self.B = nn.Parameter(B, requires_grad=True)

        PCs = []
        for part in self.partition:
            m = len(part)
            if x is not None:
                PCs.append(Simple(m, x[:,part]))
            else:
                PCs.append(Simple(m))
        self.PCs = nn.ModuleList(PCs)

    def forward(self, x):
        n = self.n
        dpp_size = self.dpp_size
        batch_size = x.shape[0]

        p = []
        for i, part in enumerate(self.partition):
            p.append(self.PCs[i](x[:, part]))
        p = torch.stack(p, -1)

        eps = 1e-8
        I = torch.eye(dpp_size).to(x.device)
        L = torch.matmul(torch.transpose(self.B, 0, 1), self.B) + eps * I
        L0 = L.clone()
        L = L.unsqueeze(0).repeat(batch_size, 1, 1)
        L = L * p.unsqueeze(1)
        L[torch.diag_embed(1-p) == 1.0] = 1.0
        y = torch.logdet(L)
        alpha = torch.logdet(L0 + I)
        return y - alpha


class NonSymSimplePGC(nn.Module):
    def __init__(self, n, partition, x=None):
        super().__init__()

        self.n = n
        self.partition = partition
        self.dpp_size = len(partition)

        dpp_size = self.dpp_size

        A = torch.randn(dpp_size, dpp_size)
        B = torch.randn(dpp_size, dpp_size)
        C = torch.randn(dpp_size, dpp_size)

        A_norm = torch.norm(A, dim=0)
        for i in range(0, dpp_size):
            A[:,i] /= A_norm[i]

        B_norm = torch.norm(B, dim=0)
        for i in range(0, dpp_size):
            B[:,i] /= B_norm[i]

        C_norm = torch.norm(C, dim=0)
        for i in range(0, dpp_size):
            C[:,i] /= C_norm[i]

        self.A = nn.Parameter(A, requires_grad=True)
        self.B = nn.Parameter(B, requires_grad=True)
        self.C = nn.Parameter(C, requires_grad=True)

        PCs = []
        for part in self.partition:
            m = len(part)
            if x is not None:
                PCs.append(Simple(m, x[:,part]))
            else:
                PCs.append(Simple(m))
        self.PCs = nn.ModuleList(PCs)

    def forward(self, x):
        n = self.n
        dpp_size = self.dpp_size
        batch_size = x.shape[0]

        p = []
        for i, part in enumerate(self.partition):
            p.append(self.PCs[i](x[:, part]))
        p = torch.stack(p, -1)

        eps = 1e-8
        I = torch.eye(dpp_size).to(x.device)
        # L = torch.matmul(torch.transpose(self.B, 0, 1), self.B) + eps * I
        A = torch.matmul(torch.transpose(self.A, 0, 1), self.A)
        S = torch.matmul(self.B, torch.transpose(self.C, 0, 1)) - torch.matmul(self.C, torch.transpose(self.B, 0, 1))
        L = A + S + eps * I
        L0 = L.clone()
        L = L.unsqueeze(0).repeat(batch_size, 1, 1)
        L = L * p.unsqueeze(1)
        L[torch.diag_embed(1-p) == 1.0] = 1.0
        y = torch.logdet(L)
        alpha = torch.logdet(L0 + I)
        return y - alpha


class FF(nn.Module):
    def __init__(self, d, m):
        super().__init__()

        self.d = d
        self.m = m

        W = torch.randn(m)
        P = torch.randn(m, d)

        self.W = nn.Parameter(W, requires_grad=True)
        self.P = nn.Parameter(P, requires_grad=True)


    def forward(self, x):
        batch_size = x.shape[0]
        d, m = self.d, self.m

        W = torch.log(torch.softmax(self.W, -1).unsqueeze(0))
        P = torch.sigmoid(self.P).unsqueeze(0)
        x = x.unsqueeze(1)

        x = x * P + (1.0 - x) * (1.0 - P)
        x = torch.sum(torch.log(x), -1) + W

        return torch.logsumexp(x, -1)


class LEnsemble(nn.Module):
    def __init__(self, n, B=None):
        super().__init__()
        self.n = n

        if B is None:
            B = torch.randn(n, n)
            B_norm = torch.norm(B, dim=0)
            for i in range(0, n):
                B[:,i] /= B_norm[i]
        self.B = nn.Parameter(B, requires_grad=True)

    def forward(self, x):
        n = self.n
        batch_size = x.shape[0]

        eps = 1e-8
        I = torch.eye(n).to(x.device)
        L = torch.matmul(torch.transpose(self.B, 0, 1), self.B) + eps * I
        L0 = L.clone()
        L = L.unsqueeze(0).repeat(batch_size, 1, 1)

        L[x == 0] = 0.0
        L[x.unsqueeze(1).repeat(1,n,1) == 0] = 0.0
        L[torch.diag_embed(1-x) == 1] = 1.0

        y = torch.logdet(L)
        return y - torch.logdet(L0 + I)


class DPP(nn.Module):
    def __init__(self, n, B=None):
        super().__init__()
        self.n = n
        k = min(2000, n)
        self.k = k

        if B is None:
            B = torch.randn(k, n)
            B_norm = torch.norm(B, dim=0)
            for i in range(0, n):
                B[:,i] /= B_norm[i]
        self.B = nn.Parameter(B, requires_grad=True)

    def forward(self, x):
        n = self.n
        B = self.B
        batch_size = x.shape[0]

        C = torch.matmul(B, torch.transpose(B, 0, 1))
        ys = []
        for i in range(0, batch_size):
            B_ = B[:, x[i,:] == 1.0]
            y_ = torch.logdet(torch.matmul(torch.transpose(B_, 0, 1), B_)).unsqueeze(0)
            ys.append(y_)
        y = torch.cat(ys, -1)

        I = torch.eye(C.shape[0]).to(x.device)
        return y - torch.logdet(C + I)

    def marginal(self, x):
        n = self.n
        B = self.B
        batch_size = x.shape[0]
        
        L = torch.matmul(torch.transpose(B, 0, 1), B)
        I = torch.eye(L.shape[0]).to(x.device)
        K = torch.matmul(L, torch.inverse(L + I))
        ys = []
        for i in range(0, batch_size):
            ys.append(torch.logdet(K[:, x[i] == 1][x[i] == 1, :]).unsqueeze(0))
        y = torch.cat(ys, -1)

        return y



# a sum mixture over m models
# n variables
class Sum(nn.Module):
    def __init__(self, n, models):
        super().__init__()
        self.n = n
        self.m = len(models)

        W = torch.randn(self.m - 1)
        self.W = nn.Parameter(W, requires_grad=True)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        ys = []
        for model in self.models:
            ys.append(model(x))
        y = torch.stack(ys, 0)

        w = torch.cat((self.W, torch.zeros(1).to(x.device)), -1)
        w = nn.functional.softmax(w, dim=0)
        w = torch.log(w)

        y = w.unsqueeze(-1) + y
        y = torch.logsumexp(y, 0)
        if (y != y).any():
            print('Sum: ERROR!')
            pdb.set_trace()
            print('!')

        return y

# a product mixture over m models
# n variables
class Product(nn.Module):
    def __init__(self, n, models):
        super().__init__()
        self.n = n
        self.m = len(models)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        offset = 0
        ys = []
        for model in self.models:
            ys.append(model(x[:, offset:offset+model.n]))
            offset += model.n

        y = torch.stack(ys, 0)
        y = torch.sum(y, 0)

        return y