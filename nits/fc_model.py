import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from nits.layer import *
from nits.resmade import ResidualMADE


class MLP(nn.Module):
    def __init__(self, arch, residual=False):
        super(MLP, self).__init__()
        self.layers = self.build_net(arch, name_str='d', linear_final_layer=True)
        self.residual = residual

    def build_net(self, arch, name_str='', linear_final_layer=True):
        net = nn.ModuleList()
        for i, (a1, a2) in enumerate(zip(arch[:-1], arch[1:])):
            net.append(Linear(a1, a2))

            # add nonlinearities
            if i < len(arch) - 2 or not linear_final_layer:
                net.append(nn.ReLU())

        return net

    def forward(self, x):
        for l in self.layers:
            residual = self.residual and l.weight.shape[0] == l.weight.shape[1]
            x = l(x) + x if residual else l(x)

        return x


class RotationParamModel(nn.Module):
    def __init__(self, arch, nits_model, rotate=True, residual=False):
        super(RotationParamModel, self).__init__()
        self.arch = arch
        self.mlp = MLP(arch, residual=residual)
        self.d = arch[0]
        self.n_params = arch[-1]
        self.nits_model = nits_model

        self.rotate = rotate
        if rotate:
            self.A = nn.Parameter(torch.randn(self.d, self.d))

    def proj(self, x, transpose=False):
        if not self.rotate:
            return x

        Q, R = torch.linalg.qr(self.A)
        P = Q.to(x.device)

        if transpose:
            P = P.T

        return x.mm(P)

    def apply_mask(self, x):
        x = x.clone()

        x_vec = []
        for i in range(self.d):
            tmp = torch.cat([x[:, :i], torch.zeros(len(x), self.d - i, device=x.device)], axis=-1)
            x_vec.append(tmp.unsqueeze(1))

        x = torch.cat(x_vec, axis=1).to(x.device)

        return x.reshape(-1, self.d)

    def forward(self, x):
        n = len(x)

        # rotate x
        x = self.proj(x)

        # obtain parameters
        x_masked = self.apply_mask(x)
        params = self.mlp(x_masked).reshape(n, self.n_params * self.d)

        # compute log likelihood
        grad, _ = self.nits_model.pdf(x, params)
        ll = (grad + 1e-10).log().sum()

        return ll

    def forward_temp(self, x):
        n = len(x)

        # rotate x
        x = self.proj(x)

        # obtain parameters
        x_masked = self.apply_mask(x)
        params = self.mlp(x_masked).reshape(n, self.n_params * self.d)

        # compute log likelihood
        grad, y = self.nits_model.pdf(x, params)
        ll = (grad + 1e-10).log().sum()

        return ll

    def forward_vec(self, x):
        n = len(x)

        # rotate x
        x = self.proj(x)

        # obtain parameters
        x_masked = self.apply_mask(x)
        params = self.mlp(x_masked).reshape(n, self.n_params * self.d)

        # compute log likelihood
        grad, _ = self.nits_model.pdf(x, params)
        ll = (grad + 1e-10).log()

        return ll

    def sample(self, n):
        with torch.no_grad():
            data = torch.zeros((n, d), device=device)

            for i in range(d):
                # rotate x
                x = self.proj(x)

                # obtain parameters
                x_masked = self.apply_mask(x)
                params = self.mlp(x_masked).reshape(n, self.n_params * self.d)

                sample = self.nits_model.sample(1, params)
                data[:, i] = sample[:, i]

            # apply the inverse projection to the data
            data = self.proj(data, transpose=True)

        return data


class Normalizer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Normalizer, self).__init__()
        self.weight_diag = nn.Parameter(torch.zeros(size=(d,)), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(size=(d,)), requires_grad=False)
        self.register_buffer("weights_set", torch.tensor(False).bool())
        self.d = in_features

    def set_weights(self, x, device):
        assert x.device == torch.device('cpu')
        m, v = x.mean(dim=(0,)), x.var(dim=(0,))
        self.bias.data = -m
        self.weight_diag.data = 1 / (v + 1e-8).sqrt()
        self.weights_set.data = torch.tensor(True).bool().to(self.weights_set.device)

    def forward(self, x):
        if not self.weights_set:
            raise Exception("Need to set weights first!")

        return self.weight_diag * x + self.bias


class ResMADEModel(nn.Module):
    def __init__(self, d, nits_model, n_residual_blocks=4, hidden_dim=512,
                 dropout_probability=0., use_batch_norm=False,
                 zero_initialization=True, weight_norm=False,
                 rotate=True, normalizer=None, scarf=False,
                 unique_feature_vals=None,
                 nits_input_dim=[]):
        super(ResMADEModel, self).__init__()
        self.d = nits_input_dim
        self.n_params = nits_model.n_params
        self.nits_model = nits_model
        self.normalizer = normalizer

        if len(nits_input_dim) > 1:
            self.encoder = nn.Sequential(
                nn.Linear(d, d),
                nn.BatchNorm1d(d),
                nn.Linear(d, int(nits_input_dim[0] * d)),
                nn.SiLU(True),
                # nn.Dropout(0.2),
                nn.BatchNorm1d(int(nits_input_dim[0] * d)),
                nn.Linear(int(nits_input_dim[0] * d), int(nits_input_dim[1] * d)),
                nn.SiLU(True),
                # nn.Dropout(0.2)
            )
            self.decoder = nn.Sequential(
                nn.BatchNorm1d(int(nits_input_dim[1] * d)),
                nn.Linear(int(nits_input_dim[1] * d), int(nits_input_dim[0] * d)),
                nn.SiLU(True),
                # nn.Dropout(0.2),
                nn.BatchNorm1d(int(nits_input_dim[0] * d)),
                nn.Linear(int(nits_input_dim[0] * d), d),
                nn.SiLU(True),
                # nn.Dropout(0.2),
                nn.Linear(d, d),
                nn.SiLU(True)
            )

        self.mlp = ResidualMADE(
            input_dim=int(nits_input_dim[-1] * d),
            n_residual_blocks=n_residual_blocks,
            hidden_dim=hidden_dim,
            output_dim_multiplier=nits_model.n_params,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            zero_initialization=zero_initialization,
            weight_norm=weight_norm
        )

        self.rotate = rotate
        if rotate:
            self.A = nn.Parameter(torch.randn(self.d, self.d))

        self.scarf = scarf
        self.unique_feature_vals = unique_feature_vals

    def proj(self, x, transpose=False):
        if not self.rotate:
            return x

        Q, R = torch.linalg.qr(self.A)
        P = Q.to(x.device)

        if transpose:
            P = P.T

        return x.mm(P)

    def forward(self, x, mask_train_phase=False, num_reduce_dims=[]):
        # rotate x
        if mask_train_phase:
            x_aug, mask, x = self.mask_input(x)
            ll, y, recon = self.forward_vec(x, num_reduce_dims)
            with torch.no_grad():
                ll_aug, _, _ = self.forward_vec(x_aug, num_reduce_dims)
            return ll, y, mask, ll_aug, recon
        else:
            mask = torch.ones_like(x)
            ll, y, recon = self.forward_vec(x, num_reduce_dims)
            ll_aug = np.zeros(1)
            return ll, y, mask, ll_aug, recon
        # import matplotlib.pyplot as plt
        # plt.scatter(ll.cpu().detach().numpy(), ll_aug.cpu().detach().numpy())
        # plt.xlabel('ll original')
        # plt.ylabel('ll scarf aug')
        # plt.show()

    def forward_vec(self, x, num_reduce_dims=[]):
        if len(num_reduce_dims):
            x = self.encoder(x)

        x = self.proj(x)

        # obtain parameters
        if self.normalizer is not None:
            x = self.normalizer(x)
        params = self.mlp(x)

        # compute log likelihood
        grad, y = self.nits_model.pdf(x, params)
        ll = (grad + 1e-10).log()
        if len(num_reduce_dims):
            recon = self.decoder(x)
        else:
            recon = None
        return ll, y, recon

    def mask_input(self, x):
        batch_size, num_features = x.size()
        sampled_ind = np.random.randint(1, num_features, (batch_size, int((num_features) * 0.1)))
        mask = torch.ones_like(x)
        scarf_mask = torch.zeros_like(x)
        for i in range(batch_size):
            mask[i, sampled_ind[i, :]] = 0
            if self.scarf:
                for feature_num in sampled_ind[i, :]:
                    scarf_mask[i, feature_num] = np.random.choice(self.unique_feature_vals[feature_num])

        x_sampled = x * mask

        if self.scarf:
            x_sampled += scarf_mask

        return x_sampled, mask, x

    def sample(self, n, device):
        with torch.no_grad():
            data = torch.zeros((n, self.d), device=device)

            for i in range(self.d):
                # rotate x
                #                 x_proj = self.proj(data)

                # obtain parameters
                params = self.mlp(data)

                sample = self.nits_model.sample(1, params)
                data[:, i] = sample[:, i]

            # apply the inverse projection to the data
            data = self.proj(data, transpose=True)

        return data
