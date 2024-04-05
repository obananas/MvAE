import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, dims, bn=False):
        super(Encoder, self).__init__()
        models = []
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                models.append(nn.ReLU())
            else:
                models.append(nn.Dropout(p=0.5))
                models.append(nn.Softmax())
        self.models = nn.Sequential(*models)

    def forward(self, X):
        return self.models(X)


class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        models = []
        for i in range(len(dims) - 1):
            models.append(nn.Linear(dims[i], dims[i + 1]))
            if i == len(dims) - 2:
                models.append(nn.Dropout(p=0.5))
                models.append(nn.Sigmoid())
            else:
                models.append(nn.ReLU())
        self.models = nn.Sequential(*models)

    def forward(self, X):
        return self.models(X)


class Clustering(nn.Module):
    def __init__(self, K, d):
        super(Clustering, self).__init__()
        self.weights = nn.Parameter(torch.randn(K, d).cuda(), requires_grad=True)

    def forward(self, comz):
        q1 = 1.0 / (1.0 + (torch.sum(torch.pow(torch.unsqueeze(comz, 1) - self.weights, 2), 2)))
        q = torch.t(torch.t(q1) / torch.sum(q1))
        loss_q = torch.log(q)
        return loss_q, q


class Discriminator(nn.Module):
    def __init__(self, input_dim, feature_dim=64):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


def discriminator_loss(real_latentRepre_out, fake_latentRepre_out, lambda_dis=1):
    real_loss = nn.BCEWithLogitsLoss()(real_latentRepre_out, torch.ones_like(real_latentRepre_out))
    fake_loss = nn.BCEWithLogitsLoss()(fake_latentRepre_out, torch.zeros_like(fake_latentRepre_out))
    return lambda_dis * (real_loss + fake_loss)


class MvAEModel(nn.Module):
    def __init__(self, input_dims, view_num, nc, h_dims=[200, 100], device='cuda'):
        super().__init__()
        self.input_dims = input_dims
        self.view_num = view_num
        self.nc = nc
        self.h_dims = h_dims
        self.device = device
        self.discriminators = nn.ModuleList()
        for v in range(view_num):
            self.discriminators.append((Discriminator(nc).to(self.device)))
        h_dims_reverse = list(reversed(h_dims))
        encoders_private = []
        decoders_private = []

        for v in range(self.view_num):
            encoder_v = Encoder([input_dims[v]] + h_dims + [nc], bn=True)
            encoders_private.append(encoder_v.to(self.device))
            decoder_v = Decoder([nc * 2] + h_dims_reverse + [input_dims[v]])
            decoders_private.append(decoder_v.to(self.device))
        self.encoders_specific = nn.ModuleList(encoders_private)
        self.decoders_specific = nn.ModuleList(decoders_private)

        d_sum = 0
        for d in input_dims:
            d_sum += d
        self.encoder_share = Encoder([d_sum] + h_dims + [nc])
        self.encoder_share.to(self.device)

    def forward_for_hiddens(self, x_list):
        with torch.no_grad():
            x_total = torch.cat(x_list, dim=-1)
            hiddens_share = self.encoder_share(x_total)
            hiddens_specific = []
            for v in range(self.view_num):
                x = x_list[v]
                hiddens_specific_v = self.encoders_specific[v](x)
                hiddens_specific.append(hiddens_specific_v)
            hiddens_list = [hiddens_share] + hiddens_specific
            hiddens = torch.cat(hiddens_list, dim=-1)
            return hiddens

    def discriminators_loss(self, hiddens_specific, i, LAMB_DIS=1):
        discriminate_loss = 0.
        for j in range(self.view_num):
            if j != i:
                real_out = self.discriminators[i](hiddens_specific[i])
                fake_out = self.discriminators[i](hiddens_specific[j])
                discriminate_loss += discriminator_loss(real_out, fake_out, LAMB_DIS)
        return discriminate_loss

    def forward(self, x_list):
        x_total = torch.cat(x_list, dim=-1)
        x_total = x_total.to(self.device)
        hiddens_share = self.encoder_share(x_total)
        recs = []
        hiddens_specific = []
        for v in range(self.view_num):
            x = x_list[v]
            hiddens_specific_v = self.encoders_specific[v](x)
            hiddens_specific.append(hiddens_specific_v)
            hiddens_v = torch.cat((hiddens_share, hiddens_specific_v), dim=-1)
            rec = self.decoders_specific[v](hiddens_v)
            recs.append(rec)
        hiddens_list = [hiddens_share] + hiddens_specific
        hiddens = torch.cat(hiddens_list, dim=-1)
        return hiddens_share, hiddens_specific, hiddens, recs
