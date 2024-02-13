import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq import VectorQuantizer

# 残差连接网络，用于encoder和decoder
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            GNorm(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            GNorm(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        if in_channels != out_channels:
            self.change_shape = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.conv(x) + self.change_shape(x)
        else:
            return x + self.conv(x)

class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        #print('Upsample called')
        x = F.interpolate(x, scale_factor=2.)
        return self.conv(x)


class GNorm(nn.Module):
    def __init__(self, in_channels):
        super(GNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.gn(x))



class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, lamda):
        super().__init__()

        # encoder: define a encoder 
        my_encoder = [128, 'D', 128, 'D', 128, 'D', 256, 'D', 256, 512]
        layers = [nn.Conv2d(3, my_encoder[0], 3, 1, 1)]
        in_channels = my_encoder[0]
        for i in range(1, len(my_encoder)):
            if my_encoder[i] == 'D':
                layers.append(nn.Conv2d(in_channels, in_channels, 3, 2, 1))
                continue
            out_channels = my_encoder[i]
            layers.append(ResidualBlock(in_channels, out_channels))
            layers.append(ResidualBlock(out_channels, out_channels))
            in_channels = out_channels
        layers.append(GNorm(my_encoder[-1]))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(512, 256, 3, 1, 1))
        self.encoder = nn.Sequential(*layers)



        # decoder: define a decoder
        #print(my_encoder)
        my_decoder = my_encoder[::-1]
        #print(my_decoder)
        #layers = []
        layers = [nn.Conv2d(256, my_decoder[0], 3, 1, 1)]# ResidualBlock(my_decoder[0], my_decoder[0])]
        in_channels = my_decoder[0]
        for i in range(1, len(my_decoder)):
            if my_decoder[i] == 'D':
                #layers.append(nn.Conv2d(in_channels, in_channels, 3, 2, 1))
                #print('first downsample')
                layers.append(UpSampleBlock(in_channels))
                continue
            out_channels = my_decoder[i]
            layers.append(ResidualBlock(in_channels, out_channels))
            layers.append(ResidualBlock(out_channels, out_channels))
            in_channels = out_channels
        layers.append(GNorm(my_decoder[-1]))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(my_decoder[-1], 3, 3, 1, 1))
        #print(layers)
        #quit()
        self.decoder = nn.Sequential(*layers)
        #print(self.decoder)
        # define a vector quantization module
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, lamda)

        self.lamda = lamda

    def forward(self, x):
        #print(x.shape)
        z = self.encoder(x)
        q_z, vq_loss, _ = self.vq_layer(z)
        #print(q_z.shape)
        #print(self.decoder)
        x_recon = self.decoder(q_z)

        recon_loss = F.mse_loss(x_recon, x)

        total_loss = recon_loss + self.lamda * vq_loss

        return x_recon, total_loss

    def vqencoder(self, x):
        # encoding input images into quantized latents 
        with torch.no_grad():
            z = self.encoder(x)
            q_z, _, label = self.vq_layer(z)
        return q_z, label 

    def vqdecoder(self, q_z):
        # reconstructing images from quantized latents 
        with torch.no_grad():
            return self.decoder(q_z)
    
    def calculate_lambda(self, nll_loss, g_loss):
        last_layer = self.decoder[-1]
        last_layer_weight = last_layer.weight
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ
