import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    '''
    Reference:
    Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).
    '''
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        # num_embeddings: the number of embeddings of codebook
        # embedding_dim: the dimensions of each embedding
        # beta: the weight of 'embedding loss'
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embeddings = nn.Parameter(torch.randn(embedding_dim, num_embeddings))

    def forward(self, latents: torch.Tensor) :
        # latents: features from encoder

        # Compute L2 distance between latents and embedding weights
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_latents**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings**2, dim=0)
                    - 2 * torch.matmul(flat_latents, self.embeddings))

        # Get the encoding indices that has the min distance
        # Quantize the latents
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings).to(latents.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized_latents = torch.matmul(encodings, self.embeddings.t()).view(*latents_shape)

        # Compute the VQ Losses
            # commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
            # embedding_loss = F.mse_loss(quantized_latents, latents.detach())
            # vq_loss = commitment_loss * self.beta + embedding_loss
        e_latent_loss = F.mse_loss(quantized_latents.detach(), latents)
        q_latent_loss = F.mse_loss(quantized_latents, latents.detach())
        loss = q_latent_loss + self.beta * e_latent_loss

        # Add the residue back to the latents
        # quantized_latents = latents + (quantized_latents - latents).detach()
        quantized_latents = latents + (quantized_latents - latents).detach()

        quantized_latents = quantized_latents.permute(0, 3, 1, 2)
        # return quantized_latents, vq_loss, encoding indices
        return quantized_latents, loss, encoding_indices 
    