import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.vqvae import VQVAE


class TransformerLayer(nn.Module):

    def __init__(self, dim = 768, hidden_dim = 3072):
        super().__init__()
        self.MultiHeadAttention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True, dropout=0.1)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MultiLayerPerceptron = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        ])
        # multi-head attention 
        # add & layer normalization
        # feed forward
        # add & layer normalization
        # ......
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        attentioned, _ = self.MultiHeadAttention(x, x, x, need_weights = False)
        attentioned = self.dropout(attentioned)
        x = x.add(attentioned)
        x = self.LayerNorm1(x)
        x = x + self.MultiLayerPerceptron(x)
        x = self.LayerNorm2(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        #print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)

class BidirectionalTransformer(nn.Module):
    def __init__(self, FLAGS):
        super().__init__()
        # positional embedding
        # transfomer layer
        # read learned codebook 
        # ......
        self.num_tokens = FLAGS.num_tokens
        self.token_embeddings = nn.Embedding(FLAGS.num_embeddings + 2, 768)
        self.positional_embeddings = nn.init.trunc_normal_(nn.Parameter(torch.zeros(self.num_tokens + 1, 768)), 0, 0.02)
        block_list = []
        for _ in range(FLAGS.n_layers):
            block_list.append(TransformerLayer())
        self.blocks = nn.Sequential(*block_list)
        self.Token_Prediction = nn.Sequential(*[
            nn.Linear(in_features=768, out_features=768),
            nn.GELU(),
            nn.LayerNorm(768, eps=1e-12)
        ])
        self.bias = nn.Parameter(torch.zeros(self.num_tokens+1, FLAGS.num_embeddings+2))
        self.layer_norm = nn.LayerNorm(768, eps=1e-12)
        self.drop = nn.Dropout(p=0.1)
        self.apply(weights_init)

    def forward(self, x):
        tokenEmb = self.token_embeddings(x)
        n = tokenEmb.shape[1]
        positionEmb = self.positional_embeddings[:n, :]
        #print(tokenEmb.shape, positionEmb.shape)
        embed = self.drop(self.layer_norm(tokenEmb + positionEmb))
        embed = self.blocks(embed)
        embed = self.Token_Prediction(embed)
        logits = torch.matmul(embed, self.token_embeddings.weight.T) + self.bias

        return logits

class MaskGIT(nn.Module):
    def __init__(self, FLAGS):
        super().__init__()

        self.num_tokens = FLAGS.num_tokens
        self.mask_id = FLAGS.num_embeddings
        self.sos_id = FLAGS.num_embeddings + 1
        self.sample_temperature = FLAGS.sample_temperature
        self.gamma = lambda x: np.cos(x * np.pi / 2)
        self.transformer = BidirectionalTransformer(FLAGS)
        self.vqvae = VQVAE(num_embeddings=FLAGS.num_embeddings,
                           embedding_dim=FLAGS.embedding_dim,
                           lamda=FLAGS.lamda)
    
    @torch.no_grad()
    def encode(self, x):
        #print('during encoding')
        q_z, indices = self.vqvae.vqencoder(x)
        #print(f'indices: f{indices.shape}')
        #print(f'q_z: f{q_z.shape}')
        indices = indices.view(q_z.shape[0], -1)
        #print(f'after_indices: f{indices.shape}')
        return q_z, indices

    @torch.no_grad()
    def decode(self, x):
        return self.vqvae.vqdecode(x)

    def forward(self, x):
        with torch.no_grad():
            _, indices = self.encode(x) # x: [B, 64, 64, 3]
        #indices: [B, 16, 16, 1]
        n = np.ceil(self.gamma(np.random.uniform()) * indices.shape[1])
        topn = torch.rand(indices.shape, device=indices.device).topk(int(n), dim=1).indices
        mask = torch.zeros(indices.shape, dtype = torch.bool, device=indices.device)
        mask.scatter_(dim=1, index=topn, value=True)
        masked = self.mask_id * torch.ones_like(indices, device=indices.device)

        masked_indices = mask * indices + (~mask) * masked
        start = self.sos_id * torch.ones(x.shape[0], 1, dtype=torch.long, device=indices.device)
        masked_indices = torch.cat((start, masked_indices), dim=1)
        target_indices = torch.cat((start, indices), dim=1)
        logits = self.transformer(masked_indices)
        return logits, target_indices
    
    @torch.no_grad()
    def logits2img(self, outputs):
        outputs = outputs.reshape(outputs.shape[0], 16, 16)
        features = self.model.vqvae.vq_layer.embeddings[:, outputs]
        features = features.permute(1, 0, 2, 3)
        # print(features.shape)
        outputs = self.model.vqvae.vqdecoder(features)
        return outputs

    
        

