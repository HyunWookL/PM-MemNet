import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fourier(query):
    """
    Function for the fourier transform
      input:
         - query: historical speed (B, N, T)
      output
         - fourier_coeff: fast fourier transform results (B, N, T * 2)
            - Note: convert fft results from complex number into real number
    """
    n_bins = query.shape[-1]
    f_trans = torch.fft.fft(query)
    fourier_coeff = torch.stack([2 * f_trans.real, -2 * f_trans.imag], dim = -1)
    return fourier_coeff / n_bins


def get_nearest_key(query, key_dict, k = 3, sim = 'cosine'):
    """
    Function for the k-NN
      input:
         - query: fourier transform of historical speed (B, N, 2 * T)
         - key_dict: fourier transform of representative pattern (K, 2 * T)
         - k: k for k-Nearest Neighbor 
         - sim: pairwise similarity function, default is CosineSimilarity
      output:
         - k_index: k-Nearest indices for the memory selection
         - k_value: k-Nearest similarity values for the memory selection
      Note:
         - Instead of calculating similarity matrix between query and key_dict
           we utilize lazy evaluation to avoid large memory consumption
         - k-NN operation doesn't need gradient calculation - torch.no_grad()
    """
    if sim == 'cosine':
        sim_func = nn.CosineSimilarity(dim = -1, eps = 1e-6)
    else:
        raise NotImplementedError
    with torch.no_grad():
        B, N = query.size(0), query.size(1)
        key_dict = key_dict.view(1,*key_dict.shape) # 1, K, 2 * T
        k_index = []
        k_score = []
        for n in range(N):
            query_n = query[:,[n]] # B, 2 * N
            similarity = sim_func(key_dict, query_n) #B, K
            topk = torch.topk(similarity, k)
            k_score.append(topk.values)
            k_index.append(topk.indices)
        k_index = torch.stack(k_index, dim = 1).to(query.device)
        k_score = torch.stack(k_score, dim = 1).to(query.device)
    if k > 1:
        return k_index.squeeze(), k_score.squeeze()
    else:
        return k_index.squeeze()


def init_params(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform(p)


class gconv(nn.Module):
    """
    Graph Convolutional Neural Networks
      arguments:
         - order: number of gcn hops
         - support_len: number of adjacency matrices - default: 3 (outflow, inflow, adaptive)
         - hidden_size: hidden size, denoted as H, shared for all layers
      inputs:
         - x: input (B, N, H)
         - supports: support_len length list of adjacency matrices (N, N)
         - score: pattern-wise attention score (B, N, N)
      output (B, N, E)
    """
    def __init__(self, order = 2, hidden_size = 64, support_len = 3):
        super(gconv, self).__init__()
        self.order = order
        self.support_len = support_len
        self.linear = nn.Linear(hidden_size * (order * support_len + 1), hidden_size)

    def forward(self, x, supports, score):
        x0 = x
        out = [x0]
        for adj in supports:
            x1 = torch.matmul(adj, x0)
            out.append(torch.matmul(score, x1))
            for i in range(1, self.order):
                x2 = torch.matmul(adj, x1)
                out.append(torch.matmul(score, x2))
                x1 = x2
        out = self.linear(torch.cat(out, dim = -1))
        return out


class Encoder(nn.Module):
    '''
    PM-MemNet Encoder
      arguments:
         - num_nodes: number of nodes in graph, N
         - sequence_length: total signal length, T
         - embedding_dim: H
         - hops: number of layer stack, L
         - key_dict: representative patterns, (K, 2 * T)
         - t_classes: number of classes for the temporal embedding
      params:
         - proj, noise: projection for the initial dimension matching
         - nodevec1, nodevec2: E1, E2 in paper
      input:
         - x: input signal, (B, N, T, C), C is 2 in default setting.
         - supports: list of adjacency matrices, default length: 2 (inflow, outflow)
      output:
         - out: encoded information, (B, N, H)
    '''
    def __init__(self, num_nodes, key_size, embedding_dim = 64, hops = 3, sequence_length = 18, t_classes = 288):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hops = hops
        self.tmp_embedding = nn.Embedding(t_classes, embedding_dim)
        self.proj = nn.Parameter(torch.ones(sequence_length,))
        self.noise = nn.Parameter(torch.empty(sequence_length, embedding_dim))
        self.nodevec1 = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.nodevec2 = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        
        self.mem_embedding = nn.ModuleList()
        self.bn = nn.ModuleList([nn.BatchNorm1d(num_nodes)])
        self.gconv = nn.ModuleList()
        for hop in range(self.hops):
            self.mem_embedding.append(nn.Embedding(key_size, embedding_dim))
            self.gconv.append(gconv(2, embedding_dim))
            self.bn.append(nn.BatchNorm1d(num_nodes))
        self.mem_embedding.append(nn.Embedding(key_size, embedding_dim))

    def forward(self, x, supports, topk_results): # 이후 수정필요
        adj = torch.matmul(self.nodevec1, self.nodevec2.T)
        adj = torch.softmax(torch.relu(adj), dim = -1)
        supports = supports + [adj]
        Temb = self.tmp_embedding((x[...,1].permute(0,2,1) * 288).round().long()) #B, N, T, H
        Temb = torch.matmul(Temb.permute(0,1,3,2), self.proj) #B, N, T, H --> B, N, H
        Nemb = torch.matmul(x[...,0].permute(0,2,1), self.noise)
        matched_idx, k_score = topk_results #B, N, k
        k_score = torch.softmax(k_score, dim = -1).unsqueeze(-1)
        query = self.bn[0](torch.cat([Temb, Nemb], dim = -1))
        query = F.glu(query, dim = -1)
        for hop in range(self.hops):
            mem_k = self.mem_embedding[hop]
            mem_v = self.mem_embedding[hop+1]
            key = mem_k(matched_idx) # shape B, N, k, H
            key = torch.sum(key * k_score, dim = 2)
            score = torch.matmul(query, key.permute(0,2,1)) / (self.embedding_dim ** 0.5)
            score = torch.softmax(score, dim = -1) # shape: B, N, N
            value = mem_v(matched_idx) # shape: B, N, k, H
            value = torch.sum(value * k_score, dim = 2) # B, N, H
            out = self.gconv[hop](value, supports, score)
            out = self.bn[hop+1](out)
            query = query + out

        return query


class Decoder(nn.Module):
    """
    PM-MemNet Decoder
      arguments:
         - num_nodes: equivalent to encoder
         - seq_length: equivalent to encoder
         - embedding_dim: equivalent to encoder
         - hops: equivalent to encoder
         - key_dict: equivalent to encoder
         - output_dim: output dimension - default: 1 (only speed)
      params:
         - nodevec1, nodevec2: equivalent to encoder
         - sentinel: sentinel weights for layer-wise attention
      input:
         - hidden: last hidden state of encoder, (B, N, H)
         - supports: equivalent to encoder
      output:

    """
    def __init__(self, num_nodes, key_size, embedding_dim = 64, sequence_length = 12, hops = 3, output_dim = 1):
        super(Decoder, self).__init__()
        self.gru = nn.GRUCell(output_dim, embedding_dim)
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.seq_length = sequence_length
        self.hops = hops
        self.gru = nn.GRUCell(output_dim, embedding_dim)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.nodevec2 = nn.Parameter(torch.randn(num_nodes, embedding_dim))

        self.mem_embedding = nn.ModuleList()
        self.sentinel = nn.ModuleList()
        self.bn = nn.ModuleList([nn.BatchNorm1d(num_nodes)])
        self.gconv = nn.ModuleList()
        for hop in range(self.hops):
            self.mem_embedding.append(nn.Embedding(key_size, embedding_dim))
            self.sentinel.append(nn.Linear(embedding_dim, embedding_dim, bias = False))
            self.bn.append(nn.BatchNorm1d(num_nodes))
            self.gconv.append(gconv(2, embedding_dim))
        self.mem_embedding.append(nn.Embedding(key_size, embedding_dim))
        self.out_linear = nn.Linear(embedding_dim, output_dim)
        self.global_steps = 0

    def load_memory(self, topk_results):
        matched_idx, k_score = topk_results
        k_score = torch.softmax(k_score, dim = -1).unsqueeze(-1)
        self.memory = []
        for hop in range(self.hops):
            mem_k = self.mem_embedding[hop]
            mem_v = self.mem_embedding[hop + 1]
            key = mem_k(matched_idx) # shape B, N, k, E
            key = torch.sum(key*k_score, dim = 2) # B, N, E
            self.memory.append(key)
            value = mem_v(matched_idx)
            value = torch.sum(value*k_score, dim = 2) # B, N, E
        self.memory.append(value)


    def forward(self, hidden, supports, target):
        if self.training:
            self.global_steps += 1
        B, N, _ = hidden.size()
        adj = torch.matmul(self.nodevec1, self.nodevec2.T)
        adj = torch.softmax(torch.relu(adj), dim = -1)
        supports = supports + [adj]
        outs = []
        for i in range(self.seq_length):
            if i == 0:
                enc_query = torch.zeros(B, N, self.output_dim).to(hidden.device)
            else:
                tgt = outs[-1]
                enc_query = tgt
            hidden = self.gru(enc_query.contiguous().view(B * N, -1), hidden.view(B * N, -1))
            hidden = hidden.view(B, N, -1)
            u = hidden
            sentinels = []
            mems = []
            for hop in range(self.hops):
                key = self.memory[hop] # B, N, E
                energy = torch.matmul(u, key.permute(0, 2, 1)) / (self.embedding_dim ** 0.5)
                sentinel_energy = torch.sum(u * self.sentinel[hop](key), dim = -1) / (self.embedding_dim ** 0.5)
                sentinels.append(sentinel_energy)
                score = torch.softmax(energy, dim = -1) # B, N, N
                value = self.memory[hop+1] # B, N, E
                o = self.bn[hop](self.gconv[hop](value, supports, score))
                mems.append(o)
                u = u + o
            sentinels = torch.stack(sentinels, dim = -1)
            sentinel_score = torch.softmax(sentinels, dim = -1)
            mems = torch.stack(mems, dim = -2)
            out = torch.sum(sentinels.unsqueeze(-1) * mems, dim = -2)
            out = self.out_linear(out)
            outs.append(out)
        outs = torch.stack(outs, dim = 1)
        return outs


class PMMemNet(nn.Module):
    def __init__(self, num_nodes, key_dict, hops = 3, embedding_dim = 64, output_dim = 1, decay_every = 2000, sequence_length = 18):
        super(PMMemNet, self).__init__()
        self.key_dict = key_dict
        key_size = len(key_dict)
        self.encoder = Encoder(num_nodes, key_size, embedding_dim, hops = hops, sequence_length = sequence_length)
        self.decoder = Decoder(num_nodes, key_size, embedding_dim, hops = hops, output_dim = output_dim, sequence_length = sequence_length)
        self.out_linear = nn.Linear(embedding_dim, sequence_length)

    def forward(self, x, supports, target = None):
        B, T, N, C = x.size()
        mem_ref = fourier(x[...,0].permute(0,2,1)).view(B, N, -1)
        topk_results = get_nearest_key(mem_ref, self.key_dict)
        encoder_hidden = self.encoder(x, supports, topk_results)
        self.decoder.load_memory(topk_results)
        decoder_output = self.decoder(encoder_hidden, supports, target)
        return decoder_output


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    import numpy as np
    x = torch.randn(64,18,207,2)
    x[...,1] = 0
    key_dict = torch.randn(10, 36)
    values = np.zeros((520, 36))
    num_nodes = 207
    supports = [torch.zeros(207,207) for _ in range(2)] 
    model = PM_MemNet(num_nodes, key_dict, embedding_dim = 64, sequence_length = 18)
    model.eval()
    print(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    exit()

    print(model(x, supports).shape)
