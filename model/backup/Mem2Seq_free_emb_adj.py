import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def inverse_fourier(query):
    """
       query size: B, N, n_bins * 2
    """
    B, N = query.size(0), query.size(1)
    n_bins = 18
    query = query.view(B, N, n_bins, 2)
    fourier_coeff = torch.stack([query[...,0], - query[...,1]]) * n_bins / 2
    return torch.fft.ifft(fourier_coeff)


def fourier(query):
    n_bins = query.shape[-1]
    f_trans = torch.fft.fft(query)
    fourier_coeff = torch.stack([2 * f_trans.real, -2 * f_trans.imag], dim = -1)
    return fourier_coeff / n_bins

def get_nearest_key(query, key_dict, k = 3, sim = 'cosine', threshold = 0.5):
    """
      query size: B, N, 2 * D
      key size: K, 2 * D
      k: k for topk 
      similarity result = B, N, K --> B, N
    """
    if sim == 'cosine':
        sim_func = nn.CosineSimilarity(dim = -1, eps = 1e-6)
    else:
        raise NotImplementedError
    with torch.no_grad():
        batch_size, num_nodes = query.size(0), query.size(1)
        key_dict = key_dict.view(1,*key_dict.shape)
        k_index = []
        score = []
        for n in range(num_nodes):
            query_n = query[:,[n]] # B, 2 * N
            similarity = sim_func(key_dict, query_n) #B, K
            topk = torch.topk(similarity, k)
            score.append(topk.values)
            k_index.append(topk.indices)
        k_index = torch.stack(k_index, dim = 1).to(query.device)
        k_value = torch.stack(score, dim = 1).to(query.device)
    if k > 1:
        return k_index.squeeze(), k_value.squeeze()
    else:
        return k_index.squeeze()


class gconv(nn.Module):
    def __init__(self, order = 2, hidden_size = 64, support_len = 3):
        super(gconv, self).__init__()
        import pickle
        adj = pickle.load(open('data/adj_mx_METR.pkl','rb'))
        self.adj = [torch.from_numpy(adj[i]) for i in range(len(adj))]
        self.order = order
        self.support_len = support_len
        self.linear = nn.Linear(hidden_size * (order*support_len + 1), hidden_size)
        #self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x, adj, score = None):
        if self.adj[0].device != x.device:
            self.adj = [self.adj[i].to(x.device) for i in range(len(self.adj))]
        adj = self.adj + [adj]
        x0 = x
        out = [x0]
        for a in adj:
            x1 = torch.matmul(a.unsqueeze(0), x0)
            out.append(torch.matmul(score, x1))
            for i in range(1, self.order):
                x2 = torch.matmul(a.unsqueeze(0), x1)
                out.append(torch.matmul(score, x2))
                x1 = x2
        out = self.linear(torch.cat(out, dim = -1))
        #out = self.dropout(out)
        return out


class Mem2Seq(nn.Module):
    def __init__(self, num_nodes, key_dict, values, hops = 3, embedding_dim = 64, output_dim = 1, decay_every = 2000, sequence_length = 18):
        super(Mem2Seq, self).__init__()
        self.encoder = EncoderMemNN(num_nodes, key_dict, values, embedding_dim, hops)
        self.decoder = DecoderMemNN(num_nodes, key_dict, values, embedding_dim, hops = hops, output_dim = 1, sequence_length = sequence_length)
        self.decaying_factor = decay_every
        self.out_linear = nn.Linear(embedding_dim, sequence_length)
        self.global_steps = 0

    def _scheduled_sampling(self):
        return self.decaying_factor / (self.decaying_factor + np.exp(self.global_steps / self.decaying_factor))

    def forward(self, x, target = None):
        encoder_hidden = self.encoder(x)
        self.decoder.load_memory(x)
        decoder_output = self.decoder(encoder_hidden, target, self._scheduled_sampling(), x)
        #decoder_output = self.out_linear(encoder_hidden)
        #decoder_output = decoder_output.permute(0,2,1).unsqueeze(-1)
        return decoder_output
        

class EncoderMemNN(nn.Module):
    '''
    Memory Network Encoder
      arguments:
        - graph_embedding
        - embedding_dim
        - hops
        - dropout
        - unk_mask
      params:
      input:
        - x = (B, T, N, C), C = 2 (speed, tod)
        - inci = (B, T, N, C'), C' = 4 (incident_flag, lng, lat, time_delta(from incident))
      output:
        - o = (B, N, E), E = embedding_dim
    '''
    def __init__(self, num_nodes, key_dict, values, embedding_dim = 64, hops = 3, dropout = 0, unk_mask = False, input_dim = 2, seq_length = 18):
        super(EncoderMemNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.hops = hops
        self.C_inci = nn.ModuleList()
        self.key_dict = key_dict
        self.C_temp = nn.Embedding(288, embedding_dim)
        self.proj = nn.Parameter(1e-3 * torch.ones(seq_length,))
        self.noise = nn.Parameter(1e-3 * torch.ones(seq_length, embedding_dim))
        self.bn = nn.ModuleList([nn.BatchNorm1d(num_nodes) for i in range(self.hops+1)])
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.nodevec2 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.gconv = nn.ModuleList([gconv(2, embedding_dim) for _ in range(self.hops)])
        for hop in range(self.hops+1):
            C_inci = nn.Embedding(len(key_dict)+1, embedding_dim)
            self.C_inci.append(C_inci)

    def forward(self, x):
        B, T, N, C = x.size()
        adj = torch.matmul(self.nodevec1, self.nodevec2.T)
        adj = torch.softmax(torch.relu(adj), dim = -1)
        Temb = self.C_temp((x[...,1].permute(0,2,1) * 288).round().long()) #B, N, T, E
        Temb = torch.matmul(Temb.permute(0,1,3,2), self.proj) #B, N, E
        Nemb = torch.matmul(x[...,0].permute(0,2,1), self.noise)
        #Nemb = torch.randn_like(Temb)
        mem_ref = fourier(x[...,0].permute(0,2,1)) # B, N, T
        matched_idx, k_score = get_nearest_key(mem_ref.view(B, N, -1), self.key_dict) #B, N, D_k
        k_score = torch.softmax(k_score, dim = -1).unsqueeze(-1)
        query = self.bn[0](torch.cat([Temb, Nemb], dim = -1))
        query = F.glu(query, dim = -1)
        for hop in range(self.hops):
            mem_k = self.C_inci[hop]
            mem_v = self.C_inci[hop + 1]
            key = mem_k(matched_idx) # shape B, N, k, E
            key = torch.sum(key*k_score, dim = 2)
            score = torch.matmul(query, key.permute(0,2,1)) / (self.embedding_dim ** 0.5)
            score = torch.softmax(score, dim = -1) # shape: B, N, N
            value = mem_v(matched_idx) # shape: B, N, k, E
            value = torch.sum(value*k_score, dim = 2)
            #out = torch.matmul(score, value) # shape B, N, E
            out = self.gconv[hop](value, adj, score)
            out = self.bn[hop+1](out)
            query = query + out
            #query = self.gconv(query, adj)

        return query


class DecoderMemNN(nn.Module):
    def __init__(self, num_nodes, key_dict, values, embedding_dim = 64, sequence_length = 12, hops = 3, dropout = 0, unk_mask = False, input_dim = 2, output_dim = 1):
        super(DecoderMemNN, self).__init__()
        self.gru = nn.GRUCell(output_dim, embedding_dim)
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.seq_length = sequence_length
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.hops = hops
        #self.graph_embedding = torch.from_numpy(graph_embedding).float()
        self.key_dict = key_dict
        self.linear = nn.Linear(embedding_dim * (hops + 1), embedding_dim)
        self.C_inci = nn.ModuleList()
        self.sentinel = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias = False) for _ in range(self.hops)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(num_nodes) for i in range(self.hops)])
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.nodevec2 = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        self.gconv = nn.ModuleList([gconv(2, embedding_dim) for _ in range(self.hops)])
        #self.gconv = gconv(2, embedding_dim)
        for hop in range(self.hops+1):
            C_inci = nn.Embedding(len(key_dict)+1, embedding_dim)
            self.C_inci.append(C_inci)
        self.out_linear = nn.Linear(embedding_dim, output_dim)

    def load_memory(self, x):
        B, T, N, C = x.size()
        mem_ref = fourier(x[...,0].permute(0,2,1))
        matched_idx, k_score = get_nearest_key(mem_ref.view(B, N, -1), self.key_dict)
        k_score = torch.softmax(k_score, dim = -1).unsqueeze(-1)
        self.memory = []
        for hop in range(self.hops):
            mem_k = self.C_inci[hop]
            mem_v = self.C_inci[hop + 1]
            key = mem_k(matched_idx) # shape B, N, k, E
            key = torch.sum(key*k_score, dim = 2)
            #key = torch.sum(key, dim = 2).squeeze()
            self.memory.append(key)
            value = mem_v(matched_idx)
            value = torch.sum(value*k_score, dim = 2)
            #value = torch.sum(value, dim = 2).squeeze() # shape B, N, k, E
        self.memory.append(value)


    def forward(self, hidden, target, sampling, x):
        """
        hidden: encoder hidden - shape: B, N, E
        """
        if self.training:
            target = target[...,[0]]
        B, N, _ = hidden.size()
        adj = torch.matmul(self.nodevec1, self.nodevec2.T)
        adj = torch.softmax(torch.relu(adj), dim = -1)
        outs = []
        for i in range(self.seq_length):
            if i == 0:
                enc_query = torch.zeros(B, N, self.output_dim).to(x.device)
            #elif self.training and (sampling > np.random.uniform(0,1)):
            #    tgt = target[:,i-1] # shape B, N, 1
            #    enc_query = tgt
            else:
                tgt = outs[-1]
                enc_query = tgt
            embed_q = enc_query
            hidden = self.gru(embed_q.contiguous().view(B * N, -1), hidden.view(B * N, -1))
            hidden = hidden.view(B, N, -1)
            u = hidden
            sentinels = []
            mems = []
            for hop in range(self.hops):
                key = self.memory[hop] # B, N, k, E
                energy = torch.matmul(u, key.permute(0, 2, 1)) / (self.embedding_dim ** 0.5)
                sentinel_energy = torch.sum(u * self.sentinel[hop](key), dim = -1) / (self.embedding_dim ** 0.5)
                sentinels.append(sentinel_energy)
                score = torch.softmax(energy, dim = -1) # B, N, k
                value = self.memory[hop+1] # B, N, k, E
                #o = torch.matmul(score, value)
                o = self.bn[hop](self.gconv[hop](value, adj, score))
                mems.append(o)
                u = u + o
                #u = self.gconv(u, adj)
                #u = self.bn[hop](u)
            sentinels = torch.stack(sentinels, dim = -1)
            sentinel_score = torch.softmax(sentinels, dim = -1)
            mems = torch.stack(mems, dim = -2)
            out = torch.sum(sentinels.unsqueeze(-1) * mems, dim = -2)
            out = self.out_linear(out)
            outs.append(out)
        outs = torch.stack(outs, dim = 1)
        return outs




if __name__ == "__main__":
    import numpy as np
    x = torch.randn(64,18,207,2)
    key_dict = torch.randn(520, 36)
    values = np.zeros((520, 36))
    num_nodes = 207
    model = Mem2Seq(num_nodes, key_dict, values, embedding_dim = 64, sequence_length = 18)
    model.eval()
    print(model(x).shape)