import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.basic_modules.layers import *

class MemoryGating_class(nn.Module):
    '''
    Input: 
    - prompt embedding
    - adapter embedding
    
    Output:
    - gate
    
    Args:
    - input_dim
    - hidden_dim
    - gate_dim
    
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_experts = len(args.model_list)
        self.num_memory = args.num_memory
        self.memory_dim = args.memory_dim
        # mem
        self.sim = nn.CosineSimilarity(dim = -1)
        self.memMatrix = nn.Parameter(torch.zeros(self.num_memory, self.memory_dim))  # M,C memory
        self.keyMatrix = nn.Parameter(torch.zeros(self.num_memory, self.memory_dim))  # M,C key
        self.x_proj = nn.Linear(self.memory_dim, self.memory_dim)
        
        self.w_gate = nn.Parameter(
            torch.zeros(self.memory_dim, self.n_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(self.memory_dim, self.n_experts), requires_grad=True
        )
        self.softplus = nn.Softplus()
        self.noise_epsilon = 1e-2
        
        self.initialize_weights()
        print("model initialized memory")
    
    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)
        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w_noise, a=math.sqrt(5))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, prop_embed, adap_embed, Type='',shape=None):
        # agg over time dimension
        B, T, N, D = prop_embed.shape
        prop_embed = torch.mean(prop_embed, dim=1) # B,T,N,D -> B,N,D
        # query memory
        assert prop_embed.shape[-1]==self.memMatrix.shape[-1]==self.keyMatrix.shape[-1], "memory dimension mismatch"
        x_query = torch.tanh(self.x_proj(prop_embed))  # easy proj to hidden space [B,N,D]
        att_weight = F.linear(input=x_query, weight=self.keyMatrix)  # [B,N,D] by [M,D]^T --> [B,N,M]
        # noi_label
        clean_logits = self.memMatrix.clone() @ self.w_gate
        raw_noise_stddev = self.memMatrix.clone() @ self.w_noise
        noise_stddev = (
            self.softplus(raw_noise_stddev) + self.noise_epsilon
        ) * self.training
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev) # [M, n_model]
        mem_label = torch.softmax(noisy_logits, dim=-1) # [M, n_model]

        
        # topk
        if self.args.sparse_qk:
            topk = int(math.log(int(self.num_memory)))
            values, indices = torch.topk(att_weight, k = topk, dim = -1)
            att_weight = att_weight.fill_(-float('inf')).scatter_(-1, indices, values)
        
        att_weight = F.softmax(att_weight, dim=-1)  # B,N,M  attn
        mem_retrieved = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [B,N,M] by [M,D]  --> [B,N,D] mixture of retrieved memory for every node
        label_retrived = torch.matmul(att_weight, mem_label) # [B,N,3]
        gate = label_retrived
        return gate, (mem_retrieved, label_retrived, mem_label, None, att_weight)