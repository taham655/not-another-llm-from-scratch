from embedding import Embedding
from block import Block
from rmsnorm import RMSNorm
from linear import Linear
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, context_length, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()

        self.token_embedding = Embedding(vocab_size=vocab_size,  d_model=d_model, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.lm_head = Linear(input_features=d_model, output_features=vocab_size, device=device, dtype=dtype)

        self.layers = nn.ModuleList([Block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta, device=device, dtype=dtype) for _ in range(num_layers)])

    def forward(self, x, token_positions=None): 
        
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        
        post_norm = self.ln_final(x)

        return self.lm_head(post_norm)