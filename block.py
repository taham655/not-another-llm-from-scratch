from attention import MultiHeadSelfAttention
from rope import RoPE
from swiglu import SwiGLU
from rmsnorm import RMSNorm
import torch.nn as nn 


class Block(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        
        self.d_k = d_model // num_heads
        
        self.rms_1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rms_2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        
        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rope=self.rope, device=device, dtype=dtype)

    def forward(self, x, token_positions=None): 
        residual = x
        
        #diverge to calculate attention
        pre_norm_attn = self.rms_1(residual)
        attention = self.mha(pre_norm_attn, token_positions=token_positions)

        residual = residual + attention

        #diverge to produce a ffn
        pre_norm_ffn = self.rms_2(residual)
        ffn = self.swiglu(pre_norm_ffn)

        return residual + ffn

        



