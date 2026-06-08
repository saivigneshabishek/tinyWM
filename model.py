import torch
import torch.nn as nn
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, dim, inner_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=inner_dim, out_features=dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.mlp(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self,
                dim,
                n_heads,
                dropout,
                use_flash_attn=True,
                causal=False,
                ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim:{self.dim} must be divisible by n_heads:{self.n_heads}")
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn
        self.causal = causal
        self.qkv_proj = nn.Linear(in_features=self.dim, out_features=self.dim*3)
        self.out_proj = nn.Linear(in_features=self.dim, out_features=self.dim)

    def ref_attn(self, q, k, v, dropout, causal):
        """
        inputs : q,k,v -> (b, n_heads, seq_len, head_dim)
        returns: attn -> (b, n_heads, seq_len, head_dim)
        """
        qk = q @ k.transpose(-2, -1) # (b, n_heads, seq_len, head_dim) @ (b, n_heads, head_dim, seq_len) -> (b, n_heads, seq_len, seq_len)
        qk = qk / self.head_dim ** (1/2)

        # mask out future tokens, only attend to past and curr tokens; no cheating :p
        if causal:
            l = qk.shape[-1]
            mask = torch.triu(torch.full((l, l), float("-inf"), device=qk.device, dtype=qk.dtype), diagonal=1)
            qk += mask

        qk = nn.functional.softmax(qk, dim=-1)
        qk = nn.functional.dropout(qk, p=dropout)
        attn = qk @ v # (b, n_heads, seq_len, seq_len) @ (b, n_heads, seq_len, head_dim) -> (b, n_heads, seq_len, head_dim)
        return attn

    def forward(self, x):
        """
        inputs : (b, seq_len, dim)
        outputs: (b, seq_len, dim)
        seq_len: t or l (H*W)
        """
        b, seq_len, dim = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        # (b, n_heads, seq_len, head_dim)
        q = q.view(b, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(b, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(b, seq_len, self.n_heads, self.head_dim).transpose(1,2)

        if self.use_flash_attn:
            attn = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            attn = self.ref_attn(q, k, v, dropout=self.dropout if self.training else 0, causal=self.causal)
        
        attn = attn.transpose(1,2).contiguous() # (b, seq_len, n_heads, head_dim)
        attn = attn.view(b, seq_len, dim)
        out = self.out_proj(attn)
        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self,
                dim,
                n_heads,
                dropout,
                use_flash_attn=True,
                causal=False,
                ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim:{self.dim} must be divisible by n_heads:{self.n_heads}")
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn
        self.causal = causal
        self.q_proj = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_proj = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_proj = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_proj = nn.Linear(in_features=self.dim, out_features=self.dim)

    def ref_attn(self, q, k, v, dropout, causal):
        """
        inputs : q,k,v -> (b, n_heads, seq_len, head_dim)
        returns: attn -> (b, n_heads, seq_len, head_dim)
        """
        qk = q @ k.transpose(-2, -1) # (b, n_heads, seq_len, head_dim) @ (b, n_heads, head_dim, seq_len) -> (b, n_heads, seq_len, seq_len)
        qk = qk / self.head_dim ** (1/2)

        # mask out future tokens, only attend to past and curr tokens; no cheating :p
        if causal:
            q_len, k_len = qk.shape[-2], qk.shape[-1]
            mask = torch.triu(torch.full((q_len, k_len), float("-inf"), device=qk.device, dtype=qk.dtype), diagonal=1)
            qk += mask

        qk = nn.functional.softmax(qk, dim=-1)
        qk = nn.functional.dropout(qk, p=dropout)
        attn = qk @ v # (b, n_heads, seq_len, seq_len) @ (b, n_heads, seq_len, head_dim) -> (b, n_heads, seq_len, head_dim)
        return attn

    def forward(self, query, key=None, val=None):
        """
        inputs - query: (b, seq_len, dim)
        outputs: (b, seq_len, dim)
        seq_len: t or l (H*W)
        """

        if key is None: key = query
        if val is None: val = key
        assert key.shape[1] == val.shape[1], "Key and Value should have same length."

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(val)

        b, seq_len, dim = q.shape
        kv_len = k.shape[1]

        # (b, n_heads, seq_len, head_dim)
        q = q.view(b, seq_len, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(b, kv_len, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(b, kv_len, self.n_heads, self.head_dim).transpose(1,2)

        if self.use_flash_attn:
            attn = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            attn = self.ref_attn(q, k, v, dropout=self.dropout if self.training else 0, causal=self.causal)
        
        attn = attn.transpose(1,2).contiguous() # (b, seq_len, n_heads, head_dim)
        attn = attn.view(b, seq_len, dim)
        out = self.out_proj(attn)
        return out

class AdaLNCrossCondBlock(nn.Module):
    """AdaLNCrossCondBlock"""
    def __init__(self, dim=256, n_heads=4, dropout=0.1, causal=False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim:{self.dim} must be divisible by n_heads:{self.n_heads}")
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = MultiHeadCrossAttention(dim=self.dim, n_heads=self.n_heads, dropout=self.dropout, use_flash_attn=True, causal=causal)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = MLP(dim=self.dim, inner_dim=self.dim*4, dropout=self.dropout)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.dim*6),
        )
        nn.init.constant_(self.adaLN[-1].weight, 0) # adaLN-zero
        nn.init.constant_(self.adaLN[-1].bias, 0) # adaLN-zero

    @staticmethod
    def ada_shift_scale(x, shift, scale):
        return x * (1 + scale) + shift
    
    def forward(self, q, k, v, c):
        '''
        q,k,v: inp, c: condition
        Residual attn block with adaLN conditioning; here action_embeddings are used as cond
        == Ref:
        adaLN: <https://arxiv.org/pdf/1707.06065>
        adaLN-zero: <https://arxiv.org/pdf/2212.09748>
        '''
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)
        q = q + gate_msa* self.attn(query=self.ada_shift_scale(x=self.norm1(q), shift=shift_msa, scale=scale_msa), key=k, val=v)
        q = q + gate_mlp* self.mlp(self.ada_shift_scale(x=self.norm2(q), shift=shift_mlp, scale=scale_mlp))
        return q

class AdaLNCondBlock(nn.Module):
    """AdaLNCondBlock"""
    def __init__(self, dim=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim:{self.dim} must be divisible by n_heads:{self.n_heads}")
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = MultiHeadAttention(dim=self.dim, n_heads=self.n_heads, dropout=self.dropout, use_flash_attn=True, causal=True)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = MLP(dim=self.dim, inner_dim=self.dim*4, dropout=self.dropout)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.dim*6),
        )
        nn.init.constant_(self.adaLN[-1].weight, 0) # adaLN-zero
        nn.init.constant_(self.adaLN[-1].bias, 0) # adaLN-zero

    @staticmethod
    def ada_shift_scale(x, shift, scale):
        return x * (1 + scale) + shift
    
    def forward(self, x, c):
        '''
        x: inp, c: condition
        Residual attn block with adaLN conditioning; here action_embeddings are used as cond
        == Ref:
        adaLN: <https://arxiv.org/pdf/1707.06065>
        adaLN-zero: <https://arxiv.org/pdf/2212.09748>
        '''
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate_msa* self.attn(self.ada_shift_scale(x=self.norm1(x), shift=shift_msa, scale=scale_msa))
        x = x + gate_mlp* self.mlp(self.ada_shift_scale(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp))
        return x

class ResidualBlock(nn.Module):
    """standard AttnBlock with res skip connections"""
    def __init__(self, dim=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim:{self.dim} must be divisible by n_heads:{self.n_heads}")
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = MultiHeadAttention(dim=self.dim, n_heads=self.n_heads, dropout=self.dropout, use_flash_attn=True, causal=False)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = MLP(dim=self.dim, inner_dim=self.dim*4, dropout=self.dropout)
    
    def forward(self, x):
        '''
        LN->Attn(+skip)->LN->MLP(+skip)
        '''
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config["dim"]
        self.patch_size = config["patch_size"]
        self.n_blocks = config["n_blocks"]
        self.dropout = config["dropout"]
        self.n_heads = config["n_heads"]
        self.n_patches = config["height"] // self.patch_size * config["width"] // self.patch_size
        self.patch_emb = nn.Conv2d(in_channels=3, out_channels=self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_ = nn.Parameter(torch.randn(1, 1, self.dim)*0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches+1, self.dim)*0.02)
        self.blocks = nn.Sequential(*[
            ResidualBlock(dim=self.dim, n_heads=self.n_heads, dropout=self.dropout)
            for _ in range(self.n_blocks)
        ])

    def forward(self, frames):
        '''
        inputs : frames (bt, 3, h, w)
        returns: embeds (bt, [CLS]+H*W, dim)
        H: h//patch_size ; W: w//patch_size
        '''
        emb = self.patch_emb(frames)  # (bt, dim, H, W)
        emb = rearrange(emb, "bt dim H W -> bt (H W) dim")
        cls_ = self.cls_.expand(emb.shape[0], -1, -1)
        emb = torch.cat((cls_, emb), dim=1)
        emb = emb + self.pos_emb
        emb = self.blocks(emb)
        return emb

class DynamicsPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config["dim"]
        self.n_blocks = config["n_blocks"]
        self.dropout = config["dropout"]
        self.n_heads = config["n_heads"]
        self.n_pred_frames = config["n_pred_frames"]
        self.temporal_pos_emb = nn.Parameter(torch.randn(1, self.n_pred_frames, self.dim )*0.02)
        self.norm = nn.LayerNorm(self.dim)
        self.blocks = nn.ModuleList([
            AdaLNCondBlock(dim=self.dim, n_heads=self.n_heads, dropout=self.dropout)
            for _ in range(self.n_blocks)
        ])
        self.out_proj = MLP(self.dim, self.dim*4, dropout=self.dropout)

    def forward(self, state_emb, act_emb):
        """
        state_emb  (b t dim)
        act_emb    (b t dim); passed as condition
        """
        _, t, _ = state_emb.shape
        state_emb = state_emb + self.temporal_pos_emb[:,:t]
        state_emb = nn.functional.dropout(state_emb, p=self.dropout if self.training else 0)
        for block in self.blocks:
            state_emb = block(state_emb, act_emb)
        pred_state_emb = self.out_proj(self.norm(state_emb))
        return pred_state_emb

    def rollout_step(self, idx, state_emb, act_emb):
        # placeholder. currently it does the same as fwd function, but it will likely change
        state_emb = state_emb + self.temporal_pos_emb[:,:idx]
        state_emb = nn.functional.dropout(state_emb, p=self.dropout if self.training else 0)
        for block in self.blocks:
            state_emb = block(state_emb, act_emb)
        pred_state_emb = self.out_proj(self.norm(state_emb))
        return pred_state_emb
    
class TinyWorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config["dim"]
        self.encoder = ViTEncoder(config["ViTEncoder"])
        self.predictor = DynamicsPredictor(config["DynamicsPredictor"])
        self.action_emb = nn.Sequential(
                            nn.Embedding(num_embeddings=config["n_actions"], embedding_dim=self.dim//4),
                            nn.SiLU(),
                            nn.Linear(self.dim//4, self.dim//2),
                            nn.SiLU(),
                            nn.Linear(self.dim//2, self.dim),
                        )
        self.state_proj = MLP(self.dim, self.dim*4, dropout=0.1)

    def encode(self, frames, actions, return_patches=False):
        b, t, _, _, _ = frames.shape
        frames = rearrange(frames, "b t c h w -> (b t) c h w")
        frames_emb = self.encoder(frames)
        state_emb = rearrange(frames_emb[:,0], "(b t) dim -> b t dim", b=b, t=t) # [CLS] token
        state_emb = self.state_proj(state_emb)
        act_emb = self.action_emb(actions)
        if return_patches:
            return state_emb, act_emb, frames_emb[:,1:]
        return state_emb, act_emb

    def predict(self, state_emb, action_emb):
        return self.predictor(state_emb, action_emb)

    def rollout_step(self, idx, state_emb, action_emb):
        return self.predictor.rollout_step(idx, state_emb, action_emb)

class TinyWMDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config["dim"]
        self.n_blocks = config["n_blocks"]
        self.n_heads = config["n_heads"]
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim:{self.dim} must be divisible by n_heads:{self.n_heads}")
        
        self.height = config["height"]
        self.width = config["width"]
        self.patch_size = config["patch_size"]
        self.grid_h = self.height // self.patch_size
        self.grid_w = self.width // self.patch_size
        self.n_patches = self.grid_h * self.grid_w
        self.dropout = config["dropout"]
        self.up_channels = config["up_channels"]

        self.latent_proj = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.LayerNorm(self.dim),
                    nn.SiLU(),
                    )
        self.patch_proj = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.LayerNorm(self.dim),
                    nn.SiLU(),
                    )
        self.action_proj = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.LayerNorm(self.dim),
                    nn.SiLU(),
                    )
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "adaln_cross_attn": AdaLNCrossCondBlock(dim=self.dim, n_heads=self.n_heads, dropout=self.dropout, causal=False),
            }) for _ in range(self.n_blocks)])
        
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.up_channels),
            nn.SiLU(),
                    )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(self.up_channels, self.up_channels*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            nn.Conv2d(self.up_channels, self.up_channels*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            nn.Conv2d(self.up_channels, self.up_channels*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
            nn.Conv2d(self.up_channels, 3*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
                    )

    def forward(self, pred_latent, patches, action_emb):
        """
        pred_latent: (b, t, dim)
        patches: (bt, HW, dim)
        action_emb: (b, t, dim)
        """
        b,t,dim = pred_latent.shape
        latent_token = self.latent_proj(pred_latent).unsqueeze(2) # (b t 1 dim)
        latent_token = rearrange(latent_token, "b t 1 dim -> (b t) 1 dim", b=b, t=t, dim=dim) # (bt 1 dim)
        kv = latent_token

        if patches.size(1) != self.n_patches:
            raise ValueError(f"Expected {self.n_patches} patch tokens, got {patches.size(1)}")

        q = self.patch_proj(patches)
        action_cond = self.action_proj(action_emb)
        action_cond = rearrange(action_cond, "b t dim -> (b t) 1 dim", b=b, t=t, dim=dim) # (bt 1 dim)
        cond = action_cond + latent_token

        for block in self.blocks:
            q = block["adaln_cross_attn"](q, kv, kv, cond)

        feature_map = self.token_proj(q).transpose(1, 2).reshape(
            b*t,
            self.up_channels,
            self.grid_h,
            self.grid_w,
                )
        frame = torch.sigmoid(self.to_rgb(feature_map))
        frame = rearrange(frame, "(b t) c h w -> b t c h w", b=b, t=t)
        return frame
