import torch
import torch.nn as nn


class LinearProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class MLPProjector(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(approximate='tanh'),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        elif mlp_type == "silu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.SiLU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """

        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


def FeedForward(dim, mult=4, ffd_type="gelu-ffd"):
    inner_dim = int(dim * mult)
    if ffd_type = "gelu-ffd":
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(approximate='tanh'),
            nn.Linear(inner_dim, dim, bias=False),
        )
    elif ffd_type = "silu-ffd":
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.SiLU(),
            nn.Linear(inner_dim, dim, bias=False),
        )
    else:
        raise ValueError(f"Projector with `{mlp_type = }` is not supported!")
 

class TokenProjector(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=1,
        dim_head=64,
        heads=16,
        num_queries=16,
        output_dim=1024,
        ff_mult=4,
        chunck_size=None,
    ):
        super().__init__()
        self.num_queries = num_queries 
        self.chunck_size = chunck_size
        if chunck_size is not None: 
            num_queries = num_queries * chunck_size

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(dim)
 
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x)
        latents = self.latents.repeat(x.size(0), 1, 1)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)
