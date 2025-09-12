import torch
import torch.nn as nn
import kornia
import open_clip
import math

from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

from unifolm_wma.utils.common import autocast
from unifolm_wma.utils.utils import count_params
from unifolm_wma.modules.encoders.resampler import reshape_tensor


class AbstractEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):

    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * torch.ones_like(c) * (self.n_classes -
                                                              1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs, ), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""

    def __init__(self,
                 version="google/t5-v1_1-xxl",
                 device="cuda",
                 max_length=77,
                 freeze=True
                 ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length  # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text,
                                        truncation=True,
                                        max_length=self.max_length,
                                        return_length=True,
                                        return_overflowing_tokens=False,
                                        padding="max_length",
                                        return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = ["last", "pooled", "hidden"]

    def __init__(self,
                 version="openai/clip-vit-large-patch14",
                 device="cuda",
                 max_length=77,
                 freeze=True,
                 layer="last",
                 layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text,
                                        truncation=True,
                                        max_length=self.max_length,
                                        return_length=True,
                                        return_overflowing_tokens=False,
                                        padding="max_length",
                                        return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens,
                                   output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class ClipImageEmbedder(nn.Module):

    def __init__(self,
                 model,
                 jit=False,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 antialias=True,
                 ucg_rate=0.):
        super().__init__()
        from clip import load as load_clip
        self.model, _ = load_clip(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean',
                             torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
                             persistent=False)
        self.register_buffer('std',
                             torch.Tensor([0.26862954, 0.26130258,
                                           0.27577711]),
                             persistent=False)
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',
                                   align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # re-normalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x, no_dropout=False):
        # x is assumed to be in range [-1,1]
        out = self.model.encode_image(self.preprocess(x))
        out = out.to(x.dtype)
        if self.ucg_rate > 0. and not no_dropout:
            out = torch.bernoulli(
                (1. - self.ucg_rate) *
                torch.ones(out.shape[0], device=out.device))[:, None] * out
        return out


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self,
                 arch="ViT-H-14",
                 version="laion2b_s32b_b79k",
                 device="cuda",
                 max_length=77,
                 freeze=True,
                 layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(
            text)  ## all clip models use 77 as context length
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting(
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(self,
                 arch="ViT-H-14",
                 version="laion2b_s32b_b79k",
                 device="cuda",
                 max_length=77,
                 freeze=True,
                 layer="pooled",
                 antialias=True,
                 ucg_rate=0.):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device('cpu'),
            pretrained=version,
        )
        del model.transformer
        self.model = model
        # self.mapper = torch.nn.Linear(1280, 1024)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer('mean',
                             torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
                             persistent=False)
        self.register_buffer('std',
                             torch.Tensor([0.26862954, 0.26130258,
                                           0.27577711]),
                             persistent=False)
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',
                                   align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @autocast
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        if self.ucg_rate > 0. and not no_dropout:
            z = torch.bernoulli(
                (1. - self.ucg_rate) *
                torch.ones(z.shape[0], device=z.device))[:, None] * z
        return z

    def encode_with_vision_transformer(self, img):
        img = self.preprocess(img)
        x = self.model.visual(img)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedderV2(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(self,
                 arch="ViT-H-14",
                 version="laion2b_s32b_b79k",
                 device="cuda",
                 freeze=True,
                 layer="pooled",
                 antialias=True):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device('cpu'),
            pretrained=version,
        )
        del model.transformer
        self.model = model
        self.device = device

        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer('mean',
                             torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
                             persistent=False)
        self.register_buffer('std',
                             torch.Tensor([0.26862954, 0.26130258,
                                           0.27577711]),
                             persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',
                                   align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, no_dropout=False):
        ## image: b c h w
        z = self.encode_with_vision_transformer(image)
        return z

    def encode_with_vision_transformer(self, x):
        x = self.preprocess(x)

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.model.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1],
                          self.model.visual.grid_size[0],
                          self.model.visual.patch_size[0],
                          self.model.visual.grid_size[1],
                          self.model.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(
                x.shape[0], self.model.visual.grid_size[0] *
                self.model.visual.grid_size[1], -1)
            x = self.model.visual.patchnorm_pre_ln(x)
            x = self.model.visual.conv1(x)
        else:
            x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1],
                          -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([
            self.model.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x


class FrozenCLIPT5Encoder(AbstractEncoder):

    def __init__(self,
                 clip_version="openai/clip-vit-large-patch14",
                 t5_version="google/t5-v1_1-xl",
                 device="cuda",
                 clip_max_length=77,
                 t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version,
                                               device,
                                               max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version,
                                           device,
                                           max_length=t5_max_length)
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.e-6:.2f} M parameters, "
            f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.e-6:.2f} M params."
        )

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class LinearProjector(nn.Module):

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class MLPProjector(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=True),
                nn.GELU(approximate='tanh'),
                nn.Linear(output_dim, output_dim, bias=True),
            )
        elif mlp_type == "silu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=True),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim, bias=True),
            )
        else:
            raise ValueError(
                f"Projector with `{mlp_type = }` is not supported!")

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

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(
            -2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


def FeedForward(dim, mult=4, ffd_type="gelu-ffd"):
    inner_dim = int(dim * mult)
    if ffd_type == "gelu-ffd":
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(approximate='tanh'),
            nn.Linear(inner_dim, dim, bias=False),
        )
    elif ffd_type == "silu-ffd":
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.SiLU(),
            nn.Linear(inner_dim, dim, bias=False),
        )
    else:
        raise ValueError(f"Projector with `{mlp_type = }` is not supported!")


class SATokenProjector(nn.Module):

    def __init__(self,
                 dim=1024,
                 depth=1,
                 dim_head=64,
                 heads=16,
                 num_queries=16,
                 output_dim=1024,
                 ff_mult=4,
                 chunk_size=None):
        super().__init__()
        self.num_queries = num_queries
        self.chunk_size = chunk_size

        if chunk_size is not None:
            num_queries = num_queries * chunk_size

        self.latents = nn.Parameter(
            torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head,
                                       heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ]))

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)
        return latents
