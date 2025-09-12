"""
wild mixture of
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import random
import torch
import torch.nn as nn
import copy
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import logging

mainlogger = logging.getLogger('mainlogger')

from functools import partial
from contextlib import contextmanager
from tqdm import tqdm
from einops import rearrange, repeat, reduce
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torchvision.utils import make_grid
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import OmegaConf
from typing import Optional, Sequence, Any, Tuple, Union, List, Dict
from collections.abc import Mapping, Iterable, Callable
from torch import Tensor

from unifolm_wma.utils.utils import instantiate_from_config
from unifolm_wma.utils.ema import LitEma
from unifolm_wma.utils.distributions import DiagonalGaussianDistribution
from unifolm_wma.utils.diffusion import make_beta_schedule, rescale_zero_terminal_snr
from unifolm_wma.utils.basics import disabled_train
from unifolm_wma.utils.common import (extract_into_tensor, noise_like, exists,
                                      default)

from unifolm_wma.models.samplers.ddim import DDIMSampler
from unifolm_wma.models.diffusion_head.common.lr_scheduler import get_scheduler, SelectiveLRScheduler
from unifolm_wma.models.diffusion_head.ema_model import EMAModel
from unifolm_wma.models.diffusion_head.positional_embedding import SinusoidalPosEmb
from unifolm_wma.modules.encoders.condition import MLPProjector
from unifolm_wma.data.normolize import Normalize, Unnormalize

__conditioning_keys__ = {
    'concat': 'c_concat',
    'crossattn': 'c_crossattn',
    'adm': 'y'
}


class DDPM(pl.LightningModule):
    """
    Denoising Diffusion Probabilistic Model (DDPM) LightningModule.
    """

    def __init__(
        self,
        wma_config: OmegaConf,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        loss_type: str = "l2",
        ckpt_path: Optional[str] = None,
        ignore_keys: Optional[Sequence[str]] = [],
        load_only_unet: bool = False,
        monitor: str = None,
        use_ema: bool = True,
        first_stage_key: str = "image",
        image_size: int = 256,
        channels: int = 3,
        log_every_t: int = 100,
        clip_denoised: bool = True,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
        given_betas: Optional[np.ndarray] = None,
        original_elbo_weight: float = 0.0,
        v_posterior: float = 0.0,
        l_simple_weight: float = 1.0,
        conditioning_key: Optional[str] = None,
        parameterization: str = "eps",
        scheduler_config: Optional[Mapping[str, Any]] = None,
        use_positional_encodings: bool = False,
        learn_logvar: bool = False,
        logvar_init: float = 0.0,
        rescale_betas_zero_snr: bool = False,
    ):
        """
        wma_config: Config object used to build the underlying model.
        timesteps: Number of diffusion steps.
        beta_schedule: Schedule type for betas (e.g., 'linear', 'cosine').
        loss_type: Loss type.
        ckpt_path: Optional checkpoint path to restore weights.
        ignore_keys: Keys to ignore when loading the checkpoint.
        load_only_unet: If True, load the backbone into self.model only.
        monitor: Metric key for monitoring.
        use_ema: If True, maintain EMA weights.
        first_stage_key: Key in batch dict for inputs.
        image_size: Image size.
        channels: Number of channels.
        log_every_t: Interval of timesteps to log intermediates during sampling.
        clip_denoised: Clamp x_0 predictions or not.
        linear_start: Linear schedule start.
        linear_end: Linear schedule end.
        cosine_s: Cosine schedule s parameter.
        given_betas: Externally provided betas; overrides schedule if set.
        original_elbo_weight: Weight for VLB term.
        v_posterior: Interpolation weight for posterior variance.
        l_simple_weight: Weight for simple loss term.
        conditioning_key: Conditioning mechanism key (if used by wrapper).
        parameterization: One of {'eps','x0','v'}.
        scheduler_config: Optional LR scheduler config.
        use_positional_encodings: Whether to inject positional encodings.
        learn_logvar: If True, learn per-timestep log-variance.
        logvar_init: Initial value for log-variance.
        rescale_betas_zero_snr: If True, apply zero-SNR rescaling to betas.
        """
        super().__init__()
        assert parameterization in [
            "eps", "x0", "v"
        ], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        mainlogger.info(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.temporal_length = wma_config.params.temporal_length
        self.image_size = image_size
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(wma_config, conditioning_key)
        self.use_ema = use_ema
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(
                f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path,
                                ignore_keys=ignore_keys,
                                only_model=load_only_unet)
        self.register_schedule(given_betas=given_betas,
                               beta_schedule=beta_schedule,
                               timesteps=timesteps,
                               linear_start=linear_start,
                               linear_end=linear_end,
                               cosine_s=cosine_s)

        # For reschedule
        self.given_betas = given_betas
        self.beta_schedule = beta_schedule
        self.timesteps = timesteps
        self.cosine_s = cosine_s
        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init,
                                 size=(self.num_timesteps, ))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self,
                          given_betas: Optional[np.ndarray] = None,
                          beta_schedule: str = "linear",
                          timesteps: int = 1000,
                          linear_start: float = 1e-4,
                          linear_end: float = 2e-2,
                          cosine_s: float = 8e-3) -> None:
        """
        Create and register diffusion buffers (betas, alphas, posteriors, weights).

        Args:
            given_betas: If provided, use these instead of building a schedule.
            beta_schedule: Name of schedule to create if betas not given.
            timesteps: Number of diffusion steps.
            linear_start: Linear schedule start.
            linear_end: Linear schedule end.
            cosine_s: Cosine schedule parameter
        """
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule,
                                       timesteps,
                                       linear_start=linear_start,
                                       linear_end=linear_end,
                                       cosine_s=cosine_s)
        if self.rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[
            0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))

        if self.parameterization != 'v':
            self.register_buffer('sqrt_recip_alphas_cumprod',
                                 to_torch(np.sqrt(1. / alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod',
                                 to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        else:
            self.register_buffer('sqrt_recip_alphas_cumprod',
                                 torch.zeros_like(to_torch(alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod',
                                 torch.zeros_like(to_torch(alphas_cumprod)))

        posterior_variance = (1 - self.v_posterior) * betas * (
            1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # Above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # Below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_variance_clipped',
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(alphas_cumprod_prev) /
                     (1. - alphas_cumprod)))
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) /
                     (1. - alphas_cumprod)))
        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (2 * self.posterior_variance *
                                            to_torch(alphas) *
                                            (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (
                2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(
                self.betas**2 /
                (2 * self.posterior_variance * to_torch(alphas) *
                 (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context: Optional[str] = None) -> Iterable[None]:
        """
        Context manager that temporarily swaps to EMA weights (if enabled).

        Args:
            context: Optional label for logging.
                                                
        """
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                mainlogger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    mainlogger.info(f"{context}: Restored training weights")

    def init_from_ckpt(self,
                       path: str,
                       ignore_keys: Sequence[str] = tuple(),
                       only_model: bool = False) -> None:
        """
        Load a checkpoint, optionally filtering keys or loading only the inner model.

        Args:
            path: Path to checkpoint.
            ignore_keys: State-dict keys (prefix match) to drop.
            only_model: If True, load only into self.model.
        """
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    mainlogger.info(
                        "Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(
            sd,
            strict=False) if not only_model else self.model.load_state_dict(
                sd, strict=False)
        mainlogger.info(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            mainlogger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            mainlogger.info(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start: Tensor,
                        t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute q(x_t | x_0): mean, variance, and log-variance.

        Args:
            x_start: the [N x C x ...] tensor of noiseless inputs..
            t: the number of diffusion steps (minus 1). Here, 0 means one step..

        Returns:
            (mean, variance, log_variance), each shaped like x_start.
        """
        mean = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t,
                                       x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod,
                                           t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor,
                                 noise: Tensor) -> Tensor:
        """
        Predict x_0 from x_t and noise.
        """
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) *
            x_t - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
                                      x_t.shape) * noise)

    def predict_start_from_z_and_v(self, x_t: Tensor, t: Tensor,
                                   v: Tensor) -> Tensor:
        """
        Predict x_0 from z and v (v-parameterization).
        """
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t,
                                x_t.shape) * v)

    def predict_eps_from_z_and_v(self, x_t: Tensor, t: Tensor,
                                 v: Tensor) -> Tensor:
        """
        Predict epsilon from z and v (v-parameterization).
        """
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t,
                                x_t.shape) * x_t)

    def q_posterior(self, x_start: Tensor, x_t: Tensor,
                    t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0): mean and (log-)variance.
        """
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) *
            x_start +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract_into_tensor(self.posterior_variance, t,
                                                 x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: Tensor, t: Tensor,
                        clip_denoised: bool) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predict mean and variance of p(x_{t-1} | x_t) using the model.
        """
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self,
                 x: Tensor,
                 t: Tensor,
                 clip_denoised: bool = True,
                 repeat_noise: bool = False) -> Tensor:
        """
        Draw a single reverse-diffusion sample step: x_{t-1} from x_t.

        Args:
            x: Current noisy sample (B, C, ...).
            t: Current timestep indices (B,).
            clip_denoised: Clamp x_0 predictions or not.
            repeat_noise: Reuse the same noise across the batch.

        Returns:
            Next sample x_{t-1}.
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # No noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1, ) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 *
                                            model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: Sequence[int],
        return_intermediates: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Run the full reverse process starting from Gaussian noise.

        Args:
            shape: Output tensor shape (B, C, ...).
            return_intermediates: If True, also return intermediate frames.

        Returns:
            Final sample, and optionally the intermediate list.
        """
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='Sampling t',
                      total=self.num_timesteps):
            img = self.p_sample(img,
                                torch.full((b, ),
                                           i,
                                           device=device,
                                           dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        return_intermediates: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Convenience wrapper to sample square images of configured size.

        Args:
            batch_size: Number of samples.
            return_intermediates: If True, also return intermediate frames.

        Returns:
            Final sample (and optionally intermediates).
        """
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            return_intermediates=return_intermediates)

    def q_sample(self,
                 x_start: Tensor,
                 t: Tensor,
                 noise: Optional[Tensor] = None) -> Tensor:
        """
        Forward noising step: sample x_t ~ q(x_t | x_0).
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                          t, x_start.shape) * noise)

    def get_v(self, x: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """
        Compute v-target given x and epsilon.
        """
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t,
                                x.shape) * x)

    def get_loss(self,
                 pred: Tensor,
                 target: Tensor,
                 mean: bool = True) -> Tensor:
        """
        Compute training loss between prediction and target.

        Args:
            pred: Model output.
            target: Target tensor.
            mean: If True, reduce to mean.

        Returns:
            Loss tensor (scalar if reduced).
        """
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target,
                                                    pred,
                                                    reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self,
                 x_start: Tensor,
                 t: Tensor,
                 noise: Optional[Tensor] = None
                 ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute the per-step training loss for a batch.

        Args:
            x_start: Clean inputs (B, C, ...).
            t: Timesteps (B,).
            noise: Optional pre-sampled epsilon.

        Returns:
            (loss, log_dict)
        """

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x: Tensor, *args: Any,
                **kwargs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Lightning forward: sample random timesteps and compute losses.

        Args:
            x: Clean batch (B, C, ...).

        Returns:
            (loss, log_dict)
        """
        t = torch.randint(0,
                          self.num_timesteps, (x.shape[0], ),
                          device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch: Mapping[str, Tensor], k: str) -> Tensor:
        """
        Fetch and format the network input from batch.

        Args:
            batch: Batch mapping.
            k: Key for the tensor to use.

        Returns:
            (B, C, ...) float32 contiguous tensor.
        """
        x = batch[k]
        '''
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        '''
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(
            self, batch: Mapping[str,
                                 Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Common train/val step computing loss and logs.
        """
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch: Mapping[str, Tensor],
                      batch_idx: int) -> Tensor:
        """
        PyTorch Lightning training step.
        """
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict,
                      prog_bar=True,
                      logger=True,
                      on_step=True,
                      on_epoch=True)

        self.log("global_step",
                 self.global_step,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs',
                     lr,
                     prog_bar=True,
                     logger=True,
                     on_step=True,
                     on_epoch=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Mapping[str, Tensor],
                        batch_idx: int) -> None:
        """
        PyTorch Lightning validation step with and without EMA.
        """
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {
                key + '_ema': loss_dict_ema[key]
                for key in loss_dict_ema
            }
        self.log_dict(loss_dict_no_ema,
                      prog_bar=False,
                      logger=True,
                      on_step=False,
                      on_epoch=True)
        self.log_dict(loss_dict_ema,
                      prog_bar=False,
                      logger=True,
                      on_step=False,
                      on_epoch=True)

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        """
        Update EMA after each train batch (if enabled).
        """
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples: List[Tensor]) -> Tensor:
        """
        Arrange a list of (B, C, ...) tensors into a grid for logging.

        Args:
            samples: List of tensors at different timesteps.

        Returns:
            Grid image tensor suitable for visualization.
        """
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(
        self,
        batch: Mapping[str, Tensor],
        N: int = 8,
        n_row: int = 2,
        sample: bool = True,
        return_keys: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        """
        Create tensors for image logging: inputs, diffusion row, (optional) samples.

        Args:
            batch: Batch mapping.
            N: Number of examples to visualize.
            n_row: Number of examples for diffusion-row visualization.
            sample: If True, also run reverse diffusion to produce samples.
            return_keys: If provided, filter the returned dict to these keys.

        Returns:
            Dict of image tensors.
        """
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # Get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # Get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N,
                                                   return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Build the optimizer (AdamW) over model parameters (+ logvar if learned).
        """
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """
    Main Class: Latent-diffusion model on top of DDPM (first/cond stages + guidance).
    """

    def __init__(self,
                 first_stage_config: OmegaConf,
                 cond_stage_config: OmegaConf,
                 num_timesteps_cond: int | None = None,
                 cond_stage_key: str = "instruction",
                 cond_stage_trainable: bool = False,
                 cond_stage_forward: str | None = None,
                 conditioning_key: str | None = None,
                 uncond_prob: float = 0.2,
                 uncond_type: str = "empty_seq",
                 scale_factor: str = 1.0,
                 scale_by_std: bool = False,
                 encoder_type: str = "2d",
                 only_model: bool = False,
                 noise_strength: float = 0.0,
                 use_dynamic_rescale: bool = False,
                 base_scale: float = 0.7,
                 turning_step: int = 400,
                 interp_mode: bool = False,
                 fps_condition_type: str = 'fs',
                 perframe_ae: bool = False,
                 logdir: str | None = None,
                 rand_cond_frame: bool = False,
                 en_and_decode_n_samples_a_time: int | None = None,
                 *args,
                 **kwargs):
        """
        Args:
            first_stage_config: OmegaConf config for the first-stage autoencoder.
            cond_stage_config: OmegaConf config for the conditioning encoder/module.
            num_timesteps_cond: Number of condition timesteps used for schedule shortening.
            cond_stage_key: Batch key for conditioning input (e.g., "instruction").
            cond_stage_trainable: Whether the conditioning module is trainable.
            cond_stage_forward: Optional method name to call on cond model instead of default.
            conditioning_key: Conditioning mode (e.g., "crossattn", "concat").
            uncond_prob: Probability to drop/zero the condition for classifier-free guidance.
            uncond_type: Strategy for unconditional condition ("zero_embed" or "empty_seq").
            scale_factor: Fixed latent scale multiplier if not using std-scaling.
            scale_by_std: If True, compute scale as 1/std of latents at first batch.
            encoder_type: "2d" (per-frame) or "3d" (volumetric) first-stage behavior.
            only_model: If True, load only inner model weights when restoring from ckpt.
            noise_strength: Extra offset noise strength for inputs (when > 0).
            use_dynamic_rescale: If True, apply time-dependent rescaling array.
            base_scale: Target base scale used by dynamic rescaling after turning_step.
            turning_step: Steps to transition from 1.0 to base_scale in dynamic rescaling.
            interp_mode: Flag for interpolation-specific behaviors (reserved).
            fps_condition_type: Frame-per-second conditioning mode label.
            perframe_ae: If True, encode/decode one frame at a time.
            logdir: Optional directory for logs.
            rand_cond_frame: If True, randomly select conditioning frames.
            en_and_decode_n_samples_a_time: Optional per-step batch size for (en|de)code loops.
        """

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # For backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, 'crossattn')
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.noise_strength = noise_strength
        self.use_dynamic_rescale = use_dynamic_rescale
        self.interp_mode = interp_mode
        self.fps_condition_type = fps_condition_type
        self.perframe_ae = perframe_ae

        self.logdir = logdir
        self.rand_cond_frame = rand_cond_frame
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        try:
            self.num_downs = len(
                first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        if use_dynamic_rescale:
            scale_arr1 = np.linspace(1.0, base_scale, turning_step)
            scale_arr2 = np.full(self.num_timesteps, base_scale)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer('scale_arr', to_torch(scale_arr))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.clip_denoised = False

        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert (encoder_type in ["2d", "3d"])
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert (uncond_type in ["zero_embed", "empty_seq"])
        self.uncond_type = uncond_type

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self) -> None:
        """
        Build the condition timestep schedule.
        """
        self.cond_ids = torch.full(size=(self.num_timesteps, ),
                                   fill_value=self.num_timesteps - 1,
                                   dtype=torch.long)
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1,
                           self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self,
                             batch: Mapping[str, Any],
                             batch_idx: int,
                             dataloader_idx: int | None = None) -> None:
        """
        Args:
            batch: Current batch mapping.
            batch_idx: Index of the batch within the epoch.
            dataloader_idx: Optional dataloader index in multi-loader setups.
        """
        # Only for very first batch, reset the self.scale_factor
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and \
                not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            mainlogger.info("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            mainlogger.info(
                f"setting self.scale_factor to {self.scale_factor}")
            mainlogger.info("### USING STD-RESCALING ###")
            mainlogger.info(f"std={z.flatten().std()}")

    def register_schedule(self,
                          given_betas: np.ndarray | None = None,
                          beta_schedule: str = "linear",
                          timesteps: int = 1000,
                          linear_start: float = 1e-4,
                          linear_end: float = 2e-2,
                          cosine_s: float = 8e-3) -> None:
        """
        Extend base schedule registration and optionally shorten conditioning schedule.

        Args:
            given_betas: Optional precomputed beta schedule.
            beta_schedule: Name of schedule function ("linear", "cosine", etc.).
            timesteps: Number of diffusion steps.
            linear_start: Linear schedule start beta (if used).
            linear_end: Linear schedule end beta (if used).
            cosine_s: Cosine schedule parameter (if used).
        """
        super().register_schedule(given_betas, beta_schedule, timesteps,
                                  linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config: OmegaConf) -> None:
        """
        Build and freeze the first-stage (autoencoder) model.

        Args:
            config: OmegaConf config describing the first-stage model to instantiate.
        """
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config: OmegaConf) -> None:
        """
        Build the conditioning stage model.

        Args:
            config: OmegaConf config describing the conditioning model to instantiate.

        """
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c: Any) -> Tensor:
        """
        Encode conditioning input into an embedding tensor.

        Args:
            c: Raw conditioning input (tensor, list/dict of strings, tokens, etc.).

        Returns:
            Conditioning embedding as a tensor (shape depends on cond model).
        """
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(
                    self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(
            self,
            encoder_posterior: DiagonalGaussianDistribution | Tensor,
            noise: Tensor | None = None) -> Tensor:
        """
        Convert encoder posterior to latent z and apply scaling.

        Args:
            encoder_posterior: First-stage output; either a Gaussian posterior or a latent tensor.
            noise: Optional noise for sampling if posterior is Gaussian.

        Returns:
            Scaled latent tensor z.
        """
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    @torch.no_grad()
    def encode_first_stage(self, x: Tensor) -> Tensor:
        """
        Encode input frames/images into latent space.

        Args:
            x: Input tensor, either (B, C, ...).

        Returns:
            Latent tensor with shape matched to input.
        """
        if self.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False

        ## Consume more GPU memory but faster
        if not self.perframe_ae:
            encoder_posterior = self.first_stage_model.encode(x)
            results = self.get_first_stage_encoding(encoder_posterior).detach()
        else:  ## Consume less GPU memory but slower
            results = []
            for index in range(x.shape[0]):
                frame_batch = self.first_stage_model.encode(x[index:index +
                                                              1, :, :, :])
                frame_result = self.get_first_stage_encoding(
                    frame_batch).detach()
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b, t=t)

        return results

    def decode_core(self, z: Tensor, **kwargs: Any) -> Tensor:
        """
        Decode latent z back to pixel space (2D or per-frame).

        Args:
            z: Latent tensor (B, C, ...).

        Returns:
            Decoded tensor in pixel space with shape matching the input layout.
        """
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = z.shape
            z = rearrange(z, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False

        if not self.perframe_ae:
            z = 1. / self.scale_factor * z
            results = self.first_stage_model.decode(z, **kwargs)
        else:
            results = []
            for index in range(z.shape[0]):
                frame_z = 1. / self.scale_factor * z[index:index + 1, :, :, :]
                frame_result = self.first_stage_model.decode(frame_z, **kwargs)
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b, t=t)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z: Tensor, **kwargs: Any) -> Tensor:
        """
        Decode latent with no gradient.

        Args:
            z: Latent tensor to decode.
            **kwargs: Extra args for the decoder.

        Returns:
            Decoded tensor in pixel space.

        """
        return self.decode_core(z, **kwargs)

    # Same as above but without decorator
    def differentiable_decode_first_stage(self, z: Tensor,
                                          **kwargs: Any) -> Tensor:
        """
        Decode latent with gradients enabled.

        Args:
            z: Latent tensor to decode.
        Returns:
            ecoded tensor in pixel space.

        """
        return self.decode_core(z, **kwargs)

    @torch.no_grad()
    def get_batch_input(self,
                        batch: Mapping[str, Any],
                        random_uncond: bool,
                        return_first_stage_outputs: bool = False,
                        return_original_cond: bool = False) -> list[Any]:
        """
        Prepare batch: encode inputs to latents and produce conditioning embeddings.

        Args:
            batch: Batch mapping containing first-stage inputs and conditioning.
            random_uncond: If True and `uncond_type` allows, randomly drop conditions.
            return_first_stage_outputs: If True, also decode z to xrec for logging.
            return_original_cond: If True, also return the raw condition object.
        """
        x = super().get_input(batch, self.first_stage_key)

        # Encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)

        # Get instruction condition
        cond = batch[self.cond_stage_key]
        if random_uncond and self.uncond_type == 'empty_seq':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond[i] = ""
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond)
        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device))
        if random_uncond and self.uncond_type == 'zero_embed':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond_emb[i] = torch.zeros_like(cond_emb[i])

        out = [z, cond_emb]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond)

        return out

    def forward(
        self,
        x: Tensor,
        x_action: Tensor,
        x_state: Tensor,
        c: Any,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Args:
            x: Input latent (or pixel) tensor for the primary stream.
            x_action: Action tensor associated with the batch.
            x_state: State tensor associated with the batch.
            c: Conditioning object (tensor/list/dict) consumed by `apply_model`.

        Returns:
            (loss, loss_dict) tuple.
        """

        t = torch.randint(0,
                          self.num_timesteps, (x.shape[0], ),
                          device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)

        return self.p_losses(x, x_action, x_state, c, t, **kwargs)

    def shared_step(self, batch: Mapping[str, Any], random_uncond: bool,
                    **kwargs: Any) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Common train/val step: build inputs, run forward, return loss/logs.

        Args:
            batch: Input batch mapping.
            random_uncond: Whether to apply classifier-free guidance dropout to cond.
            **kwargs: Extra args forwarded to `forward`.

        Returns:
            (loss, loss_dict) tuple.
        """
        x, c = self.get_batch_input(batch, random_uncond=random_uncond)
        loss, loss_dict = self(x, c, **kwargs)

        return loss, loss_dict

    def apply_model(self, x_noisy: Tensor, x_action_noisy: Tensor,
                    x_state_noisy: Tensor, t: Tensor, cond: Any,
                    **kwargs: Any) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """
        Apply inner diffusion model with standardized conditioning dict.

        Args:
            x_noisy: Noisy latent input for the primary stream.
            x_action_noisy: Noisy action tensor aligned with t.
            x_state_noisy: Noisy state tensor aligned with t.
            t: Timestep indices (B,).
            cond: Raw conditioning; will be wrapped into the proper key if not a dict.
            **kwargs: Extra args forwarded to the inner model call.

        Returns:
            Either a single tensor or a tuple of tensors (x, action, state) depending on model.
        """
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon, x_action_recon, x_state_recon = self.model(
            x_noisy, x_action_noisy, x_state_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon, x_action_recon, x_state_recon

    def p_losses(
        self,
        x_start: Tensor,
        x_action_start: Tensor,
        x_state_start: Tensor,
        cond: Any,
        t: Tensor,
        noise: Tensor | None = None,
        action_noise: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute the per-step training losses for latent diffusion.

        Args:
            x_start: Clean primary latent (or pixel) tensor.
            x_action_start: Clean action tensor aligned with x_start.
            x_state_start: Clean state tensor aligned with x_start.
            cond: Conditioning object; may include masks for action/state losses.
            t: Timestep indices (B,).
            noise: Optional epsilon noise for the primary stream (else sampled).
            action_noise: Optional epsilon noise for the action stream (else sampled).
            **kwargs: Extra args forwarded into `apply_model` (and logged if needed).

        Returns:
            (loss, loss_dict)
        """
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(
                noise, lambda: torch.randn_like(x_start) + self.noise_strength
                * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
            action_noise = torch.randn(x_action_start.shape,
                                       device=x_action_start.device)
            action_noise_new = action_noise + self.input_pertub * torch.randn(
                x_action_start.shape, device=x_action_start.device)

            state_noise = torch.randn(x_state_start.shape,
                                      device=x_state_start.device)
            state_noise_new = state_noise + self.input_pertub * torch.randn(
                x_state_start.shape, device=x_state_start.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_action_noisy = self.dp_noise_scheduler_action.add_noise(
            x_action_start, action_noise_new, t[:x_action_start.shape[0]])
        x_state_noisy = self.dp_noise_scheduler_state.add_noise(
            x_state_start, state_noise_new, t[:x_state_start.shape[0]])

        kwargs['x_start'] = x_start
        model_output, model_action_output, model_state_output = self.apply_model(
            x_noisy, x_action_noisy, x_state_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        target_action = action_noise
        target_state = state_noise

        loss_simple = self.get_loss(model_output, target,
                                    mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        loss_action_simple = F.mse_loss(model_action_output,
                                        target_action,
                                        reduction='none')
        action_mask = cond['c_crossattn_action'][-1]
        loss_action_simple *= action_mask
        loss_action_simple = loss_action_simple.type(loss_action_simple.dtype)
        loss_action_simple = reduce(loss_action_simple, 'b ... -> b (...)',
                                    'mean')
        loss_action_simple = loss_action_simple.sum() / action_mask.sum()
        loss_dict.update({f'{prefix}/loss_action_simple': loss_action_simple})

        loss_state_simple = F.mse_loss(model_state_output,
                                       target_state,
                                       reduction='none')
        state_mask = cond['c_crossattn_action'][-2]
        loss_state_simple *= state_mask
        loss_state_simple = loss_state_simple.type(loss_state_simple.dtype)
        loss_state_simple = reduce(loss_state_simple, 'b ... -> b (...)',
                                   'mean')
        loss_state_simple = loss_state_simple.sum() / state_mask.sum()
        loss_dict.update({f'{prefix}/loss_state_simple': loss_state_simple})

        if self.logvar.device is not self.device:
            self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        loss_action = loss_action_simple
        loss_state = loss_state_simple

        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
        loss_vlb = self.get_loss(model_output, target,
                                 mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        loss_dict.update({f'{prefix}/loss_action_vlb': loss_action})
        loss_dict.update({f'{prefix}/loss_state_vlb': loss_state})

        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        loss_dict.update({f'{prefix}/loss_action': loss_action})
        loss_dict.update({f'{prefix}/loss_state': loss_state})

        if cond['c_crossattn_action'][2]:
            return loss + loss_state + loss_action * 0.0, loss_dict
        else:
            return loss + loss_action + loss_state * 0.0, loss_dict

    def training_step(self, batch: Mapping[str, Any],
                      batch_idx: int) -> Tensor:
        """
        Lightning training step: compute loss and log metrics.

        Args:
            batch: Training batch mapping.
            batch_idx: Batch index within current epoch.

        Returns:
            Scalar loss tensor for optimization.
        """
        loss, loss_dict = self.shared_step(
            batch, random_uncond=self.classifier_free_guidance)
        loss_dict.update(
            {'lr': self.trainer.optimizers[0].param_groups[0]['lr']})
        loss_dict.update({
            'lr_action_unet':
            self.trainer.optimizers[0].param_groups[1]['lr']
        })
        # Sync_dist | rank_zero_only
        self.log_dict(loss_dict,
                      prog_bar=True,
                      logger=True,
                      on_step=True,
                      on_epoch=True,
                      sync_dist=False)
        if (batch_idx + 1) % self.log_every_t == 0:
            mainlogger.info(
                f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}"
            )
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Mapping[str, Any],
                        batch_idx: int) -> None:
        """
        Lightning validation step: compute loss and log metrics.

        Args:
            batch: Validation batch mapping.
            batch_idx: Batch index in validation loop.
        """
        _, loss_dict_no_ema = self.shared_step(batch, random_uncond=False)
        self.log_dict(loss_dict_no_ema,
                      prog_bar=False,
                      logger=True,
                      on_step=False,
                      on_epoch=True)

    def _get_denoise_row_from_list(self,
                                   samples: Sequence[Tensor],
                                   desc: str = '') -> Tensor:
        """
        Decode a list of latents and pack into a grid for visualization.

        Args:
            samples: Sequence of latent tensors to decode and tile.
            desc: Optional tqdm description string.

        Returns:
            Grid image tensor suitable for logging.

        """
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)

        if denoise_row.dim() == 5:
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, 'n b c t h w -> b n c t h w')
            denoise_grid = rearrange(denoise_grid,
                                     'b n c t h w -> (b n) c t h w')
            denoise_grid = rearrange(denoise_grid, 'n c t h w -> (n t) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid

    @torch.no_grad()
    def log_images(self,
                   batch: Mapping[str, Any],
                   sample: bool = True,
                   ddim_steps: int = 200,
                   ddim_eta: float = 1.0,
                   plot_denoise_rows: bool = False,
                   unconditional_guidance_scale: float = 1.0,
                   **kwargs: Any) -> dict[str, Tensor]:
        """ Log images for LatentDiffusion """
        # Control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 2
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        # TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, xrec, xc = self.get_batch_input(batch,
                                              random_uncond=False,
                                              return_first_stage_outputs=True,
                                              return_original_cond=True,
                                              logging=True)

        N = xrec.shape[0]
        log["reconst"] = xrec
        log["condition"] = xc

        if sample:
            uc = None
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                    x0=z,
                    **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def p_mean_variance(
        self,
        x: Tensor,
        c: Any,
        t: Tensor,
        clip_denoised: bool,
        return_x0: bool = False,
        score_corrector: Any = None,
        corrector_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Predict posterior parameters (and optionally x0) at timestep t.

        Args:
            x: Current latent at timestep t.
            c: Conditioning object passed to the inner model/score corrector.
            t: Timestep indices (B,).
            clip_denoised: If True, clamp predicted x0 to [-1, 1].
            return_x0: If True, also return predicted x0.
            score_corrector: Optional score-corrector object with `modify_score`.
            corrector_kwargs: Extra kwargs for the score corrector.
            **kwargs: Forwarded to `apply_model`.

        Returns:
            (mean, var, log_var) or (mean, var, log_var, x0) tensors.
        """

        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c,
                                                     **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self,
                 x: Tensor,
                 c: Any,
                 t: Tensor,
                 clip_denoised: bool = False,
                 repeat_noise: bool = False,
                 return_x0: bool = False,
                 temperature: float = 1.0,
                 noise_dropout: float = 0.0,
                 score_corrector: Any = None,
                 corrector_kwargs: Mapping[str, Any] | None = None,
                 **kwargs: Any) -> Tensor | tuple[Tensor, Tensor]:
        """
        Draw a single reverse-diffusion step (optionally return x0).

        Args:
            x: Current latent at timestep t.
            c: Conditioning object for the model.
            t: Timestep indices (B,).
            clip_denoised: Clamp predicted x0 to [-1, 1] when forming the mean.
            repeat_noise: If True, reuse the same noise across batch.
            return_x0: If True, also return the predicted x0.
            temperature: Temperature for sampling noise scale.
            noise_dropout: Dropout probability applied to the sampled noise.
            score_corrector: Optional score-corrector to adjust model outputs.
            corrector_kwargs: Extra kwargs for the corrector.
            **kwargs: Forwarded to `p_mean_variance`.

        Returns:
            Next latent (and optionally x0).
        """

        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_x0=return_x0, \
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, **kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # No noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1, ) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (
                0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (
                0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self,
                      cond: Any,
                      shape: Sequence[int],
                      return_intermediates: bool = False,
                      x_T: Tensor | None = None,
                      verbose: bool = True,
                      callback: Callable[[int], Any] | None = None,
                      timesteps: int | None = None,
                      mask: Tensor | None = None,
                      x0: Tensor | None = None,
                      img_callback: Callable[[Tensor, int], Any] | None = None,
                      start_T: int | None = None,
                      log_every_t: int | None = None,
                      **kwargs: Any) -> Tensor | tuple[Tensor, list[Tensor]]:
        """
        Run the full reverse process from noise to sample(s).

        Args:
            cond: Conditioning object (tensor/dict/list), optionally noised when cond schedule is shortened.
            shape: Output latent shape (B, C, ...).
            return_intermediates: If True, also return intermediate latents.
            x_T: Optional starting noise latent (else sampled from N(0, I)).
            verbose: If True, show tqdm progress.
            callback: Optional function called with the current timestep i.
            timesteps: Number of reverse steps to perform (default: self.num_timesteps).
            mask: Optional inpainting mask; ones keep original x0 regions.
            x0: Optional original latent for masked regions (when using `mask`).
            img_callback: Optional function called with (img, i) every step.
            start_T: Optional cap to limit starting step (min(timesteps, start_T)).
            log_every_t: Logging frequency for collecting intermediates (defaults to self.log_every_t).

        Returns:
            Final latent sample (and optionally the list of intermediates).
        """

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        # Sample an initial noise
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(
            reversed(range(0, timesteps)), desc='Sampling t',
            total=timesteps) if verbose else reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]

        for i in iterator:
            ts = torch.full((b, ), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond,
                                     t=tc,
                                     noise=torch.randn_like(cond))

            img = self.p_sample(img,
                                cond,
                                ts,
                                clip_denoised=self.clip_denoised,
                                **kwargs)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self,
               cond,
               batch_size: int = 16,
               return_intermediates: bool = False,
               x_T: Tensor | None = None,
               verbose: bool = True,
               timesteps: int | None = None,
               mask: Tensor | None = None,
               x0: Tensor | None = None,
               shape: Sequence[int] | None = None,
               **kwargs: Any) -> Tensor | tuple[Tensor, list[Tensor]]:
        """
        Convenience wrapper to run `p_sample_loop` with a full batch.

        Args:
            cond: Conditioning object; dict/list items are truncated to batch_size.
            batch_size: Number of samples to generate.
            return_intermediates: If True, return intermediates as well.
            x_T: Optional starting noise latent (else sampled).
            verbose: Whether to print sampling progress.
            timesteps: Number of reverse steps (default: self.num_timesteps).
            mask: Optional mask for partial generation/inpainting.
            x0: Optional original latent used with `mask` during sampling.
            shape: Optional output shape; if None, uses (B, C, T, H, W) from model config.

        Returns:
            Final latent (and optionally intermediates).
        """
        if shape is None:
            shape = (batch_size, self.channels, self.temporal_length,
                     *self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key:
                    cond[key][:batch_size] if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(
                    cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates,
                                  x_T=x_T,
                                  verbose=verbose,
                                  timesteps=timesteps,
                                  mask=mask,
                                  x0=x0,
                                  **kwargs)

    @torch.no_grad()
    def sample_log(self, cond: Any, batch_size: int, ddim: bool,
                   ddim_steps: int,
                   **kwargs: Any) -> tuple[Any, Any, Any, Any]:
        """
        Produce samples (and intermediates), optionally via DDIM sampler.

        Args:
            cond: Conditioning object passed to the sampler.
            batch_size: Number of samples to generate.
            ddim: If True, use DDIM sampler; otherwise use ancestral sampling.
            ddim_steps: Number of DDIM steps when `ddim` is True.

        """
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.temporal_length, *self.image_size)
            samples, actions, states, intermediates = ddim_sampler.sample(
                ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond,
                                                 batch_size=batch_size,
                                                 return_intermediates=True,
                                                 **kwargs)

        return samples, actions, states, intermediates

    def configure_schedulers(
            self, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """
        Build LR scheduler dict compatible with PyTorch Lightning.

        Args:
            optimizer: Optimizer instance for which to build the scheduler dict.

        Returns:
            Dict with keys {'scheduler', 'interval', 'frequency'} per Lightning API.
        """
        assert 'target' in self.scheduler_config
        scheduler_name = self.scheduler_config.target.split('.')[-1]
        interval = self.scheduler_config.interval
        frequency = self.scheduler_config.frequency
        if scheduler_name == "LambdaLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {
                'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                'interval': interval,
                'frequency': frequency
            }
        elif scheduler_name == "CosineAnnealingLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {
                'scheduler':
                CosineAnnealingLR(optimizer,
                                  T_max=decay_steps,
                                  last_epoch=last_step),
                'interval':
                interval,
                'frequency':
                frequency
            }
        else:
            raise NotImplementedError
        return lr_scheduler


class LatentVisualDiffusion(LatentDiffusion):
    """
    Visual-conditioned latent diffusion with action/state heads and schedulers.

    """

    def __init__(self,
                 img_cond_stage_config: OmegaConf,
                 image_proj_stage_config: OmegaConf,
                 noise_scheduler_config: OmegaConf,
                 dp_optimizer_config: OmegaConf,
                 dp_ema_config: OmegaConf,
                 freeze_embedder: bool = True,
                 image_proj_model_trainable: bool = True,
                 n_obs_steps_imagen: int = 2,
                 n_obs_steps_acting: int = 2,
                 agent_state_dim: int = 14,
                 agent_action_dim: int = 14,
                 global_emb_dim: int = 1024,
                 input_pertub: float = 0.1,
                 lr_scheduler: str = 'cosine',
                 lr_warmup_steps: int = 500,
                 num_epochs: int = 15000,
                 gradient_accumulate_every: int = 1,
                 use_scheduler: bool = False,
                 dp_use_ema: bool = False,
                 pretrained_checkpoint: str | None = None,
                 decision_making_only: bool = True,
                 *args,
                 **kwargs):
        """
        Args:
            img_cond_stage_config: OmegaConf for the *image* conditioning encoder.
            image_proj_stage_config: OmegaConf for the image feature projector.
            noise_scheduler_config: OmegaConf for DP noise schedulers (state/action).
            dp_optimizer_config: OmegaConf for optimizer params of the UNet heads.
            dp_ema_config: Optional EMA config for the action UNet.
            freeze_embedder: If True, freeze the image embedder params.
            image_proj_model_trainable: If True, train the image projector.
            n_obs_steps_imagen: Number of observed steps for image conditions.
            n_obs_steps_acting: Number of observed steps for acting head.
            agent_state_dim: Dimension of agent state vector.
            agent_action_dim: Dimension of agent action vector.
            global_emb_dim: Embedding size for state/action/text/image fusion.
            input_pertub: Perturbation scale added to action/state noises.
            lr_scheduler: Name of LR scheduler (for SelectiveLRScheduler wrapper).
            lr_warmup_steps: Warmup steps for scheduler creation.
            num_epochs: Total training epochs.
            gradient_accumulate_every: Gradient accumulation steps.
            use_scheduler: If True, enable LR scheduling.
            dp_use_ema: If True, maintain EMA for action UNet head.
            pretrained_checkpoint: Optional path to a pretrained checkpoint.
            decision_making_only: If True, use decision-only augmentation path.
        """

        super().__init__(*args, **kwargs)
        self.image_proj_model_trainable = image_proj_model_trainable
        self.agent_state_dim = agent_state_dim
        self.agent_action_dim = agent_action_dim
        self.global_emb_dim = global_emb_dim
        self.n_obs_steps_imagen = n_obs_steps_imagen
        self.n_obs_steps_acting = n_obs_steps_acting
        self.decision_making_only = decision_making_only

        self._init_embedder(img_cond_stage_config, freeze_embedder)
        self._init_img_ctx_projector(image_proj_stage_config,
                                     image_proj_model_trainable)
        self._init_dp_noise_scheduler(noise_scheduler_config)
        self._init_projectors()
        if dp_use_ema:
            self._init_dp_ema(dp_ema_config)

        # Create a pos_embedder for state and action info, our state and action have an unified vector space
        self.pos_embedder = SinusoidalPosEmb(self.global_emb_dim)
        self.register_buffer('cond_pos_emb',
                             self.pos_embedder(torch.arange(
                                 0, 16)))  #NOTE HAND-CODE 16

        self.input_pertub = input_pertub
        self.dp_optimizer_config = dp_optimizer_config
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.num_epochs = num_epochs
        self.gradient_accumulate_every = gradient_accumulate_every
        self.use_scheduler = use_scheduler
        self.dp_use_ema = dp_use_ema
        self.pretrained_checkpoint = pretrained_checkpoint

    def _init_img_ctx_projector(self, config: OmegaConf,
                                trainable: bool) -> None:
        """
        Instantiate image context projector; optionally freeze.

        Args:
            config: OmegaConf for the projector module to instantiate.
            trainable: If False, freeze the projector.
        """
        self.image_proj_model = instantiate_from_config(config)
        if not trainable:
            self.image_proj_model.eval()
            self.image_proj_model.train = disabled_train
            for param in self.image_proj_model.parameters():
                param.requires_grad = False

    def _init_embedder(self, config: OmegaConf, freeze: bool = True) -> None:
        """
        Instantiate the image embedder; optionally freeze.

        Args:
            config: OmegaConf for the embedder to instantiate.
            freeze: If True, set to eval/disable grads.
        """
        self.embedder = instantiate_from_config(config)
        if freeze:
            self.embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False

    def init_normalizers(self, normalize_config: OmegaConf,
                         dataset_stats: Mapping[str, Any]) -> None:
        """
        Create normalization and unnormalization utilities.

        Args:
            normalize_config: Config with shapes and normalization modes.
            dataset_stats: Statistics dict used to compute normalization.
        """
        self.normalize_inputs = Normalize(
            normalize_config.input_shapes,
            normalize_config.input_normalization_modes, dataset_stats)
        self.unnormalize_outputs = Unnormalize(
            normalize_config.output_shapes,
            normalize_config.output_normalization_modes, dataset_stats)

    def _init_dp_noise_scheduler(self, config: OmegaConf) -> None:
        """
        Instantiate separate DP noise schedulers for action and state.

        Args:
            config: OmegaConf used to create scheduler instances.
        """
        self.dp_noise_scheduler_action = instantiate_from_config(config)
        self.dp_noise_scheduler_state = instantiate_from_config(config)

    def _init_dp_ema(self, config: OmegaConf | None) -> None:
        """
        Initialize EMA for UNet head.

        Args:
            config: EMA config, must contain 'params' sub-dict.
        """
        self.dp_ema_model = copy.deepcopy(
            self.model.diffusion_model.action_unet)
        self.dp_ema_model_on_device = False
        self.dp_ema = EMAModel(**config['params'], model=self.dp_ema_model)

    def _init_projectors(self):
        """
        Build small MLP projectors and positional embeddings for state/action.
        """
        self.state_projector = MLPProjector(self.agent_state_dim,
                                            1024)  # NOTE HAND CODE
        self.action_projector = MLPProjector(self.agent_action_dim,
                                             1024)  # NOTE HAND CODE

        self.agent_action_pos_emb = nn.Parameter(
            torch.randn(1, 16, self.global_emb_dim))
        self.agent_state_pos_emb = nn.Parameter(
            torch.randn(1, self.n_obs_steps_imagen, self.global_emb_dim))

    def _get_augmented_batch(
            self,
            z: Tensor,
            state: Tensor,
            obs_state: Tensor,
            action: Tensor,
            ins: Tensor,
            null_ins: Tensor,
            img: Tensor,
            sim_mode: bool = False,
            pre_action: Tensor | None = None,
            logging: bool = False) -> tuple[Tensor, Tensor, list[Tensor]]:
        """
        Construct augmented conditioning batch for decision/simulation modes.

        Args:
            z: Latent video tensor (B, C, ...).
            state: Full state tensor (B, T, D_s).
            obs_state: Observed state embeddings (B, T, E).
            action: Action embeddings (B, T, E).
            ins: Instruction/text embeddings (B, L, E) after projector.
            null_ins: Null/empty instruction embedding for CFG.
            img: Image conditioning embedding (B, E_img) or batched equivalent.
            sim_mode: If True, build simulated-mode batch; else decision-making.
            pre_action: Optional previous action(s). (unused here; reserved)
            logging: If True, may include extra returns for logs. (unused)

        Returns:
            Tuple of (z, state, [mode_batch]) where mode_batch is a single tensor combining the selected conditioning streams.
        """

        b, _, t, _, _ = z.shape
        if self.decision_making_only:
            mode_batch = torch.cat([obs_state, ins, img], dim=1)
            return z, state, [mode_batch]

        if not sim_mode:
            zero_action = torch.zeros_like(action)
            mode_batch = torch.cat([obs_state, zero_action, ins, img], dim=1)
        else:
            null_ins_batch = null_ins.repeat_interleave(repeats=ins.shape[0],
                                                        dim=0)
            mode_batch = torch.cat([obs_state, action, null_ins_batch, img],
                                   dim=1)
        return z, state, [mode_batch]

    def on_train_batch_end(self, outputs: Any, batch: Mapping[str, Any],
                           batch_idx: int) -> None:
        """
        Update EMA for action UNet after each train batch (if enabled).

        Args:
            batch: Current training batch mapping.
            batch_idx: Batch index within the epoch.
        """
        if self.dp_use_ema:
            if self.dp_ema_model is not None and not self.dp_ema_model_on_device:
                device = self.model.device
                self.dp_ema_model.to(device)
                self.dp_ema_model_on_device = True
            self.dp_ema.step(self.model.diffusion_model.action_unet)

    def shared_step(self, batch: Mapping[str, Any], random_uncond: bool,
                    **kwargs: Any) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Common train/val step for visual diffusion.

        Args:
            batch: Input batch mapping.
            random_uncond: Whether to apply classifier-free guidance dropout.

        Returns:
            (loss, loss_dict) tuple.
        """
        x, x_action, x_state, c, fs = self.get_batch_input(
            batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, x_action, x_state, c, **kwargs)
        return loss, loss_dict

    def get_batch_input(self,
                        batch: Mapping[str, Any],
                        random_uncond: bool,
                        return_first_stage_outputs: bool = False,
                        return_original_cond: bool = False,
                        return_fs: bool = False,
                        return_cond_frame: bool = False,
                        return_original_input: bool = False,
                        logging: bool = False,
                        **kwargs: Any) -> list[Any]:
        """
        Prepare model inputs & conditioning from a raw training batch.

        Args:
            batch: Batch mapping with keys like image/state/action/obs/etc.
            random_uncond: Apply stochastic condition dropout for CFG.
            return_first_stage_outputs: If True, also return xrec (decoded z).
            return_original_cond: If True, also return raw instruction text.
            return_fs: If True, return fps or frame_stride per config.
            return_cond_frame: If True, return conditioning frames (obs images).
            return_original_input: If True, return original x (pre-encoding).
            logging: If True, append sim_mode flag at the end.

        Returns:
            A list of inputs
        """
        # x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        b, _, t, _, _ = x.shape
        # Get actions: b t d
        action = super().get_input(batch, 'action')
        # Get states: b t d
        state = super().get_input(batch, 'next.state')
        # Get observable states: b t d
        obs_state = super().get_input(batch, 'observation.state')
        # Get observable images: b c t h w
        obs = super().get_input(batch, 'observation.image')

        # Encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)

        cond = {}
        # Get instruction condition
        cond_ins_input = batch[self.cond_stage_key]
        if isinstance(cond_ins_input, dict) or isinstance(
                cond_ins_input, list):
            cond_ins_emb = self.get_learned_conditioning(cond_ins_input)
        else:
            cond_ins_emb = self.get_learned_conditioning(
                cond_ins_input.to(self.device))
        # To support classifier-free guidance, randomly drop out only text conditioning
        # 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(b, device=x.device)
        else:
            random_num = torch.ones(b, device=x.device)
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob,
                                "n -> n 1 1")
        null_prompt = self.get_learned_conditioning([""])
        cond_ins_emb = torch.where(prompt_mask, null_prompt,
                                   cond_ins_emb.detach())

        # Get conditioning frames
        cond_frame_index = 0
        img = obs[:, :, -1, ...]
        input_mask = 1 - rearrange(
            (random_num >= self.uncond_prob).float() *
            (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        cond_img = input_mask * img
        cond_img_emb = self.embedder(cond_img)
        cond_img_emb = self.image_proj_model(cond_img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            else:
                img_cat_cond = z[:, :, cond_frame_index, :, :]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond,
                                      'b c t h w -> b c (repeat t) h w',
                                      repeat=z.shape[2])
            cond["c_concat"] = [img_cat_cond]

        cond_action = self.action_projector(action)
        cond_action_emb = self.agent_action_pos_emb + cond_action
        # Get conditioning states
        cond_state = self.state_projector(obs_state)
        cond_state_emb = self.agent_state_pos_emb + cond_state

        if self.decision_making_only:
            is_sim_mode = False
        else:
            is_sim_mode = torch.rand(1) < 0.5
        z, state, cond["c_crossattn"] = self._get_augmented_batch(
            z,
            state,
            cond_state_emb,
            cond_action_emb,
            cond_ins_emb,
            null_prompt,
            cond_img_emb,
            sim_mode=is_sim_mode,
            logging=logging)

        cond["c_crossattn_action"] = [
            obs[:, :, -self.n_obs_steps_acting:],
            state[:, -self.n_obs_steps_acting:], is_sim_mode,
            batch['state_mask'], batch['action_mask']
        ]

        out = [z, action, state, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])
        if return_original_cond:
            out.append(cond_ins_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(fs)
        if return_cond_frame:
            out.append(obs)
        if return_original_input:
            out.append(x)
        if logging:
            out.append(is_sim_mode)
        return out

    @torch.no_grad()
    def log_images(self,
                   batch: Mapping[str, Any],
                   sample: bool = True,
                   ddim_steps: int = 50,
                   ddim_eta: float = 1.0,
                   plot_denoise_rows: bool = False,
                   unconditional_guidance_scale: float = 1.0,
                   mask: Tensor | None = None,
                   **kwargs) -> dict[str, Tensor]:
        """ 
        Log images for LatentVisualDiffusion 

        Args:
            batch: Batch mapping used to form inputs/conditions.
            sample: If True, also run sampling for visualization.
            ddim_steps: Number of DDIM steps when using DDIM.
            ddim_eta: DDIM eta parameter (stochasticity).
            plot_denoise_rows: If True, include denoise progression grid.
            unconditional_guidance_scale: Guidance scale for CFG sampling.
            mask: Optional mask for sampling-time inpainting.

        Returns:
            Dict of visualization tensors (images/actions/states/progress).
        """

        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 1
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()

        z, act, state, c, xrec, xc, fs, cond_x, is_sim_mode = self.get_batch_input(
            batch,
            random_uncond=False,
            return_first_stage_outputs=True,
            return_original_cond=True,
            return_fs=True,
            return_cond_frame=True,
            logging=True)

        kwargs['x_start'] = z

        N = xrec.shape[0]
        log["image_condition"] = cond_x
        log["reconst"] = xrec
        if is_sim_mode:
            xc = ["NULL"]
        xc_with_fs = []
        for idx, content in enumerate(xc):
            xc_with_fs.append(content + '_fs=' + str(fs[idx].item()))
        log['instruction'] = xc
        log["condition"] = xc_with_fs
        kwargs.update({"fs": fs.long()})

        if sample:
            uc = None
            with self.ema_scope("Plotting"):
                samples, action_samples, state_samples, z_denoise_row = self.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uc,
                    x0=z,
                    **kwargs)

            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            # Log actions
            mb, mt, _ = batch['action_mask'].shape
            act_mask = batch['action_mask'] == 1.0
            action_target = act[act_mask].reshape(mb, mt, -1)
            action_samples = action_samples[act_mask].reshape(mb, mt, -1)
            log["action"] = torch.cat((action_target, action_samples), dim=0)

            # Log states
            mb, mt, _ = batch['state_mask'].shape
            state_mask = batch['state_mask'] == 1.0
            state_target = state[state_mask].reshape(mb, mt, -1)
            state_samples = state_samples[state_mask].reshape(mb, mt, -1)
            log["state"] = torch.cat((state_target, state_samples), dim=0)

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        log["video_idx"] = batch["path"][0].split('/')[-1][:-4]
        return log

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        params = [
            param for name, param in self.model.named_parameters()
            if not name.startswith("diffusion_model.action_unet")
            and not name.startswith("diffusion_model.state_unet")
        ]
        params_unet_head = list(
            self.model.diffusion_model.action_unet.parameters()) + list(
                self.model.diffusion_model.state_unet.parameters())

        mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

        if self.cond_stage_trainable:
            params_cond_stage = [
                p for p in self.cond_stage_model.parameters()
                if p.requires_grad == True
            ]
            mainlogger.info(
                f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model."
            )
            params.extend(params_cond_stage)

        if self.image_proj_model_trainable:
            mainlogger.info(
                f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model."
            )
            params.extend(list(self.image_proj_model.parameters()))

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        params_group = [{
            'params': params,
            'lr': lr
        }, {
            'params':
            params_unet_head,
            'lr':
            self.dp_optimizer_config['params']['lr'],
            'betas':
            self.dp_optimizer_config['params']['betas'],
            'eps':
            self.dp_optimizer_config['params']['eps'],
            'weight_decay':
            self.dp_optimizer_config['params']['weight_decay']
        }]
        optimizer = torch.optim.AdamW(params_group, lr=lr)

        if self.use_scheduler:

            # mainlogger.info("Setting up scheduler...")
            lr_scheduler = get_scheduler(
                self.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.lr_warmup_steps,
                num_training_steps=(self.datasets_len * self.num_epochs) //
                self.gradient_accumulate_every,  # 50 is handcode
                last_epoch=-1)

            scheduler = SelectiveLRScheduler(
                optimizer=optimizer,
                base_scheduler=lr_scheduler,
                group_indices=[1],
                default_lr=[lr, self.dp_optimizer_config['params']['lr']])
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

        return [optimizer]


class DiffusionWrapper(pl.LightningModule):
    """Thin wrapper that routes inputs/conditions to the underlying diffusion model."""

    def __init__(self, diff_model_config: OmegaConf,
                 conditioning_key: str | None) -> None:
        """
        Args:
            diff_model_config: OmegaConf describing the inner diffusion model to instantiate.
            conditioning_key: How conditioning is applied.
        """
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(
        self,
        x: Tensor,
        x_action: Tensor | None,
        x_state: Tensor | None,
        t: Tensor,
        c_concat: Sequence[Tensor] | None = None,
        c_crossattn: Sequence[Tensor] | None = None,
        c_crossattn_action: list[Any] | None = None,
        c_adm: Tensor | None = None,
        s: Tensor | None = None,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Route input(s) and condition(s) into the inner diffusion model based on `conditioning_key`.

        Args:
            x: Primary input tensor (e.g., latent/image) at timestep `t`.
            x_action: Action stream tensor (used by 'hybrid' variants).
            x_state: State stream tensor (used by 'hybrid' variants).
            t: Timestep indices (B,).
            c_concat: List of tensors to be concatenated channel-wise with `x` (for 'concat' / 'hybrid' modes).
            c_crossattn: List of context tensors concatenated along sequence/channel dim for cross-attention.
            c_crossattn_action: Mixed list used by action/state heads.
            c_adm: Class/ADM conditioning (e.g., labels) when required by '*adm*' modes.
            s: Optional additional time-like / scalar conditioning (e.g., fps/frame-stride) for '*time*' modes.
            mask: Optional spatial/temporal mask (e.g., inpainting) for '*mask*' modes.
            **kwargs: Extra keyword arguments forwarded to the inner diffusion model.

        Returns:
            Output from the inner diffusion model (tensor or tuple, depending on the model).
        """

        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            cc_action = c_crossattn_action
            out = self.diffusion_model(xc,
                                       x_action,
                                       x_state,
                                       t,
                                       context=cc,
                                       context_action=cc_action,
                                       **kwargs)
        elif self.conditioning_key == 'resblockcond':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm, **kwargs)
        elif self.conditioning_key == 'hybrid-time':
            assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s)
        elif self.conditioning_key == 'concat-time-mask':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, context=None, s=s, mask=mask)
        elif self.conditioning_key == 'concat-adm-mask':
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=None, y=s, mask=mask)
        elif self.conditioning_key == 'hybrid-adm-mask':
            cc = torch.cat(c_crossattn, 1)
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=cc, y=s, mask=mask)
        elif self.conditioning_key == 'hybrid-time-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s, y=c_adm)
        elif self.conditioning_key == 'crossattn-adm':
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        else:
            raise NotImplementedError()

        return out
