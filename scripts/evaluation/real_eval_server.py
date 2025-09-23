import argparse, os, sys
import torch
import torchvision
import warnings
import imageio
import logging
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import traceback
import uvicorn

from omegaconf import OmegaConf
from einops import rearrange, repeat
from collections import OrderedDict
from pytorch_lightning import seed_everything
from torch import nn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime

from unifolm_wma.utils.utils import instantiate_from_config
from unifolm_wma.models.samplers.ddim import DDIMSampler


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Args:
        module (nn.Module): PyTorch module.

    Returns:
        torch.device: The device where the module's parameters are stored.
    """
    return next(iter(module.parameters())).device


def load_model_checkpoint(model: nn.Module, ckpt: str) -> nn.Module:
    """Load model weights from checkpoint file.

    Args:
        model (nn.Module): Model to load weights into.
        ckpt (str): Path to checkpoint file.

    Returns:
        nn.Module: Model with loaded weights.
    """

    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=False)
        except:
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=False)
    else:
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]] = state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def write_video(video_path: str, stacked_frames: List[Any], fps: int) -> None:
    """Write a video to disk using imageio.

    Args:
        video_path (str): Path to save the video.
        stacked_frames (List[Any]): Frames to write.
        fps (int): Frames per second.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                "pkg_resources is deprecated as an API",
                                category=DeprecationWarning)
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def save_results(video: torch.Tensor, filename: str, fps: int = 8) -> None:
    """Save a video tensor as an MP4 file.

    Args:
        video (torch.Tensor): Video tensor of shape (B, C, T, H, W).
        filename (str): Path to save video.
        fps (int, optional): Frame rate. Defaults to 8.

    """
    video = video.detach().cpu()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4)

    frame_grids = [
        torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0)
        for framesheet in video
    ]
    grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    torchvision.io.write_video(filename,
                               grid,
                               fps=fps,
                               video_codec='h264',
                               options={'crf': '10'})


def get_latent_z(model: nn.Module, videos: torch.Tensor) -> torch.Tensor:
    """Encode videos into latent space.

    Args:
        model (nn.Module): Model with `encode_first_stage` method.
        videos (torch.Tensor): Input videos (B, C, T, H, W).

    Returns:
        torch.Tensor: Latent representation (B, C, T, H, W).

    """
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(
        model: torch.nn.Module,
        prompts: list[str],
        observation: Dict[str, torch.Tensor],
        noise_shape: tuple[int, int, int, int, int],
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        unconditional_guidance_scale: float = 1.0,
        fs: int | None = None,
        timestep_spacing: str = 'uniform',
        guidance_rescale: float = 0.0,
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run inference with DDIM sampling.

    Args:
        model (nn.Module): Diffusion model.
        prompts (Any): Conditioning text prompts.
        observation (Dict[str, torch.Tensor]): Observation dictionary.
        noise_shape (List[int]): Shape of noise tensor.
        ddim_steps (int, optional): Number of DDIM steps. Defaults to 50.
        ddim_eta (float, optional): Sampling eta. Defaults to 1.0.
        unconditional_guidance_scale (float, optional): Guidance scale. Defaults to 1.0.
        fs (Optional[int], optional): Frame stride or FPS. Defaults to None.
        timestep_spacing (str, optional): Spacing strategy. Defaults to "uniform".
        guidance_rescale (float, optional): Guidance rescale. Defaults to 0.0.
        **kwargs (Any): Additional arguments.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    b, _, t, _, _ = noise_shape
    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    img = observation['observation.images.top']
    cond_img = img[:, -1, ...]
    cond_img_emb = model.embedder(cond_img)
    cond_img_emb = model.image_proj_model(cond_img_emb)

    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, img.permute(0, 2, 1, 3, 4))
        img_cat_cond = z[:, :, -1:, :, :]
        img_cat_cond = repeat(img_cat_cond,
                              'b c t h w -> b c (repeat t) h w',
                              repeat=noise_shape[2])
        cond = {"c_concat": [img_cat_cond]}

    cond_ins_emb = model.get_learned_conditioning(prompts)
    cond_state = model.state_projector(observation['observation.state'])
    cond_state_emb = model.agent_state_pos_emb + cond_state

    cond_action = model.action_projector(observation['action'])
    cond_action_emb = model.agent_action_pos_emb + cond_action
    cond_action_emb = torch.zeros_like(cond_action_emb)

    cond["c_crossattn"] = [
        torch.cat([cond_state_emb, cond_ins_emb, cond_img_emb], dim=1)
    ]
    cond["c_crossattn_action"] = [
        observation['observation.images.top'].permute(
            0, 2, 1, 3, 4)[:, :, -model.n_obs_steps_acting:],
        observation['observation.state'][:, -model.n_obs_steps_acting:]
    ]

    uc = None

    kwargs.update({"unconditional_conditioning_img_nonetext": None})

    cond_mask = None
    cond_z0 = None

    if ddim_sampler is not None:

        samples, actions, states, intermedia = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            cfg_img=None,
            mask=cond_mask,
            x0=cond_z0,
            fs=fs,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            **kwargs)

        # Reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants = batch_images

    return batch_variants, actions, states


def run_inference(args: argparse.Namespace, gpu_num: int,
                  gpu_no: int) -> Tuple[nn.Module, List[int], Any]:
    """
    Run inference pipeline on prompts and image inputs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        gpu_num (int): Number of GPUs.
        gpu_no (int): Index of the current GPU.

    Returns:
        None
    """
    # Load config
    config = OmegaConf.load(args.config)
    # Set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    config['model']['params']['wma_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(config.model)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model = model.cuda(gpu_no)
    model.eval()
    print(">>> Model is successfully loaded ...")

    # Build unnomalizer
    logging.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    print(">>> Dataset is successfully loaded ...")

    ## Run over data
    assert (args.height % 16 == 0) and (
        args.width % 16
        == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"

    ## Get latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'>>> Generate {n_frames} frames under each generation ...')
    noise_shape = [args.bs, channels, n_frames, h, w]

    return model, noise_shape, data


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir",
                        type=str,
                        default=None,
                        help="Path to save the results.")
    parser.add_argument("--ckpt_path",
                        type=str,
                        default=None,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps. If non-positive, DDPM is used instead.")
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="Eta for DDIM sampling. Set to 0.0 for deterministic results.")
    parser.add_argument("--bs",
                        type=int,
                        default=1,
                        help="Batch size for inference. Must be 1.")
    parser.add_argument("--height",
                        type=int,
                        default=320,
                        help="Height of the generated images in pixels.")
    parser.add_argument("--width",
                        type=int,
                        default=512,
                        help="Width of the generated images in pixels.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=3,
        help=
        "frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)"
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=1.0,
        help="Scale for classifier-free guidance during sampling.")
    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Random seed for reproducibility.")
    parser.add_argument("--video_length",
                        type=int,
                        default=16,
                        help="Number of frames in the generated video.")
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="uniform",
        help=
        "Strategy for timestep scaling. See Table 2 in the paper: 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891)."
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help=
        "Rescale factor for guidance as discussed in 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891)."
    )
    parser.add_argument(
        "--perframe_ae",
        action='store_true',
        default=False,
        help=
        "Use per-frame autoencoder decoding to reduce GPU memory usage. Recommended for models with resolutions like 576x1024."
    )
    return parser


class Server:

    def __init__(self, args: argparse.Namespace) -> None:
        self.model_, self.noise_shape_, self.data_ = run_inference(args, 1, 0)
        self.args_ = args
        self.dataset_name = self.data_.dataset_configs['test']['params'][
            'dataset_name']
        self.device_ = get_device_from_parameters(self.model_)

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        return (image / 255 - 0.5) * 2

    def predict_action(self, payload: Dict[str, Any]) -> Any:
        try:
            images = payload['observation.images.top']
            states = payload['observation.state']
            actions = payload['action']  # Should be all zeros
            language_instruction = payload['language_instruction']

            images = torch.tensor(images).cuda()
            images = self.data_.test_datasets[
                self.dataset_name].spatial_transform(images).unsqueeze(0)
            images = self.normalize_image(images)
            print(f"images shape: {images.shape} ...")
            states = torch.tensor(states)
            states = self.data_.test_datasets[self.dataset_name].normalizer(
                {'observation.state': states})['observation.state']
            states, _ = self.data_.test_datasets[
                self.dataset_name]._map_to_uni_state(states, "joint position")
            print(f"states shape: {states.shape} ...")
            actions = torch.tensor(actions)
            actions, action_mask = self.data_.test_datasets[
                self.dataset_name]._map_to_uni_action(actions,
                                                      "joint position")
            print(f"actions shape: {actions.shape} ...")
            print("=" * 20)
            states = states.unsqueeze(0).cuda()
            actions = actions.unsqueeze(0).cuda()

            observation = {
                'observation.images.top': images,
                'observation.state': states,
                'action': actions
            }
            observation = {
                key: observation[key].to(self.device_, non_blocking=True)
                for key in observation
            }

            args = self.args_
            pred_videos, pred_action, _ = image_guided_synthesis(
                self.model_,
                language_instruction,
                observation,
                self.noise_shape_,
                ddim_steps=args.ddim_steps,
                ddim_ets=args.ddim_eta,
                unconditional_guidance_scale=args.unconditional_guidance_scale,
                fs=30 / args.frame_stride,
                timestep_spacing=args.timestep_spacing,
                guidance_rescale=args.guidance_rescale)

            pred_action = pred_action[..., action_mask[0] == 1.0][0].cpu()
            pred_action = self.data_.test_datasets[
                self.dataset_name].unnormalizer({'action':
                                                 pred_action})['action']

            os.makedirs(args.savedir, exist_ok=True)
            current_time = datetime.now().strftime("%H:%M:%S")
            video_file = f'{args.savedir}/{current_time}.mp4'
            save_results(pred_videos.cpu(), video_file)

            response = {
                'result': 'ok',
                'action': pred_action.tolist(),
                'desc': 'success'
            }
            return JSONResponse(response)

        except:
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions.")
            return {'result': 'error', 'desc': traceback.format_exc()}

    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/predict_action")(self.predict_action)
        print(">>> Inference server is ready ... ")
        uvicorn.run(self.app, host=host, port=port)
        print(">>> Inference server stops ... ")
        return


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)
    rank, gpu_num = 0, 1
    print(">>> Launch inference server ... ")
    server = Server(args)
    server.run()
