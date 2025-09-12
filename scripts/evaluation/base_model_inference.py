import argparse, os, glob
import datetime, time
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import random

from pytorch_lightning import seed_everything
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

from unifolm_wma.models.samplers.ddim import DDIMSampler
from unifolm_wma.utils.utils import instantiate_from_config


def get_filelist(data_dir: str, postfixes: list[str]) -> list[str]:
    """
    Get list of files in `data_dir` with extensions in `postfixes`.

    Args:
        data_dir (str): Directory path.
        postfixes (list[str]): List of file extensions (e.g., ['csv', 'jpg']).

    Returns:
        list[str]: Sorted list of matched file paths.
    """
    patterns = [
        os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes
    ]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def load_model_checkpoint(model: torch.nn.Module,
                          ckpt: str) -> torch.nn.Module:
    """
    Load model weights from checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        ckpt (str): Path to the checkpoint file.

    Returns:
        torch.nn.Module: Model with weights loaded.
    """
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            loaded = model.load_state_dict(state_dict, strict=False)
            print("Missing keys:")
            for k in loaded.missing_keys:
                print(f"  {k}")
            print("Unexpected keys:")
            for k in loaded.unexpected_keys:
                print(f"  {k}")

        except:
            # Rename the keys for 256x256 model
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


def load_prompts(prompt_file: str) -> list[str]:
    """
    Load prompts from a text file, one per line.

    Args:
        prompt_file (str): Path to the prompt file.

    Returns:
        list[str]: List of prompt strings.
    """
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_data_prompts(
    data_dir: str,
    savedir: str,
    video_size: tuple[int, int] = (256, 256),
    video_frames: int = 16
) -> tuple[list[str], list[torch.Tensor], list[str], list[float], list[float],
           list[int]]:
    """
    Load image prompts, process them into video format, and retrieve metadata.

    Args:
        data_dir (str): Directory containing images and CSV file.
        savedir (str): Output directory to check if inference was already done.
        video_size (tuple[int, int], optional): Target size of video frames.
        video_frames (int, optional): Number of frames in each video.

    Returns:
        tuple: (filenames, video tensors, prompts, fps values, fs values, num_generations)
    """

    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Load prompt csv
    prompt_file = get_filelist(data_dir, ['csv'])
    assert len(prompt_file) > 0, "Error: found NO image prompt file!"

    # Load image prompts
    file_list = get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    data_list = []
    filename_list = []
    prompt_list = []
    fps_list = []
    fs_list = []
    num_gen_list = []
    prompt_csv = pd.read_csv(prompt_file[0])
    n_samples = len(file_list)

    for idx in range(n_samples):
        image = Image.open(file_list[idx]).convert('RGB')
        image_tensor = transform(image).unsqueeze(1)
        frame_tensor = repeat(image_tensor,
                              'c t h w -> c (repeat t) h w',
                              repeat=video_frames)
        _, filename = os.path.split(file_list[idx])

        if not is_inferenced(savedir, filename):
            video_id = filename[:-4]
            prompt_csv['videoid'] = prompt_csv['videoid'].map(str)
            if not (prompt_csv['videoid'] == video_id).any():
                continue
            data_list.append(frame_tensor)
            filename_list.append(filename)
            ins = prompt_csv[prompt_csv['videoid'] ==
                             video_id]['instruction'].values[0]
            prompt_list.append(ins)
            fps = prompt_csv[prompt_csv['videoid'] ==
                             video_id]['fps'].values[0]
            fps_list.append(fps)
            fs = prompt_csv[prompt_csv['videoid'] == video_id]['fs'].values[0]
            fs_list.append(fs)
            num_gen = prompt_csv[prompt_csv['videoid'] ==
                                 video_id]['num_gen'].values[0]
            num_gen_list.append(int(num_gen))

    return filename_list, data_list, prompt_list, fps_list, fs_list, num_gen_list


def is_inferenced(save_dir: str, filename: str) -> bool:
    """
    Check if a result video already exists.

    Args:
        save_dir (str): Directory where results are saved.
        filename (str): Base filename to check.

    Returns:
        bool: True if file exists, else False.
    """
    video_file = os.path.join(save_dir, f"{filename[:-4]}.mp4")
    return os.path.exists(video_file)


def save_results_seperate(prompt: str | list[str],
                          samples: torch.Tensor,
                          filename: str,
                          fakedir: str,
                          fps: int = 8) -> None:
    """
    Save generated video samples as .mp4 files.

    Args:
        prompt (str | list[str]): The prompt text.
        samples (torch.Tensor): Generated video tensor of shape [B, C, T, H, W].
        filename (str): Output filename.
        fakedir (str): Directory to save output videos.
        fps (int, optional): Frames per second.

    Returns:
        None
    """
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    # Save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i, ...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0)
            path = os.path.join(savedirs[idx], f'{filename.split(".")[0]}.mp4')
            torchvision.io.write_video(path,
                                       grid,
                                       fps=fps,
                                       video_codec='h264',
                                       options={'crf': '0'})


def get_latent_z(model: torch.nn.Module, videos: torch.Tensor) -> torch.Tensor:
    """
    Encode videos to latent space.

    Args:
        model (torch.nn.Module): Model with encode_first_stage function.
        videos (torch.Tensor): Video tensor of shape [B, C, T, H, W].

    Returns:
        torch.Tensor: Latent representation of shape [B, C, T, H, W].
    """
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(model: torch.nn.Module,
                           prompts: list[str],
                           videos: torch.Tensor,
                           noise_shape: list[int],
                           ddim_steps: int = 50,
                           ddim_eta: float = 1.0,
                           unconditional_guidance_scale: float = 1.0,
                           fs: int | None = None,
                           text_input: bool = False,
                           timestep_spacing: str = 'uniform',
                           guidance_rescale: float = 0.0,
                           **kwargs) -> torch.Tensor:
    """
    Run DDIM-based image-to-video synthesis with hybrid/text+image guidance.

    Args:
        model (torch.nn.Module): Diffusion model.
        prompts (list[str]): Text prompts.
        videos (torch.Tensor): Input images/videos of shape [B, C, T, H, W].
        noise_shape (list[int]): Latent noise shape [B, C, T, H, W].
        ddim_steps (int, optional): Number of DDIM steps.
        ddim_eta (float, optional): Eta value for DDIM.
        unconditional_guidance_scale (float, optional): Guidance scale.
        fs (int | None, optional): FPS input for sampler.
        text_input (bool, optional): If True, use text guidance.
        timestep_spacing (str, optional): Timestep schedule spacing.
        guidance_rescale (float, optional): Rescale guidance effect.
        **kwargs: Additional sampler args.

    Returns:
        torch.Tensor: Synthesized videos of shape [B, 1, C, T, H, W].
    """

    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""] * batch_size

    b, c, t, h, w = videos.shape
    img = videos[:, :, 0]
    img_emb = model.embedder(img)
    img_emb = model.image_proj_model(img_emb)
    img_emb = rearrange(img_emb, 'b (t l) c -> (b t) l c', t=t)
    cond_emb = model.get_learned_conditioning(prompts)
    cond_emb = cond_emb.repeat_interleave(repeats=t, dim=0)

    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos)
        img_cat_cond = z[:, :, :1, :, :]
        img_cat_cond = repeat(img_cat_cond,
                              'b c t h w -> b c (repeat t) h w',
                              repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond]

    uc = None
    cond_mask = None
    kwargs.update({"unconditional_conditioning_img_nonetext": None})

    batch_variants = []
    if ddim_sampler is not None:
        samples, _, _, _ = ddim_sampler.sample(
            S=ddim_steps,
            batch_size=batch_size,
            shape=noise_shape[1:],
            conditioning=cond,
            eta=ddim_eta,
            mask=cond_mask,
            x0=None,
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            fs=fs,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            **kwargs)

        # Reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)

    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def run_inference(args: argparse.Namespace, gpu_num: int, gpu_no: int) -> None:
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
    config['model']['params']['wma_config']['params'][
        'use_checkpoint'] = False
    model = instantiate_from_config(config.model)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    # Run over data
    assert (args.height % 16 == 0) and (
        args.width % 16
        == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"

    # Get latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'>>> Generate {n_frames} frames under each generation ...')
    noise_shape = [args.bs, channels, n_frames, h, w]

    fakedir = os.path.join(args.savedir, "samples")
    os.makedirs(fakedir, exist_ok=True)

    # Prompt file setting
    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    filename_list, data_list, prompt_list, fps_list, fs_list, num_gen_list = load_data_prompts(
        args.prompt_dir,
        args.savedir,
        video_size=(args.height, args.width),
        video_frames=n_frames)

    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('>>> Prompts testing [rank:%d] %d/%d samples loaded.' %
          (gpu_no, samples_split, num_samples))

    indices = list(range(samples_split * gpu_no, samples_split * (gpu_no + 1)))
    fps_list_rank = [fps_list[i] for i in indices]
    fs_list_rank = [fs_list[i] for i in indices]
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]

    with torch.no_grad(), torch.cuda.amp.autocast():
        # Create a new result csv
        for idx, indice in enumerate(
                tqdm(range(0, len(prompt_list_rank), args.bs),
                     desc=f'Sample batch')):
            fps = fps_list_rank[indice:indice + args.bs]
            fs = fs_list_rank[indice:indice + args.bs]
            prompts = prompt_list_rank[indice:indice + args.bs]
            num_gen = num_gen_list[indice:indice + args.bs]
            videos = data_list_rank[indice:indice + args.bs]
            filenames = filename_list_rank[indice:indice + args.bs]
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to("cuda")
            else:
                videos = videos.unsqueeze(0).to("cuda")

            results = []
            print(
                f">>> {prompts[0]}, frame_stride:{fs[0]}, and {num_gen[0]} generation ..."
            )
            for _ in range(num_gen[0]):
                batch_samples = image_guided_synthesis(
                    model, prompts, videos, noise_shape, args.ddim_steps,
                    args.ddim_eta, args.unconditional_guidance_scale,
                    fps[0] // fs[0], args.text_input, args.timestep_spacing,
                    args.guidance_rescale)
                results.extend(batch_samples)
                videos = repeat(batch_samples[0][:, :, -1, :, :].unsqueeze(2),
                                'b c t h w -> b c (repeat t) h w',
                                repeat=batch_samples[0].shape[2])
            batch_samples = [torch.concat(results, axis=2)]

            # Save each example individually
            for nn, samples in enumerate(batch_samples):
                prompt = prompts[nn]
                filename = filenames[nn]
                save_results_seperate(prompt,
                                      samples,
                                      filename,
                                      fakedir,
                                      fps=8)


def get_parser() -> argparse.ArgumentParser:
    """
    Create and return the argument parser.

    Returns:
        argparse.ArgumentParser: Parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--savedir",
                        type=str,
                        default=None,
                        help="Path to save the results.")
    parser.add_argument("--ckpt_path",
                        type=str,
                        default=None,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config",
                        type=str,
                        help="Path to the YAML configuration file.")
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="Directory containing videos and corresponding prompts.")
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
        "--text_input",
        action='store_true',
        default=False,
        help=
        "Whether to provide a text prompt as input to the image-to-video model."
    )
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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2**31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)
