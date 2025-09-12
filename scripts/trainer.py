import argparse, os, datetime
import pytorch_lightning as pl
import torch

from omegaconf import OmegaConf
from transformers import logging as transf_logging
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from unifolm_wma.utils.utils import instantiate_from_config
from unifolm_wma.utils.train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from unifolm_wma.utils.train import set_logger, init_workspace, load_checkpoints, get_num_parameters


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--seed",
                        "-s",
                        type=int,
                        default=20250912,
                        help="seed for seed_everything")
    parser.add_argument("--name",
                        "-n",
                        type=str,
                        default="",
                        help="experiment name, as saving folder")
    parser.add_argument(
        "--base",
        "-b",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right.",
        default=list())
    parser.add_argument("--train",
                        "-t",
                        action='store_true',
                        default=False,
                        help='train')
    parser.add_argument("--val",
                        "-v",
                        action='store_true',
                        default=False,
                        help='val')
    parser.add_argument("--test",
                        action='store_true',
                        default=False,
                        help='test')
    parser.add_argument("--logdir",
                        "-l",
                        type=str,
                        default="logs",
                        help="directory for logging dat shit")
    parser.add_argument("--auto_resume",
                        action='store_true',
                        default=False,
                        help="resume from full-info checkpoint")
    parser.add_argument("--auto_resume_weight_only",
                        action='store_true',
                        default=False,
                        help="resume from weight-only checkpoint")
    parser.add_argument("--debug",
                        "-d",
                        action='store_true',
                        default=False,
                        help="enable post-mortem debugging")
    return parser


def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args)
                  if getattr(args, k) != getattr(default_trainer_args, k))


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))
    num_rank = int(os.environ.get('WORLD_SIZE'))

    parser = get_parser()
    # Extends existing argparse by default Trainer attributes
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    # Setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir,
                                                       config,
                                                       lightning_config,
                                                       global_rank)
    logger = set_logger(
        logfile=os.path.join(loginfo, 'log_%d:%s.txt' % (global_rank, now)))
    logger.info("@lightning version: %s [>=1.8 required]" % (pl.__version__))
    logger.info("***** Configing Model *****")
    config.model.params.logdir = workdir
    model = instantiate_from_config(config.model)
    # Load checkpoints
    model = load_checkpoints(model, config.model)

    # Register_schedule again to make ZTSNR work
    if model.rescale_betas_zero_snr:
        model.register_schedule(given_betas=model.given_betas,
                                beta_schedule=model.beta_schedule,
                                timesteps=model.timesteps,
                                linear_start=model.linear_start,
                                linear_end=model.linear_end,
                                cosine_s=model.cosine_s)

    # Update trainer config
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)

    num_nodes = trainer_config.num_nodes
    ngpu_per_node = trainer_config.devices
    logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")

    # Setup learning rate
    base_lr = config.model.base_learning_rate
    bs = config.data.params.batch_size
    if getattr(config.model, 'scale_lr', True):
        model.learning_rate = num_rank * bs * base_lr
    else:
        model.learning_rate = base_lr

    logger.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.train_datasets:
        logger.info(
            f"{k}, {data.train_datasets[k].__class__.__name__}, {len(data.train_datasets[k])}"
        )
    if hasattr(data, 'val_datasets'):
        for k in data.val_datasets:
            logger.info(
                f"{k}, {data.val_datasets[k].__class__.__name__}, {len(data.val_datasets[k])}"
            )

    for item in unknown:
        if item.startswith('--total_gpus'):
            num_gpus = int(item.split('=')[-1])
            break
    model.datasets_len = len(data)

    logger.info("***** Configing Trainer *****")
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"

    # Setup trainer args: pl-logger and callbacks
    trainer_kwargs = dict()
    trainer_kwargs["num_sanity_val_steps"] = 0
    logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # Setup callbacks
    callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir,
                                          ckptdir, logger)
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]
    strategy_cfg = get_trainer_strategy(lightning_config)
    trainer_kwargs["strategy"] = strategy_cfg if type(
        strategy_cfg) == str else instantiate_from_config(strategy_cfg)
    trainer_kwargs['precision'] = lightning_config.get('precision', 32)
    trainer_kwargs["sync_batchnorm"] = False

    # Trainer config: others
    trainer_args = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_args, **trainer_kwargs)

    # Allow checkpointing via USR1
    def melk(*args, **kwargs):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # List the key model sizes
    total_params = get_num_parameters(model)

    logger.info("***** Running the Loop *****")
    if args.train:
        try:
            if "strategy" in lightning_config and lightning_config[
                    'strategy'].startswith('deepspeed'):
                logger.info("<Training in DeepSpeed Mode>")
                if trainer_kwargs['precision'] == 16:
                    with torch.cuda.amp.autocast():
                        trainer.fit(model, data)
                else:
                    trainer.fit(model, data)
            else:
                logger.info("<Training in DDPSharded Mode>")
                trainer.fit(model, data)
        except Exception:
            raise
