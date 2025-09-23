import argparse
import os
import time
import cv2
import numpy as np
import torch
import tqdm

from typing import Any, Deque, MutableMapping, OrderedDict
from collections import deque
from pathlib import Path

from unitree_deploy.real_unitree_env import make_real_env
from unitree_deploy.utils.eval_utils import (
    ACTTemporalEnsembler,
    LongConnectionClient,
    populate_queues,
)

# -----------------------------------------------------------------------------
# Network & environment defaults
# -----------------------------------------------------------------------------
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"

# fmt: off
INIT_POSE = {
    'g1_dex1': np.array([0.10559805, 0.02726714, -0.01210221, -0.33341318, -0.22513399, -0.02627627, -0.15437093,  0.1273793 , -0.1674708 , -0.11544029, -0.40095493,  0.44332668,  0.11566751,  0.3936641, 5.4, 5.4], dtype=np.float32),
    'z1_dual_dex1_realsense': np.array([-1.0262332,  1.4281361, -1.2149128,  0.6473399, -0.12425245, 0.44945636,  0.89584476,  1.2593982, -1.0737865,  0.6672816, 0.39730102, -0.47400007, 0.9894176, 0.9817477 ], dtype=np.float32),
    'z1_realsense': np.array([-0.06940782, 1.4751548, -0.7554075, 1.0501366, 0.02931615, -0.02810347, -0.99238837], dtype=np.float32),
}
ZERO_ACTION = {
    'g1_dex1': torch.zeros(16, dtype=torch.float32),
    'z1_dual_dex1_realsense': torch.zeros(14, dtype=torch.float32),
    'z1_realsense': torch.zeros(7, dtype=torch.float32),
}
CAM_KEY = {
    'g1_dex1': 'cam_right_high',
    'z1_dual_dex1_realsense': 'cam_high',
    'z1_realsense': 'cam_high',
}
# fmt: on


def prepare_observation(args: argparse.Namespace, obs: Any) -> OrderedDict:
    """
    Convert a raw env observation into the model's expected input dict.
    """
    rgb_image = cv2.cvtColor(
        obs.observation["images"][CAM_KEY[args.robot_type]], cv2.COLOR_BGR2RGB)
    observation = {
        "observation.images.top":
        torch.from_numpy(rgb_image).permute(2, 0, 1),
        "observation.state":
        torch.from_numpy(obs.observation["qpos"]),
        "action": ZERO_ACTION[args.robot_type],
    }
    return OrderedDict(observation)


def run_policy(
    args: argparse.Namespace,
    env: Any,
    client: LongConnectionClient,
    temporal_ensembler: ACTTemporalEnsembler,
    cond_obs_queues: MutableMapping[str, Deque[torch.Tensor]],
    output_dir: Path,
) -> None:
    """
    Single rollout loop:
        1) warm start the robot,
        2) stream observations,
        3) fetch actions from the policy server,
        4) execute with temporal ensembling for smoother control.
    """

    _ = env.step(INIT_POSE[args.robot_type])
    time.sleep(2.0)
    t = 0

    while True:
        # Gapture observation
        obs = env.get_observation(t)
        # Format observation
        obs = prepare_observation(args, obs)
        cond_obs_queues = populate_queues(cond_obs_queues, obs)
        # Call server to get actions
        pred_actions = client.predict_action(args.language_instruction,
                                             cond_obs_queues).unsqueeze(0)
        # Keep only the next horizon of actions and apply temporal ensemble smoothing
        actions = temporal_ensembler.update(
            pred_actions[:, :args.action_horizon])[0]

        # Execute the actions
        for n in range(args.exe_steps):
            action = actions[n].cpu().numpy()
            print(f">>> Exec => step {n} action: {action}", flush=True)
            print("---------------------------------------------")

            # Maintain real-time loop at `control_freq` Hz
            t1 = time.time()
            obs = env.step(action)
            time.sleep(max(0, 1 / args.control_freq - time.time() + t1))
            t += 1

            # Prime the queue for the next action step (except after the last one in this chunk)
            if n < args.exe_steps - 1:
                obs = prepare_observation(args, obs)
                cond_obs_queues = populate_queues(cond_obs_queues, obs)


def run_eval(args: argparse.Namespace) -> None:
    client = LongConnectionClient(BASE_URL)

    # Initialize ACT temporal moving-averge smoother
    temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff=0.01,
                                              chunk_size=args.action_horizon,
                                              exe_steps=args.exe_steps)
    temporal_ensembler.reset()

    # Initialize observation and action horizon queue
    cond_obs_queues = {
        "observation.images.top": deque(maxlen=args.observation_horizon),
        "observation.state": deque(maxlen=args.observation_horizon),
        "action": deque(
            maxlen=16),  # NOTE: HAND CODE AS THE MODEL PREDCIT FUTURE 16 STEPS
    }

    env = make_real_env(
        robot_type=args.robot_type,
        dt=1 / args.control_freq,
    )
    env.connect()

    try:
        for episode_idx in tqdm.tqdm(range(0, args.num_rollouts_planned)):
            output_dir = Path(args.output_dir) / f"episode_{episode_idx:03d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            run_policy(args, env, client, temporal_ensembler, cond_obs_queues,
                       output_dir)
    finally:
        env.close()
    env.close()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_type",
                        type=str,
                        default="g1_dex1",
                        help="The type of the robot embodiment.")
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=16,
        help="Number of future actions, predicted by the policy, to keep",
    )
    parser.add_argument(
        "--exe_steps",
        type=int,
        default=16,
        help=
        "Number of future actions to execute, which must be less than the above action horizon.",
    )
    parser.add_argument(
        "--observation_horizon",
        type=int,
        default=2,
        help="Number of most recent frames/states to consider.",
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default="Pack black camera into box",
        help="The language instruction provided to the policy server.",
    )
    parser.add_argument("--num_rollouts_planned",
                        type=int,
                        default=10,
                        help="The number of rollouts to run.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./results",
                        help="The directory for saving results.")
    parser.add_argument("--control_freq",
                        type=float,
                        default=30,
                        help="The Low-level control frequency in Hz.")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run_eval(args)
