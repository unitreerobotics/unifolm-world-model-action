import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from queue import Empty

import mujoco
import mujoco.viewer
import numpy as np

from unitree_deploy.utils.rich_logger import log_info, log_success


@dataclass
class MujocoSimulationConfig:
    xml_path: str
    dof: int
    robot_type: str
    ctr_dof: int
    stop_dof: int


def get_mujoco_sim_config(robot_type: str) -> MujocoSimulationConfig:
    if robot_type == "g1":
        return MujocoSimulationConfig(
            xml_path="unitree_deploy/robot_devices/assets/g1/g1_body29.xml",
            dof=30,
            robot_type="g1",
            ctr_dof=14,
            stop_dof=35,
        )
    elif robot_type == "z1":
        return MujocoSimulationConfig(
            xml_path="unitree_deploy/robot_devices/assets/z1/z1.xml",
            dof=6,
            robot_type="z1",
            ctr_dof=6,
            stop_dof=6,
        )
    elif robot_type == "h1_2":
        return MujocoSimulationConfig(
            xml_path="unitree_deploy/robot_devices/assets/z1/z1.urdf",
            dof=30,
            robot_type="g1",
            ctr_dof=14,
            stop_dof=35,
        )
    else:
        raise ValueError(f"Unsupported robot_type: {robot_type}")


class MujicoSimulation:
    def __init__(self, config: MujocoSimulationConfig):
        self.xml_path = config.xml_path

        self.robot_type = config.robot_type

        self.dof = config.dof
        self.ctr_dof = config.ctr_dof
        self.stop_dof = config.stop_dof

        self.action_queue = Queue()
        self.state_queue = Queue()
        self.process = Process(target=self._run_simulation, args=(self.xml_path, self.action_queue, self.state_queue))
        self.process.daemon = True
        self.process.start()

    def set_positions(self, joint_positions: np.ndarray):
        if joint_positions.shape[0] != self.ctr_dof:
            raise ValueError(f"joint_positions must contain {self.ctr_dof} values!")

        if self.robot_type == "g1":
            joint_positions = np.concatenate([np.zeros(self.dof - self.ctr_dof, dtype=np.float32), joint_positions])
        elif self.robot_type == "z1":
            pass
        elif self.robot_type == "h1_2":
            joint_positions[: self.dof - self.ctr_dof] = 0.0
        else:
            raise ValueError(f"Unsupported robot_type: {self.robot_type}")

        self.action_queue.put(joint_positions.tolist())

    def get_current_positions(self, timeout=0.01):
        try:
            return self.state_queue.get(timeout=timeout)
        except Empty:
            return [0.0] * self.stop_dof

    def stop(self):
        if hasattr(self, "process") and self.process is not None and self.process.is_alive():
            try:
                self.process.terminate()
                self.process.join()
            except Exception as e:
                print(f"[WARN] Failed to stop process: {e}")
            self.process = None

        for qname in ["action_queue", "state_queue"]:
            queue = getattr(self, qname, None)
            if queue is not None:
                try:
                    if hasattr(queue, "close") and callable(queue.close):
                        queue.close()
                    if hasattr(queue, "join_thread") and callable(queue.join_thread):
                        queue.join_thread()
                except Exception as e:
                    print(f"[WARN] Failed to cleanup {qname}: {e}")
                setattr(self, qname, None)

    def __del__(self):
        self.stop()

    @staticmethod
    def _run_simulation(xml_path: str, action_queue: Queue, state_queue: Queue):
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        joints_indices = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names
        ]
        log_info(f"len joints indices: {len(joints_indices)}")

        viewer = mujoco.viewer.launch_passive(model, data)

        current_positions = np.zeros(len(joints_indices), dtype=np.float32)
        try:
            while viewer.is_running():
                try:
                    new_pos = action_queue.get_nowait()
                    if len(new_pos) == len(joints_indices):
                        current_positions = new_pos
                except Empty:
                    pass

                for idx, pos in zip(joints_indices, current_positions, strict=True):
                    data.qpos[idx] = pos

                data.qvel[:] = 0
                mujoco.mj_forward(model, data)

                state_queue.put(data.qpos.copy())

                viewer.sync()
                time.sleep(0.001)

        except KeyboardInterrupt:
            log_success("The simulation process was interrupted.")
        finally:
            viewer.close()


def main():
    config = get_mujoco_sim_config(robot_type="g1")
    sim = MujicoSimulation(config)
    time.sleep(1)  # Allow time for the simulation to start
    try:
        while True:
            positions = np.random.uniform(-1.0, 1.0, sim.ctr_dof)

            sim.set_positions(positions)

            # print(sim.get_current_positions())

            time.sleep(1 / 50)
    except KeyboardInterrupt:
        print("Simulation stopped.")
        sim.stop()


if __name__ == "__main__":
    main()
